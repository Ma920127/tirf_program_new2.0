from hmmlearn.hmm import GaussianHMM
import numpy as np
import os
from tqdm import tqdm
import time
import pickle
from scipy.ndimage import uniform_filter1d, median_filter
from scipy.signal import savgol_filter




def _sanitize_fitted_model(model, k, seed_means, min_covar):
    """Make every HMM parameter finite and properly normalised after EM.

    EM can produce NaN parameters when a state is never visited (zero
    posterior throughout the whole dataset).  The chain is:
        dead state → stats['nobs']=0 → M-step means = 0/0 = NaN
        → next E-step log-likelihood = NaN → cascades to all states.

    This function is called on the *best* model (after epoch selection) and
    also after every individual epoch's fit so that conv[] values are
    meaningful for argmax.

    Parameters
    ----------
    model      : fitted GaussianHMM
    k          : number of states
    seed_means : (k, 1) array  — the initial means we seeded EM with
    min_covar  : float         — covariance floor
    """
    ct = model.covariance_type

    # ── 1. startprob ─────────────────────────────────────────────────────────
    sp = np.array(model.startprob_, dtype=float)
    bad = ~(np.isfinite(sp) & (sp >= 0))
    if bad.any():
        sp[bad] = 1.0 / k
    s = sp.sum()
    model.startprob_ = sp / s if s > 0 else np.ones(k) / k

    # ── 2. transmat ──────────────────────────────────────────────────────────
    tm = np.array(model.transmat_, dtype=float)
    row_sums = tm.sum(axis=1, keepdims=True)
    bad_rows = ~(np.isfinite(row_sums.squeeze()) & (row_sums.squeeze() > 0))
    if bad_rows.any():
        print(f'[HMM] {bad_rows.sum()} dead state(s) — transmat rows set to uniform')
        tm[bad_rows] = 1.0 / k
        row_sums[bad_rows] = 1.0
    model.transmat_ = tm / row_sums

    # ── 3. means ─────────────────────────────────────────────────────────────
    m = np.array(model.means_, dtype=float)
    for c in range(k):
        if not np.isfinite(m[c]).all():
            # Fall back to the seeded initial mean for this state
            fallback = seed_means[c] if c < len(seed_means) else seed_means[-1]
            m[c] = fallback
            print(f'[HMM] state {c} means NaN — restored to seed {fallback}')
    model.means_ = m

    # ── 4. covariances ───────────────────────────────────────────────────────
    cov = np.array(model._covars_, dtype=float)
    nf  = m.shape[1]
    if not np.isfinite(cov).all() or (cov <= 0).any():
        safe = max(min_covar, 1e-6)
        if ct == 'tied':
            cov = np.where(np.isfinite(cov) & (cov > 0), cov,
                           np.eye(nf) * safe)
        elif ct == 'full':
            for c in range(k):
                if not (np.isfinite(cov[c]).all() and (cov[c] > 0).all()):
                    cov[c] = np.eye(nf) * safe
        elif ct == 'diag':
            cov = np.where(np.isfinite(cov) & (cov > 0), cov, safe)
        else:  # spherical
            cov = np.where(np.isfinite(cov) & (cov > 0), cov, safe)
        model._covars_ = cov
        print(f'[HMM] covariance had NaN/non-positive — replaced with safe={safe:.2e}')

    return model


CHANNEL_CONFIG = {
    'fret_g': {'npz_key': 'fret_g', 'time_key': 'time_g', 'clip': (0.01, 0.99), 'is_fret': True,  'ylabel': 'FRET'},
    'fret_b': {'npz_key': 'fret_b', 'time_key': 'time_b', 'clip': (0.01, 0.99), 'is_fret': True,  'ylabel': 'FRET'},
    'bb':     {'npz_key': 'bb',     'time_key': 'time_b', 'clip': None,           'is_fret': False, 'ylabel': 'Intensity (BB)'},
    'gg':     {'npz_key': 'gg',     'time_key': 'time_g', 'clip': None,           'is_fret': False, 'ylabel': 'Intensity (GG)'},
    'rr':     {'npz_key': 'rr',     'time_key': 'time_r', 'clip': None,           'is_fret': False, 'ylabel': 'Intensity (RR)'},
}


class HMM_fitter:

    def __init__(self, path):
        self.path = path
        self.N_traces = 0
        self.hd_states = 0
        self.selected = []
        self.filt_Q = []
        self.means = []
        self.channel = 'fret_g'
        self.cfg = CHANNEL_CONFIG['fret_g']

    # ──────────────────────────────────────────────────────────────
    # Smoothing helper
    # ──────────────────────────────────────────────────────────────
    def _smooth_2d(self, data, method, w, polyorder=2):
        """Smooth a 2D array (N × frames) along the time axis."""
        if w <= 1:
            return data
        if method == 'moving':
            return uniform_filter1d(data, size=w, mode='nearest', axis=1)
        elif method == 'median':
            return median_filter(data, size=(1, w), mode='nearest')
        elif method == 'savgol':
            ww = w if w % 2 == 1 else w + 1
            if ww <= polyorder:
                ww = polyorder + 2 if polyorder % 2 == 0 else polyorder + 1
            return savgol_filter(data, window_length=ww, polyorder=polyorder, axis=1)
        elif method == 'strided':
            # Strided reduces length — for 2D use moving avg to keep same length
            return uniform_filter1d(data, size=w, mode='nearest', axis=1)
        return data

    # ──────────────────────────────────────────────────────────────
    # Data loading
    # ──────────────────────────────────────────────────────────────
    def load_traces(self, channel='fret_g'):
        cfg = CHANNEL_CONFIG[channel]
        self.channel = channel
        self.cfg = cfg

        data = np.load(self.path + r'\\data.npz')
        self.Q    = data[cfg['npz_key']]
        self.time = data[cfg['time_key']]
        self.N_traces = self.Q.shape[0]
        self.selected = np.load(self.path + r'\\selected_g.npy').astype(int)

        # Load intensity channels needed for photobleach detection
        self.gg = data['gg']
        self.gr = data['gr']
        self.rr = data['rr']
        self.bb = data['bb']
        self.bg = data['bg']
        return self.Q

    # ──────────────────────────────────────────────────────────────
    # Photobleach mask builder  (moved from app.py)
    # ──────────────────────────────────────────────────────────────
    def build_mask(self, channel, pb_on, thresholds,
                   smooth_method='moving', w=1, polyorder=2):
        """
        Build a photobleach mask from intensity thresholds on smoothed data.

        Parameters
        ----------
        channel      : HMM target channel ('fret_g', 'fret_b', 'gg', 'bb', 'rr')
        pb_on        : bool — if False, returns None (no masking)
        thresholds   : dict with keys:
                         fret_g → 'min_totg', 'use_rr',  'min_rr'
                         fret_b → 'min_totb', 'use_rr_b','min_rr_b'
                         gg/bb/rr → 'min_single'
        smooth_method: smoothing method matching Tools tab selection
        w            : smoothing window
        polyorder    : poly order for savgol

        Returns
        -------
        mask : np.ndarray (N_traces × total_frame, dtype=object)
               0 = usable frame, None = photobleached
               or None if pb_on is False
        """
        if not pb_on:
            return None

        N, F = self.Q.shape
        mask = np.zeros((N, F), dtype=object)

        # Smooth the intensity channels before thresholding
        gg_s = self._smooth_2d(self.gg, smooth_method, w, polyorder)
        gr_s = self._smooth_2d(self.gr, smooth_method, w, polyorder)
        rr_s = self._smooth_2d(self.rr, smooth_method, w, polyorder)
        bb_s = self._smooth_2d(self.bb, smooth_method, w, polyorder)
        bg_s = self._smooth_2d(self.bg, smooth_method, w, polyorder)

        def scan_rightleft(sig, threshold, current_mask):
            out = current_mask.copy()
            for mol in range(min(sig.shape[0], N)):
                for frame in range(min(sig.shape[1], F) - 1, -1, -1):
                    if sig[mol, frame] >= threshold:
                        # last alive frame found — mark everything after as PB
                        out[mol, frame + 1:] = np.nan
                        break
                else:
                    # never found an alive frame — entire trace is PB
                    out[mol, :] = np.nan
            return out

        try:
            if channel == 'fret_g':
                if thresholds.get('min_totg') is not None:
                    mask = scan_rightleft(gg_s + gr_s, float(thresholds['min_totg']), mask)
                if thresholds.get('use_rr') and thresholds.get('min_rr') is not None:
                    mask = scan_rightleft(rr_s, float(thresholds['min_rr']), mask)

            elif channel == 'fret_b':
                if thresholds.get('min_totb') is not None:
                    mask = scan_rightleft(bb_s + bg_s, float(thresholds['min_totb']), mask)
                if thresholds.get('use_rr_b') and thresholds.get('min_rr_b') is not None:
                    mask = scan_rightleft(gg_s, float(thresholds['min_rr_b']), mask)

            elif channel == 'gg':
                if thresholds.get('min_single') is not None:
                    mask = scan_rightleft(gg_s, float(thresholds['min_single']), mask)

            elif channel == 'bb':
                if thresholds.get('min_single') is not None:
                    mask = scan_rightleft(bb_s, float(thresholds['min_single']), mask)

            elif channel == 'rr':
                if thresholds.get('min_single') is not None:
                    mask = scan_rightleft(rr_s, float(thresholds['min_single']), mask)

        except Exception as e:
            print(f'build_mask error: {e}')
            return None

        return mask

    # ──────────────────────────────────────────────────────────────
    # Trace preparation
    # ──────────────────────────────────────────────────────────────
    def process_Q(self, w=10, mask=None, hd_states=None,
                  smooth_method='moving', polyorder=2):
        """Prepare concatenated trace data and per-trace lengths for hmmlearn.

        ``length[i]`` = number of usable (non-PB) frames for selected trace i.

        Priority for determining the PB boundary
        -----------------------------------------
        1. ``hd_states`` (the live ``hmm_states[channel]`` 2-D object array from
           the app) — reflects the PB filter **and** any manual slidebar edits.
           A frame is photobleached when its value is ``np.nan``.
        2. ``mask`` (the raw ``hmm_mask`` built by ``build_mask``) — used as
           fallback when ``hd_states`` is not available.
        3. Full trace length — when neither source is available.
        """
        Q = self._smooth_2d(self.Q, smooth_method, w, polyorder)

        # For strided, Q length is already kept the same (we use moving avg),
        # but if truly strided we note the stride factor for mask index scaling.
        stride = w if smooth_method == 'strided' else 1

        if self.cfg['clip'] is not None:
            Q = np.clip(Q, *self.cfg['clip'])

        N_traces = Q.shape[0]
        pro_Q = np.zeros(0)
        length = []
        pro_Q_indices = []   # (trace_index, end_frame) for every included fragment

        for i in range(0, N_traces):
            if self.selected[i] == 1:
                end_frame = Q.shape[1]   # default: use full trace

                # ── Priority 1: hmm_states row (primary source of truth) ──────
                # Counts frames from the start up to (but not including) the
                # first NaN frame, reflecting the PB filter and all manual edits.
                if hd_states is not None and i < len(hd_states):
                    row = hd_states[i]
                    pb_at = len(row)          # assume no PB by default
                    for f in range(len(row)):
                        v = row[f]
                        if v is not None:
                            try:
                                if np.isnan(float(v)):
                                    pb_at = f
                                    break
                            except (TypeError, ValueError):
                                pass
                    end_frame = min(pb_at, Q.shape[1])

                # ── Priority 2: raw hmm_mask ──────────────────────────────────
                # PB frames are np.nan; usable frames are 0.
                elif mask is not None and i < len(mask):
                    for f in range(len(mask[i])):
                        try:
                            if np.isnan(float(mask[i][f])):
                                end_frame = min(f // stride, Q.shape[1])
                                break
                        except (TypeError, ValueError):
                            pass

                frag = Q[i][:end_frame].copy()
                # Replace any NaN / inf so hmmlearn's Cholesky decomposition
                # does not crash.  Use the finite mean of the fragment as fill;
                # fall back to 0.0 when the whole fragment is non-finite.
                finite_mask = np.isfinite(frag)
                if not finite_mask.all():
                    fill = float(frag[finite_mask].mean()) if finite_mask.any() else 0.0
                    frag[~finite_mask] = fill
                if len(frag) > 0:
                    pro_Q = np.concatenate([pro_Q, frag])
                    length.append(len(frag))
                    pro_Q_indices.append((i, end_frame))  # (trace_idx, pb_boundary)

        pro_Q = pro_Q.reshape(-1, 1)
        self.pro_Q = pro_Q
        self.length = length
        self.pro_Q_indices = pro_Q_indices   # used by cal_states and app.py
        return N_traces, length

    # ──────────────────────────────────────────────────────────────
    # Model fitting
    # ──────────────────────────────────────────────────────────────
    def fitHMM(self, r, w=10, means=None, epoch=10,
               covariance_type='spherical', n_iter=20,
               mask=None, hd_states=None,
               smooth_method='moving', polyorder=2,
               sticky_w=1.0):
        """Fit a Gaussian HMM to the pre-processed trace data.

        Parameters
        ----------
        means : array-like or None
            Initial means used to seed EM.  Values are used as the starting
            point only — EM is free to refine them.  Length determines k.
        hd_states : 2-D object array or None
            Live ``hmm_states[channel]`` from the app (rows = molecules,
            cols = frames; NaN = photobleached).  When supplied this is used
            as the **primary** source for per-trace lengths (number of
            non-PB frames), overriding ``mask``.
        mask : 2-D object array or None
            Raw photobleach mask from ``build_mask``; used as fallback when
            ``hd_states`` is None.
        """
        N_traces, length = self.process_Q(w, mask=mask, hd_states=hd_states,
                                          smooth_method=smooth_method,
                                          polyorder=polyorder)
        if not np.any(means):
            means = [0.5]

        self.means = means
        means = np.array(means)
        k = means.shape[0]

        # Final NaN/inf guard — smoothing can propagate raw-data NaN into
        # frames that our per-fragment sanitization in process_Q may miss
        # (e.g. boundary frames from uniform_filter1d in 'nearest' mode).
        if not np.isfinite(self.pro_Q).all():
            col = self.pro_Q[:, 0]
            finite_vals = col[np.isfinite(col)]
            fill = float(finite_vals.mean()) if len(finite_vals) else 0.0
            self.pro_Q = np.where(np.isfinite(self.pro_Q), self.pro_Q, fill)

        # ── Covariance floor derived from actual data variance ────────────────
        # hmmlearn's _init() initialises covariances as:
        #   cv = np.cov(X.T) + min_covar * np.eye(n_features)
        # With min_covar=0 (the original value) a constant trace gives cv=0
        # → singular matrix → Cholesky fails on the very first E-step.
        raw_var   = float(np.var(self.pro_Q)) if len(self.pro_Q) > 0 else 0.01
        data_var  = raw_var if (np.isfinite(raw_var) and raw_var > 0) else 1e-4
        min_covar = max(data_var * 0.01, 1e-6)

        if r:
            print('fitting')
            tic = time.perf_counter()
            models, conv = [], []

            for e in tqdm(range(epoch)):
                print(f'\n Epoch {e}')

                # init_params='st': hmmlearn initialises startprob and transmat
                # from data; means and covariances are set manually below.
                #
                # Initial covariance is set to data_var (global variance) —
                # deliberately wide so every Gaussian covers the full data range
                # and every state has non-zero posterior on the first E-step.
                # This prevents dead states (zero posterior → 0/0 NaN in M-step)
                # even when user means don't perfectly match the data clusters.
                # EM then tightens covariances naturally as it converges.
                # Sticky prior: uniform baseline (0.01) + extra weight on diagonal.
                # sticky_w=1 → same as old scalar 0.01 (no bias).
                # sticky_w>1 → self-transitions are preferred, reducing flicker.
                transmat_prior = np.full((k, k), 0.01)
                np.fill_diagonal(transmat_prior, 0.01 + float(sticky_w) - 1.0)
                transmat_prior = np.clip(transmat_prior, 1e-6, None)

                model = GaussianHMM(
                    n_components=k, n_iter=n_iter, verbose=True,
                    min_covar=min_covar,
                    startprob_prior=np.ones(k)/k, means_prior=means,
                    covars_prior=min_covar, covariance_type=covariance_type,
                    transmat_prior=transmat_prior,
                    init_params='st',
                    params='stmc',
                    implementation='scaling'
                )
                model.means_ = means   # user-provided starting point

                # Wide initial covariance — large enough to keep all states alive
                nf       = self.pro_Q.shape[1]
                init_cov = max(data_var, min_covar * 100)
                if covariance_type == 'full':
                    model._covars_ = np.tile(np.eye(nf) * init_cov, (k, 1, 1))
                elif covariance_type == 'diag':
                    model._covars_ = np.full((k, nf), init_cov)
                elif covariance_type == 'tied':
                    model._covars_ = np.eye(nf) * init_cov
                else:   # spherical
                    model._covars_ = np.full(k, init_cov)
                model.fit(self.pro_Q, length)
                # Sanitise immediately so conv[] is finite for argmax
                _sanitize_fitted_model(model, k, means, min_covar)
                models.append(model)
                last_ll = model.monitor_.history[-1]
                conv.append(last_ll if np.isfinite(last_ll) else -np.inf)

            # Pick epoch with the highest finite log-likelihood
            if any(np.isfinite(v) for v in conv):
                best = int(np.argmax(conv))
            else:
                best = 0   # all epochs diverged — take first, already sanitised
            print(f'Best likelihood {conv[best]}')
            model = models[best]
            toc = time.perf_counter()
            print(f"Finished in {toc - tic:0.4f} seconds")
            with open(self.path + r"\model.pkl", "wb") as f:
                pickle.dump(model, f)
        else:
            model = GaussianHMM(
                n_components=k, n_iter=10, verbose=True, min_covar=100,
                covariance_type=covariance_type, covars_prior=0.001,
                transmat_prior=0, init_params='stc', params=params
            )
            with open(self.path + r"\model.pkl", "rb") as f:
                model = pickle.load(f)
            #with open(r'H:\TIRF\20221229\lane3\dmc1\320s\FRET\0'+r"\model.pkl", "rb") as f: model=pickle.load(f)
            #with open(r'H:\TIRF\20221229\lane2\dmc1\530s\FRET\0'+r"\model.pkl", "rb") as f: model=pickle.load(f)

        self.mus = np.array(model.means_)
        sigmas = np.array(model.covars_)
        P = np.array(model.transmat_)
        print(sigmas)
        print(self.mus)
        print(P)

        print('predicting')
        tic = time.perf_counter()
        hidden_states = model.predict(self.pro_Q, length)
        likelihood    = model.predict(self.pro_Q, length)
        aic = model.aic(self.pro_Q, length)
        bic = model.bic(self.pro_Q, length)
        toc = time.perf_counter()

        print(f"Finished in {toc - tic:0.4f} seconds")
        print(f"Log Likelihood =  {np.sum(likelihood):.4f}")
        print(f"averaged Log Likelihood =  {np.average(likelihood):.4f}")
        print(f"AIC =  {aic:.4f}")
        print(f"aBIC =  {bic:.4f}")

        self.hidden_states = hidden_states
        return model, hidden_states

    # ──────────────────────────────────────────────────────────────
    # State assignment & export
    # ──────────────────────────────────────────────────────────────
    def cal_states(self):
        N_traces = self.Q.shape[0]
        length = self.length
        start = 0
        time_smooth = uniform_filter1d(self.time, 10, mode='reflect')
        hd_states, time_arr = [], []

        # Build a dict: trace_index → end_frame for every fragment in pro_Q.
        # Fully-PB'd / empty selected traces are absent and get empty outputs.
        included = {idx: ef for idx, ef in getattr(self, 'pro_Q_indices', [])}

        # Pre-compute mapping: unsorted model state index → sorted state index.
        # E.g. mus = [0.5, 0.2, 0.8]  →  sorted = [0.2, 0.5, 0.8]
        #      rank = [1, 0, 2]  (model-state-0 maps to sorted-index-1, etc.)
        mus_flat = self.mus.flatten()
        rank     = np.argsort(np.argsort(mus_flat))   # inverse permutation

        j = 0
        for i in tqdm(np.arange(0, N_traces)):
            if self.selected[i] == 1 and i in included:
                end = start + length[j]
                mus_frag       = self.hidden_states[start:end]   # unsorted model indices
                time_frag      = time_smooth[:length[j]]
                hd_states_frag = rank[mus_frag]                  # sorted int indices
                time_arr.append(time_frag)
                hd_states.append(hd_states_frag)
                start = end
                j += 1
            else:
                # Not selected, or selected but fully photobleached (empty frag)
                hd_states.append([])

        print(len(hd_states))
        print(len(self.length))

        self.result_hd_states = hd_states   # expose for immediate preview in app.py
        # No auto-save here — user must click "Save hmm.npz" explicitly.

    # ──────────────────────────────────────────────────────────────
    # Unified save / load  (new hmm.npz nested-dict format)
    # ──────────────────────────────────────────────────────────────

    _CHANNEL_TIME_KEY = {
        'fret_g': 'time_g', 'fret_b': 'time_b',
        'gg': 'time_g', 'bb': 'time_b', 'rr': 'time_r',
    }

    def save_hmm(self, hmm_states_dict=None, hmm_means_dict=None, hmm_mask=None):
        """
        Save in-memory HMM predictions to a single compressed hmm.npz.

        File layout — one key per channel, each holding a sub-dict:
            fret_g → {hd_states: int16 array, means: float64 array, time: float64 array}
            rr     → {hd_states: …, means: …, time: …}
            …

        Channel names are the top-level npz keys; no wrapper 'data' key needed.

        Parameters
        ----------
        hmm_states_dict : dict {channel: 2-D object array}
            Live hmm_states from app.py (None=no pred, NaN=PB, int=state index).
        hmm_means_dict  : dict {channel: 1-D float64 array}
            Fitted means per channel (sorted low→high).
        hmm_mask : ignored (kept for API compatibility)

        Returns
        -------
        saved : list[str]  channel names successfully written
        """
        if not hmm_states_dict:
            return []

        raw = np.load(self.path + r'\data.npz')
        per_channel, saved = {}, []

        for ch, hd_2d in hmm_states_dict.items():
            if ch not in CHANNEL_CONFIG:
                continue
            try:
                means = np.array([], dtype=np.float64)
                if hmm_means_dict and ch in hmm_means_dict:
                    means = np.asarray(hmm_means_dict[ch], dtype=np.float64).flatten()

                # ── Object array → int16 state-index array ────────────────────
                # Sentinels: -1 = no prediction, -2 = photobleached.
                # hmm_states already stores integer indices (0, 1, 2 …).
                int_states = np.full(hd_2d.shape, -1, dtype=np.int16)
                for idx in np.ndindex(hd_2d.shape):
                    v = hd_2d[idx]
                    if v is None:
                        pass                          # stays -1
                    else:
                        try:
                            if np.isnan(float(v)):
                                int_states[idx] = -2  # photobleached
                            else:
                                int_states[idx] = int(v)
                        except (TypeError, ValueError):
                            pass

                t_key = self._CHANNEL_TIME_KEY.get(ch, 'time_g')
                t_vec = raw[t_key].astype(np.float64)
                if len(t_vec) > hd_2d.shape[1]:
                    t_vec = t_vec[:hd_2d.shape[1]]

                # Each channel is one top-level key holding a plain dict
                per_channel[ch] = np.array(
                    {'hd_states': int_states, 'means': means, 'time': t_vec},
                    dtype=object,
                )
                saved.append(ch)

            except Exception as e:
                print(f'save_hmm: skipped {ch}: {e}')

        if per_channel:
            out_path = os.path.join(self.path, 'hmm.npz')
            if os.path.exists(out_path):
                ts     = time.strftime('%Y%m%d_%H%M%S')
                backup = os.path.join(self.path, f'hmm_{ts}.npz')
                os.rename(out_path, backup)
                print(f'save_hmm: backed up previous file → hmm_{ts}.npz')
            np.savez_compressed(out_path, **per_channel)

        return saved

    def load_hmm(self):
        """
        Load hmm.npz and return channel data.

        Supports two formats:
          New  — channel names are top-level keys (``fret_g``, ``rr``, …),
                 each holding a dict ``{hd_states, means, time}``.
          Old  — single ``data`` object-array (backward compatibility)

        Returns
        -------
        dict  {channel: {'hd_states': array, 'time': array, 'means': array}}
        None  if the file does not exist or cannot be read
        """
        npz_path = os.path.join(self.path, 'hmm.npz')
        try:
            npz = np.load(npz_path, allow_pickle=True)

            # ── New format: channel names are top-level keys ──────────────────
            known_channels = set(CHANNEL_CONFIG.keys())
            top_level_chs  = [k for k in npz.files if k in known_channels]
            if top_level_chs:
                # Each key holds a 0-d object array wrapping a plain dict
                return {ch: npz[ch].item() for ch in top_level_chs}

            # ── Old format: single 'data' wrapper (backward compatibility) ────
            if 'data' in npz.files:
                return npz['data'].item()

            return None

        except Exception as e:
            print(f'load_hmm: {e}')
            return None
