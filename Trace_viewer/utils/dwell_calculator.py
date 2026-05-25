import numpy as np


def _extract_dwells(states_2d, t_arr, left_mode, left_cut, dead_time_s, right_cut,
                    censor_last=False):
    """
    Extract dwell times from a 2D HMM state array and apply boundary cuts.

    Parameters
    ----------
    states_2d   : 2D object array (N_mol × N_frames)
                  None = no prediction, NaN = photobleached, int = state index
    t_arr       : 1D float array — time axis in seconds
    left_mode   : 'cut' or 'dead'
                  'cut'  → skip the first left_cut dwells of each molecule
                  'dead' → keep all dwells but add dead_time_s to the FIRST
                           dwell's duration (corrects for unobserved time
                           before recording started)
    left_cut    : int   — number of dwells to skip from the left (cut mode)
    dead_time_s : float — seconds to add to the first dwell (dead time mode)
    right_cut   : int   — number of dwells to skip from the right (both modes)
    censor_last : bool  — when True, any dwell whose end time equals the
                          channel recording maximum (t_arr[-1]) is marked
                          observed = 0 (right-censored) instead of 1.

    Returns
    -------
    dwell_dict  : dict  { str(state) → np.ndarray of dwell durations (seconds) }
    events_dict : dict  { str(state) → np.ndarray of 1s/0s
                          1 = complete observed event, 0 = right-censored }

    Notes
    -----
    - Cut dwells are excluded entirely — not stored, no event flag.
    - Duration of each dwell is inclusive of the last frame:
        duration = t_arr[f_end] - t_arr[f_start] + dt
    """
    dt       = float(t_arr[1] - t_arr[0]) if len(t_arr) > 1 else 1.0
    max_time = float(t_arr[-1])           # recording end — identical for all states

    dwell_dict  = {}   # { str(state): [float, ...] }
    events_dict = {}   # { str(state): [1, ...] }

    N_mol, N_frames = states_2d.shape

    for mol in range(N_mol):
        row = states_2d[mol]

        # ── Build consecutive-run list for this molecule ──────────────────
        # Stop at the first None (no prediction) or NaN (photobleached) frame.
        runs = []   # [(state_int, f_start, f_end), ...]
        f = 0
        while f < N_frames:
            v = row[f]
            if v is None:
                break
            try:
                fv = float(v)
                if np.isnan(fv):
                    break
                state = int(fv)
            except (TypeError, ValueError):
                break

            f_start = f
            f += 1
            while f < N_frames:
                v2 = row[f]
                if v2 is None:
                    break
                try:
                    fv2 = float(v2)
                    if np.isnan(fv2) or int(fv2) != state:
                        break
                except (TypeError, ValueError):
                    break
                f += 1

            runs.append((state, f_start, f - 1))

        if not runs:
            continue

        # ── Apply boundary cuts ───────────────────────────────────────────
        left_skip  = int(left_cut)  if left_mode == 'cut' else 0
        right_skip = int(right_cut)

        start_idx = left_skip
        end_idx   = len(runs) - right_skip   # exclusive upper bound

        if start_idx >= end_idx:
            continue   # all dwells removed by cuts for this molecule

        valid_runs = runs[start_idx:end_idx]

        # ── Collect dwell durations ───────────────────────────────────────
        for k, (state, f_start, f_end) in enumerate(valid_runs):
            t_s = t_arr[f_start] if f_start < len(t_arr) else f_start * dt
            t_e = t_arr[f_end]   if f_end   < len(t_arr) else f_end   * dt
            duration = (t_e - t_s) + dt   # inclusive of last frame

            # Dead time mode: extend the first dwell backward in time
            if left_mode == 'dead' and k == 0:
                duration += float(dead_time_s)

            state_key = str(state)
            observed  = 0 if (censor_last and np.isclose(t_e, max_time)) else 1
            dwell_dict.setdefault(state_key,  []).append(duration)
            events_dict.setdefault(state_key, []).append(observed)

    # ── Convert lists to numpy arrays ─────────────────────────────────────
    for key in dwell_dict:
        dwell_dict[key]  = np.array(dwell_dict[key],  dtype=np.float64)
        events_dict[key] = np.array(events_dict[key], dtype=np.int32)

    return dwell_dict, events_dict
