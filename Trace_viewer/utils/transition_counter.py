"""
utils/transition_counter.py
============================
Helpers for counting HMM state-to-state transitions and writing
per-channel transition files alongside hmm.npz.

One file is written per channel:  transition_{channel}.txt
  e.g.  transition_fret_g.txt
        transition_rr.txt

On save, only the file for the channel being saved is backed up:
  transition_fret_g.txt  →  transition_fret_g_20260518_143200.txt
Other channel files in the same folder are never touched.

Usage (from app.py hmm_save_npz callback)
------------------------------------------
    ts = rtime.strftime('%Y%m%d_%H%M%S')
    saved, ts_used = fitter.save_hmm(..., timestamp=ts)
    t_written = save_transition_txt(path, hmm_states, hmm_means, ts_used)
"""

import os
import numpy as np


# ─────────────────────────────────────────────────────────────────────────────
# Core counter
# ─────────────────────────────────────────────────────────────────────────────

def count_transitions(hd_states_2d, n_states):
    """Count state-to-state transitions across all traces in a 2-D object array.

    Only *cross-state* transitions (state_i → state_j where i ≠ j) are counted.
    Each row is processed left-to-right and stops at the first NaN frame
    (photobleach boundary) so censored regions are excluded.

    Parameters
    ----------
    hd_states_2d : array-like of rows
        Each row is a sequence whose elements are one of:
          - ``None``   : no HMM prediction (skipped)
          - ``np.nan`` : photobleached frame (stops this trace)
          - ``int``    : state index (0-based)
    n_states : int
        Number of distinct states (matrix dimension).

    Returns
    -------
    counts : np.ndarray, shape (n_states, n_states), dtype=int
        ``counts[i, j]`` = transitions from state i to state j (i ≠ j).
        Diagonal is always 0.
    n_traces_counted : int
        Number of traces that contributed at least one cross-state transition.
    """
    counts = np.zeros((n_states, n_states), dtype=int)
    n_traces_counted = 0

    for row in hd_states_2d:
        seq = []
        for v in row:
            if v is None:
                continue
            try:
                f = float(v)
                if np.isnan(f):
                    break       # photobleach boundary — stop processing this trace
                seq.append(int(f))
            except (TypeError, ValueError):
                continue

        if len(seq) < 2:
            continue

        contributed = False
        for a, b in zip(seq[:-1], seq[1:]):
            if a != b and 0 <= a < n_states and 0 <= b < n_states:
                counts[a, b] += 1
                contributed = True
        if contributed:
            n_traces_counted += 1

    return counts, n_traces_counted


# ─────────────────────────────────────────────────────────────────────────────
# File writer  (Option B — one file per channel)
# ─────────────────────────────────────────────────────────────────────────────

def save_transition_txt(path, hmm_states_dict, hmm_means_dict, timestamp):
    """Save per-channel transition counts to ``transition_{channel}.txt``.

    Each channel is written to its **own** file.  Only the file belonging to
    the channel currently being saved is backed up — all other channel files
    in the folder are left completely untouched.

    Backup naming mirrors hmm.npz:
        transition_fret_g.txt  →  transition_fret_g_20260518_143200.txt

    Parameters
    ----------
    path : str
        Folder path (same directory that contains hmm.npz).
    hmm_states_dict : dict {channel: 2-D object array}
        Live ``hmm_states`` dict from app.py globals.
    hmm_means_dict : dict {channel: list/array of float} or None
        Fitted means per channel (sorted low→high).
    timestamp : str
        'YYYYMMDD_HHMMSS' — the timestamp already used when saving hmm.npz.

    Returns
    -------
    list[str]
        Channel names whose files were successfully written.
    """
    if not hmm_states_dict:
        return []

    # Human-readable timestamp for the file header
    try:
        ts_human = (f'{timestamp[:4]}-{timestamp[4:6]}-{timestamp[6:8]} '
                    f'{timestamp[9:11]}:{timestamp[11:13]}:{timestamp[13:15]}')
    except Exception:
        ts_human = timestamp

    written = []

    for ch, hd_2d in hmm_states_dict.items():

        # ── Determine number of states ────────────────────────────────────────
        means = []
        if hmm_means_dict and ch in hmm_means_dict:
            try:
                means = [float(m) for m in hmm_means_dict[ch]
                         if m is not None]
            except (TypeError, ValueError):
                means = []

        if means:
            n_states = len(means)
        else:
            # Infer from the highest state index present in the array
            max_state = -1
            for row in hd_2d:
                for v in row:
                    if v is None:
                        continue
                    try:
                        f = float(v)
                        if not np.isnan(f):
                            max_state = max(max_state, int(f))
                    except (TypeError, ValueError):
                        pass
            if max_state < 1:
                continue       # fewer than 2 states — skip this channel
            n_states = max_state + 1

        if n_states < 2:
            continue

        counts, n_traced = count_transitions(hd_2d, n_states)

        # ── Build file content ────────────────────────────────────────────────
        means_str = ', '.join(f'{m:.3f}' for m in means) if means else 'N/A'
        col_w = 9

        lines = [
            f'=== HMM Transition Counts — {ch} ===',
            f'Saved:   {ts_human}',
            f'Path:    {path}',
            f'Channel: {ch}',
            f'States:  {n_states}  |  Means: [{means_str}]',
            f'Traces with transitions: {n_traced}',
            '',
            'Transition matrix (rows = from-state, cols = to-state):',
            'From\\To  ' + ''.join(f'{"S"+str(j):>{col_w}}' for j in range(n_states)),
        ]
        for i in range(n_states):
            row_label = f'S{i:<8}'
            row_vals  = ''.join(f'{counts[i, j]:>{col_w}}' for j in range(n_states))
            lines.append(row_label + row_vals)

        total = int(counts.sum())
        lines += [
            '',
            f'Total cross-state transitions: {total}',
        ]

        # ── Back up only THIS channel's file ──────────────────────────────────
        out_path = os.path.join(path, f'transition_{ch}.txt')
        if os.path.exists(out_path):
            backup = os.path.join(path, f'transition_{ch}_{timestamp}.txt')
            os.rename(out_path, backup)
            print(f'save_transition_txt: backed up → transition_{ch}_{timestamp}.txt')

        with open(out_path, 'w', encoding='utf-8') as fh:
            fh.write('\n'.join(lines) + '\n')

        print(f'save_transition_txt: wrote {out_path}')
        written.append(ch)

    return written
