"""
utils/undo_manager.py
=====================
Centralised undo support for Trace_viewer (Ctrl+Z).

Two independent, capped histories
---------------------------------
  - breakpoint edits  → used on the Tools / Aois / GMM tabs
  - HMM-state edits   → used on the HMM tab

Each history holds at most ``UNDO_MAX`` snapshots.  Pushing one more than
the cap silently drops the oldest entry, so the user can press Ctrl+Z up
to ``UNDO_MAX`` times in a row.

The two histories never mix: which one Ctrl+Z consumes is decided by the
active tab in the calling callback, not here.

Snapshot formats
----------------
Breakpoint snapshot (dict):
    {'scope': 'single', 'channel': str, 'i': int, 'data': list}
    {'scope': 'all',    'data': {channel: [list, list, ...]}}

HMM snapshot (dict):
    {'ch': str, 'mol': int, 'row': np.ndarray}   # one molecule's state row

Wiring (done in a later step — this file only provides the helpers)
-------------------------------------------------------------------
app.py
    import utils.undo_manager as undo
    # before a breakpoint mutation in update_fig:
    undo.push_bkps_single(bkps, channel, i)      # or push_bkps_all(bkps)
    # in the new undo_bkps_callback:
    snap = undo.pop_bkps();  undo.restore_bkps(bkps, snap)

utils/HMM_slidebar.py
    import utils.undo_manager as undo
    # before a boundary mutation in hmm_trace_click / vacuum tool:
    undo.push_hmm(hmm_states, ch, mol)
    # in the new undo_hmm_callback:
    snap = undo.pop_hmm();  ch = undo.restore_hmm(hmm_states, snap)
"""

import copy
import numpy as np


# Maximum consecutive Ctrl+Z presses (== history depth per stack)
UNDO_MAX = 5

# Two separate histories — newest entry is always last
_undo_bkps = []
_undo_hmm  = []


# ─────────────────────────────────────────────────────────────────────────────
# Breakpoint history  (Tools / Aois / GMM tabs)
# ─────────────────────────────────────────────────────────────────────────────

def push_bkps_single(bkps, channel, i):
    """Snapshot ONE trace's breakpoint list before it is mutated.

    Use for click-add/remove, dtime/etime, single-trace Clear, and
    single-trace auto-detect (find_chp on 'current trace').
    Silently does nothing if the channel/index is invalid.
    """
    if channel is None or channel not in bkps:
        return
    try:
        row = bkps[channel][i]
    except (IndexError, TypeError, KeyError):
        return
    _undo_bkps.append({
        'scope':   'single',
        'channel': channel,
        'i':       int(i),
        'data':    copy.deepcopy(row),
    })
    del _undo_bkps[:-UNDO_MAX]


def push_bkps_all(bkps):
    """Snapshot the ENTIRE breakpoint dict before a bulk mutation.

    Use for Clear All, Set All, Reset/confirm-reset, and all-trace
    auto-detect (find_chp on 'all traces' / 'all good').
    """
    _undo_bkps.append({
        'scope': 'all',
        'data':  {ch: copy.deepcopy(list(rows)) for ch, rows in bkps.items()},
    })
    del _undo_bkps[:-UNDO_MAX]


def can_undo_bkps():
    """True when there is at least one breakpoint snapshot to undo."""
    return len(_undo_bkps) > 0


def pop_bkps():
    """Remove and return the most recent breakpoint snapshot (or None)."""
    return _undo_bkps.pop() if _undo_bkps else None


def restore_bkps(bkps, snapshot):
    """Apply a breakpoint snapshot back onto the live ``bkps`` dict in place.

    Returns True if something was restored, False otherwise.
    """
    if snapshot is None:
        return False
    if snapshot['scope'] == 'single':
        ch, i = snapshot['channel'], snapshot['i']
        if ch in bkps and 0 <= i < len(bkps[ch]):
            bkps[ch][i] = copy.deepcopy(snapshot['data'])
            return True
    elif snapshot['scope'] == 'all':
        for ch, rows in snapshot['data'].items():
            bkps[ch] = copy.deepcopy(rows)
        return True
    return False


# ─────────────────────────────────────────────────────────────────────────────
# HMM-state history  (HMM tab)
# ─────────────────────────────────────────────────────────────────────────────

def push_hmm(hmm_states, ch, mol):
    """Snapshot ONE molecule's HMM state row before it is mutated.

    Use before every slidebar boundary edit (click-expand, drag) and the
    vacuum-fill tool.  Each edit only touches a single row, so one row copy
    is enough to undo it.  Silently does nothing if ch/mol is invalid.
    """
    if ch not in hmm_states:
        return
    arr = hmm_states[ch]
    if mol < 0 or mol >= arr.shape[0]:
        return
    _undo_hmm.append({
        'ch':  ch,
        'mol': int(mol),
        'row': arr[mol, :].copy(),
    })
    del _undo_hmm[:-UNDO_MAX]


def can_undo_hmm():
    """True when there is at least one HMM snapshot to undo."""
    return len(_undo_hmm) > 0


def pop_hmm():
    """Remove and return the most recent HMM snapshot (or None)."""
    return _undo_hmm.pop() if _undo_hmm else None


def restore_hmm(hmm_states, snapshot):
    """Apply an HMM snapshot back onto the live ``hmm_states`` dict in place.

    Returns the channel name that was restored, or None if nothing applied.
    The caller is responsible for refreshing ``hmm_fret_g`` and rebuilding
    the HMM bar figure afterwards.
    """
    if snapshot is None:
        return None
    ch, mol, row = snapshot['ch'], snapshot['mol'], snapshot['row']
    if ch in hmm_states and 0 <= mol < hmm_states[ch].shape[0]:
        hmm_states[ch][mol, :] = row
        return ch
    return None


# ─────────────────────────────────────────────────────────────────────────────
# Shared helpers
# ─────────────────────────────────────────────────────────────────────────────

def is_ctrl_z(event):
    """True when an EventListener key event represents Ctrl+Z.

    Requires 'ctrlKey' to be present in the EventListener props
    (see layout.py — Step 1 of the undo plan).
    """
    if not event:
        return False
    return event.get('key') in ('z', 'Z') and bool(event.get('ctrlKey'))


def clear_all():
    """Wipe both histories — call this when a new dataset is loaded so that
    Ctrl+Z cannot revert into stale data from the previous folder.
    """
    _undo_bkps.clear()
    _undo_hmm.clear()
