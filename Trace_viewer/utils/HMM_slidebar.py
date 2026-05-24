"""
HMM state-bar helpers and Dash callbacks.

Call ``register_callbacks(app, app_mod)`` once in app.py, passing
``sys.modules[__name__]`` as *app_mod* so callbacks can read/write the
shared globals (hmm_states, hmm_fret_g, time) that live in app.py.
"""

import numpy as np
import plotly.graph_objects as go
from dash import Input, Output, State, no_update, callback_context, Patch
from dash.exceptions import PreventUpdate
from . import undo_manager as undo


# ── Visual constants ───────────────────────────────────────────────────────────

STATE_COLORS = [
    '#4C72B0', '#DD8452', '#55A868', '#C44E52',
    '#8172B2', '#937860', '#DA8BC3', '#8C8C8C',
    '#CCB974', '#64B5CD',                        # states 9 & 10
]

def _to_grayscale(hex_color):
    """Convert a hex color to its luminance-based grayscale equivalent.
    Uses the standard perceptual luminance formula: L = 0.299R + 0.587G + 0.114B
    """
    h  = hex_color.lstrip('#')
    r, g, b = int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16)
    L  = int(0.299 * r + 0.587 * g + 0.114 * b)
    return f'#{L:02X}{L:02X}{L:02X}'

# Grayscale equivalent for each state color — used for excluded (cut) segments.
# Computed once at import time from STATE_COLORS so the two lists stay in sync.
STATE_GRAYS = [_to_grayscale(c) for c in STATE_COLORS]
# Result:
#   state 0  #4C72B0 (blue)   → #6E6E6E
#   state 1  #DD8452 (orange) → #999999
#   state 2  #55A868 (green)  → #888888
#   state 3  #C44E52 (red)    → #727272
#   state 4  #8172B2 (purple) → #7E7E7E
#   state 5  #937860 (brown)  → #7D7D7D
#   state 6  #DA8BC3 (pink)   → #A9A9A9
#   state 7  #8C8C8C (gray)   → #8C8C8C
#   state 8  #CCB974 (khaki)  → #B7B7B7
#   state 9  #64B5CD (cyan)   → #A0A0A0

# Maps HMM-channel selector value → key used in the global `time` dict
CHANNEL_TIME_KEY = {
    'fret_g': 'fret_g',
    'fret_b': 'fret_b',
    'gg':     'g',
    'bb':     'b',
    'rr':     'r',
}

EMPTY_BAR_FIGURE = {
    'data': [],
    'layout': {
        'height': 50,
        'margin': {'l': 0, 'r': 0, 't': 0, 'b': 0},
        'xaxis': {'visible': False, 'fixedrange': True},
        'yaxis': {'visible': False, 'fixedrange': True},
        'paper_bgcolor': '#f5f5f5',
        'plot_bgcolor':  '#f5f5f5',
        'dragmode': False,
        'annotations': [{
            'x': 0.5, 'y': 0.5, 'xref': 'paper', 'yref': 'paper',
            'text': 'No HMM prediction loaded — fit HMM first',
            'showarrow': False,
            'font': {'size': 12, 'color': '#aaa', 'family': 'Arial'},
        }],
    },
}


# ── Pure helpers ───────────────────────────────────────────────────────────────

def get_excluded_indices(segments, left_mode, left_cut, right_cut):
    """Return the set of segment indices that should be rendered in grayscale.

    Only STATE-type segments are counted for cut purposes; ``no_prediction``
    and ``pb`` blocks are never grayed and never consume a cut slot.

    Parameters
    ----------
    segments   : list of segment dicts (from hmm-segments-store)
    left_mode  : 'cut' or 'dead'
                 'cut'  → exclude first left_cut STATE segments
                 'dead' → no left exclusion (dead time only extends first dwell duration)
    left_cut   : int — number of STATE segments to exclude from the left (cut mode only)
    right_cut  : int — number of STATE segments to exclude from the right

    Returns
    -------
    excluded : set of int
        Indices into the FULL ``segments`` list that should be shown in
        grayscale.  Only state-type entries are ever included in this set.
    """
    # Collect the full-list indices of STATE segments in order
    state_indices = [k for k, seg in enumerate(segments) if seg['type'] == 'state']
    n = len(state_indices)
    if n == 0:
        return set()

    excluded = set()

    # Left boundary — only in cut mode
    if left_mode == 'cut':
        left_n = int(left_cut) if left_cut else 0
        for pos in range(min(left_n, n)):
            excluded.add(state_indices[pos])

    # Right boundary — always applied
    right_n = int(right_cut) if right_cut else 0
    for pos in range(max(0, n - right_n), n):
        excluded.add(state_indices[pos])

    return excluded


def _hex_to_rgba(hex_color: str, alpha: float) -> str:
    """Convert ``'#RRGGBB'`` to ``'rgba(R,G,B,alpha)'``."""
    h = hex_color.lstrip('#')
    r, g, b = int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16)
    return f'rgba({r},{g},{b},{alpha})'


def trace_to_segments(trace, t_arr):
    """Convert an object-dtype hmm_states row to a list of segment dicts.

    Segment types: ``'no_prediction'`` | ``'state'`` | ``'pb'``

    Every dict contains: ``type, t_start, t_end, f_start, f_end``.
    ``'state'`` dicts additionally carry ``state`` (int index, 0 = lowest).

    ``hmm_states`` stores integer state indices directly (0, 1, 2 …).
    """
    n = min(len(trace), len(t_arr))
    if n == 0:
        return []

    # Locate the first PB (NaN) frame
    pb_frame = None
    for f in range(n):
        v = trace[f]
        if v is not None:
            try:
                if np.isnan(float(v)):
                    pb_frame = f
                    break
            except (TypeError, ValueError):
                pass

    n_fitted = pb_frame if pb_frame is not None else n

    segments = []
    start_f, cur_state = None, None

    for f in range(n_fitted):
        v = trace[f]
        if v is None:
            if cur_state is not None:
                segments.append({
                    'type': 'state', 'state': cur_state,
                    't_start': float(t_arr[start_f]), 't_end': float(t_arr[f - 1]),
                    'f_start': start_f, 'f_end': f - 1,
                })
                cur_state, start_f = None, None
            continue
        try:
            fv = float(v)
            if np.isnan(fv):
                break
        except (TypeError, ValueError):
            continue

        state = int(v)
        if cur_state is None:
            cur_state, start_f = state, f
        elif state != cur_state:
            segments.append({
                'type': 'state', 'state': cur_state,
                't_start': float(t_arr[start_f]), 't_end': float(t_arr[f - 1]),
                'f_start': start_f, 'f_end': f - 1,
            })
            cur_state, start_f = state, f

    if cur_state is not None and start_f is not None:
        end_f = n_fitted - 1
        segments.append({
            'type': 'state', 'state': cur_state,
            't_start': float(t_arr[start_f]), 't_end': float(t_arr[end_f]),
            'f_start': start_f, 'f_end': end_f,
        })

    # Fill trailing None frames (e.g. after PB was deleted, or gap before PB)
    last_end_f = segments[-1]['f_end'] if segments else -1
    if last_end_f < n_fitted - 1:
        trailing_f = last_end_f + 1
        segments.append({
            'type': 'no_prediction',
            't_start': float(t_arr[trailing_f]),
            't_end':   float(t_arr[n_fitted - 1]),
            'f_start': trailing_f,
            'f_end':   n_fitted - 1,
        })

    # Prepend 'no_prediction' block when None frames precede the first state
    if segments and segments[0]['f_start'] > 0:
        segments.insert(0, {
            'type': 'no_prediction',
            't_start': float(t_arr[0]),
            't_end':   float(t_arr[segments[0]['f_start'] - 1]),
            'f_start': 0,
            'f_end':   segments[0]['f_start'] - 1,
        })

    # Append PB block
    if pb_frame is not None and pb_frame < len(t_arr):
        if not segments or all(s['type'] == 'no_prediction' for s in segments):
            if pb_frame > 0:
                segments = [{
                    'type': 'no_prediction',
                    't_start': float(t_arr[0]),
                    't_end':   float(t_arr[pb_frame - 1]),
                    'f_start': 0, 'f_end': pb_frame - 1,
                }]
            else:
                segments = []
        segments.append({
            'type': 'pb',
            't_start': float(t_arr[pb_frame]), 't_end': float(t_arr[-1]),
            'f_start': pb_frame, 'f_end': len(t_arr) - 1,
        })

    return segments


def merge_same_state_segments(segments, arr_row=None):
    """Dissolve boundaries between adjacent segments that share the same state.

    When two consecutive segments both have ``type='state'`` and the same
    ``state`` index, the boundary between them is removed and they become one
    segment spanning both frame ranges.

    Parameters
    ----------
    segments : list[dict]
        Segment list as produced by ``trace_to_segments``.
    arr_row : writable 1-D object array or None
        If supplied (``hmm_states[ch][mol]``), all frames inside the merged
        region are written to the surviving segment's ``fret`` value so that
        the array stays in sync with the visual result.

    Returns
    -------
    list[dict]  — possibly shorter than the input.
    """
    if len(segments) < 2:
        return segments

    merged = [dict(segments[0])]
    for seg in segments[1:]:
        prev = merged[-1]
        if (prev['type'] == 'state'
                and seg['type'] == 'state'
                and prev['state'] == seg['state']):
            # Absorb seg into prev — extend end boundary only
            merged[-1] = {
                'type':    'state',
                'state':   prev['state'],
                't_start': prev['t_start'],
                't_end':   seg['t_end'],
                'f_start': prev['f_start'],
                'f_end':   seg['f_end'],
            }
            # Keep the underlying array consistent so future rebuilds stay merged
            if arr_row is not None:
                f0 = int(prev['f_start'])
                f1 = int(seg['f_end'])
                arr_row[f0:f1 + 1] = prev['state']
        else:
            merged.append(dict(seg))

    return merged


def make_hmm_bar_figure(segments, view_start, view_end, selected_boundary=-1,
                        vacuum_t=None, excluded_indices=None):
    """Build the Plotly figure for the HMM state bar from a segment list.

    Shape order in the returned figure:
      [0 … n_vis_segs-1]              — filled segment rectangles (layer='below')
      [n_vis_segs … n_vis_segs+n_b-1] — vertical boundary lines
    The clientside drag JS uses this ordering to update boundary positions.

    Parameters
    ----------
    vacuum_t : (t_start, t_end) or None
        When set, renders a white "pending" overlay with bold red borders over
        that time range.  Used by the Add tool while waiting for keyboard input.
    excluded_indices : set of int or None
        Full-list segment indices (as returned by ``get_excluded_indices``) that
        are excluded by the current left / right cut settings.  Excluded STATE
        segments are drawn with their per-state grayscale colour (STATE_GRAYS)
        and their label (S0, S1 …) is shown in red.  Non-state segments are
        never affected even if their index appears in this set.
    """
    if excluded_indices is None:
        excluded_indices = set()
    fig = go.Figure()

    # Pre-compute, for each of the 400 uniformly-spaced scatter points, which
    # segment owns it.  Visual rectangles extend from seg['t_start'] to the
    # NEXT segment's t_start (same rule used for shapes below), so we use the
    # same half-open interval here.  The index is stored in customdata so the
    # click-handler can identify the exact segment without any time-range
    # re-matching that could be thrown off by scatter-point quantisation.
    n_pts = 400
    x_pts = np.linspace(view_start, view_end, n_pts)

    # Build lookup: for each x value, find owning segment index (-1 = none)
    seg_owner = [-1] * n_pts
    for k, seg in enumerate(segments):
        t_lo = float(seg['t_start'])
        t_hi = (float(segments[k + 1]['t_start'])
                if k + 1 < len(segments)
                else float(seg['t_end']) + 1e-9)
        for p, x in enumerate(x_pts):
            if t_lo <= x < t_hi:
                seg_owner[p] = k

    fig.add_trace(go.Scattergl(
        x=list(x_pts), y=[0.5] * n_pts,
        customdata=[[s] for s in seg_owner],
        mode='markers',
        marker=dict(size=15, opacity=0, color='rgba(0,0,0,0)'),
        hovertemplate='t = %{x:.2f} s<extra></extra>',
        showlegend=False, name='__click__',
    ))

    # Pre-compute vacuum clipped range for annotation suppression
    vac_ts = vac_te = None
    if vacuum_t is not None:
        vac_ts = max(float(vacuum_t[0]), view_start)
        vac_te = min(float(vacuum_t[1]), view_end)
        if vac_te <= vac_ts:
            vac_ts = vac_te = None   # vacuum fully outside view — ignore

    shapes, annotations = [], []
    for k, seg in enumerate(segments):
        ts = max(seg['t_start'], view_start)
        # Extend to next segment's start to eliminate one-frame visual gaps
        raw_end = (float(segments[k + 1]['t_start'])
                   if k + 1 < len(segments) else float(seg['t_end']))
        te = min(raw_end, view_end)
        if te <= ts:
            continue

        # Determine whether this STATE segment is cut / excluded
        is_excluded = (seg['type'] == 'state') and (k in excluded_indices)

        if seg['type'] == 'pb':
            color, label = '#888888', 'PB'
        elif seg['type'] == 'no_prediction':
            color, label = '#d9d9d9', 'No prediction'
        else:
            idx = seg['state'] % len(STATE_COLORS)
            color = STATE_GRAYS[idx] if is_excluded else STATE_COLORS[idx]
            label = f"S{seg['state']}"

        shapes.append(dict(
            type='rect', x0=ts, x1=te, y0=0, y1=1,
            fillcolor=color, line=dict(width=0), layer='below',
        ))
        # Suppress text label when this segment is fully covered by the vacuum
        in_vacuum = (vac_ts is not None and ts >= vac_ts - 1e-9 and te <= vac_te + 1e-9)
        if not in_vacuum:
            if seg['type'] == 'no_prediction':
                font_color = '#888888'
            elif is_excluded:
                font_color = 'red'       # excluded STATE → red label
            else:
                font_color = 'white'
            annotations.append(dict(
                x=(ts + te) / 2, y=0.5, text=label,
                showarrow=False, xref='x', yref='y',
                xanchor='center', yanchor='middle',
                font=dict(
                    color=font_color, size=10, family='Arial',
                    style='italic' if seg['type'] == 'no_prediction' else 'normal',
                ),
            ))

    # Boundary lines — appended AFTER all fill shapes (important for JS shape indexing)
    for j in range(1, len(segments)):
        t_b = segments[j]['t_start']
        if not (view_start <= t_b <= view_end):
            continue
        selected = (j - 1 == selected_boundary)
        shapes.append(dict(
            type='line', x0=t_b, x1=t_b, y0=0, y1=1,
            line=dict(
                color='#FFD700' if selected else 'rgba(255,255,255,0.7)',
                width=3 if selected else 1.5,
            ),
        ))

    # ── Vacuum overlay (Add tool) — drawn above everything else ──────────────
    if vac_ts is not None:
        # White fill covering the vacuum region
        shapes.append(dict(
            type='rect', x0=vac_ts, x1=vac_te, y0=0, y1=1,
            fillcolor='white', line=dict(width=0),
            layer='above',
        ))
        # Bold red left boundary
        shapes.append(dict(
            type='line', x0=vac_ts, x1=vac_ts, y0=0, y1=1,
            line=dict(color='#CC0000', width=3),
            layer='above',
        ))
        # Bold red right boundary
        shapes.append(dict(
            type='line', x0=vac_te, x1=vac_te, y0=0, y1=1,
            line=dict(color='#CC0000', width=3),
            layer='above',
        ))
        # "?" label prompting keyboard input
        annotations.append(dict(
            x=(vac_ts + vac_te) / 2, y=0.5, text='?',
            showarrow=False, xref='x', yref='y',
            xanchor='center', yanchor='middle',
            font=dict(color='#CC0000', size=13, family='Arial'),
        ))

    fig.update_layout(
        height=50,
        margin=dict(l=0, r=0, t=0, b=0),
        xaxis=dict(range=[view_start, view_end], visible=False, fixedrange=True),
        yaxis=dict(range=[0, 1], visible=False, fixedrange=True),
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='#f5f5f5',
        shapes=shapes, annotations=annotations,
        clickmode='event', dragmode=False, showlegend=False,
    )
    return fig


# ── Callback registration ──────────────────────────────────────────────────────

def register_callbacks(app, app_mod):
    """Register all HMM state-bar callbacks with *app*.

    *app_mod* must be ``sys.modules['__main__']`` (or the app module object)
    so that callbacks can read and write the shared globals
    ``hmm_states``, ``hmm_fret_g``, and ``time``.
    """

    @app.callback(
        Output('hmm-state-bar-container', 'style'),
        Input('tabs', 'value'),
    )
    def toggle_hmm_sidebar(active_tab):
        if active_tab == 'HMM':
            return {'display': 'block', 'paddingBottom': '10px'}
        return {'display': 'none'}

    # ── Helper: resolve view window from relayoutData ───────────────────────
    def _view_window(relayout_data, t_arr):
        t0, t1 = float(t_arr[0]), float(t_arr[-1])
        if relayout_data:
            if ('xaxis.range[0]' in relayout_data
                    and 'xaxis.range[1]' in relayout_data):
                vs = float(relayout_data['xaxis.range[0]'])
                ve = float(relayout_data['xaxis.range[1]'])
                if ve > vs:
                    return vs, ve
            elif relayout_data.get('xaxis.autorange'):
                return t0, t1
        return t0, t1

    # ── Helper: read hmm_states[ch][mol] and build segments + figure ────────
    def _build_from_hmm_states(ch, mol, t_arr, relayout_data, sel_bnd=-1,
                               vacuum_t=None,
                               left_mode='cut', left_cut=0, right_cut=0):
        """Always read from the authoritative hmm_states global.

        Returns (figure, segments) or (EMPTY_BAR_FIGURE, []) when no data.
        ``vacuum_t`` is forwarded to ``make_hmm_bar_figure`` to render the
        Add-tool pending overlay.
        ``left_mode``, ``left_cut``, ``right_cut`` are forwarded to
        ``get_excluded_indices`` so excluded segments are shown in grayscale.
        """
        source = app_mod.hmm_states.get(ch)
        if source is None:
            return EMPTY_BAR_FIGURE, []
        try:
            trace = np.asarray(source[mol], dtype=object)
        except Exception as e:
            print(f'HMM bar error (trace {mol}, ch={ch}): {e}')
            return EMPTY_BAR_FIGURE, []

        segs = trace_to_segments(trace, t_arr)
        segs = merge_same_state_segments(segs, source[mol])
        if not segs:
            return EMPTY_BAR_FIGURE, []

        excluded = get_excluded_indices(
            segs, left_mode or 'cut',
            int(left_cut  or 0),
            int(right_cut or 0),
        )
        vs, ve = _view_window(relayout_data, t_arr)
        fig = make_hmm_bar_figure(segs, vs, ve, sel_bnd,
                                  vacuum_t=vacuum_t, excluded_indices=excluded)
        return fig, segs

    # ════════════════════════════════════════════════════════════════════════
    # Callback 1 — RENDER
    # Triggered by anything that changes WHICH molecule/channel is shown or
    # that means the HMM prediction data changed (fit, load, filter, zoom).
    # Always reads hmm_states[ch][mol] fresh — no branching on trigger.
    # ════════════════════════════════════════════════════════════════════════
    @app.callback(
        Output('hmm-state-bar',         'figure'),
        Output('hmm-segments-store',    'data'),
        Output('hmm-selected-boundary', 'data'),
        Output('hmm-vacuum-store',      'data',  allow_duplicate=True),
        Output('hmm-boundary-seg',      'data',  allow_duplicate=True),
        Input('i',               'value'),
        Input('tabs',            'value'),
        Input('hmm-fit-status',  'children'),
        Input('hmm-pb-status',   'children'),
        Input('hmm-save-status', 'children'),
        Input('hmm-channel',     'value'),
        Input('graph',           'relayoutData'),
        Input('dwell-left-mode', 'value'),       # cut / dead → changes left exclusions
        Input('dwell-left-cut',  'value'),        # left N to gray out (cut mode)
        Input('dwell-right-cut', 'value'),        # right N to gray out
        State('hmm-selected-boundary', 'data'),
        State('hmm-vacuum-store',      'data'),
    )
    def render_hmm_bar(i, active_tab, fit_status, pb_status, save_status,
                       channel, relayout_data,
                       left_mode, left_cut, right_cut,
                       selected_boundary, vacuum_data):
        if active_tab != 'HMM':
            raise PreventUpdate

        ch = channel or 'fret_g'
        time_key = CHANNEL_TIME_KEY.get(ch, 'fret_g')
        t_arr = app_mod.time.get(time_key, [])
        if len(t_arr) == 0:
            return EMPTY_BAR_FIGURE, [], -1, None, -1

        ctx = callback_context
        triggered = ctx.triggered[0]['prop_id'].split('.')[0] if ctx.triggered else ''

        # On plain zoom keep the current boundary highlight; clear on everything else
        sel_bnd = (int(selected_boundary)
                   if triggered == 'graph' and selected_boundary is not None
                   else -1)
        bnd_out   = no_update if triggered == 'graph' else -1
        bseg_out  = no_update if triggered == 'graph' else -1   # keep/clear hmm-boundary-seg

        mol = int(i)
        # Only honour the stored zoom when the zoom itself triggered this callback.
        # For any other trigger (mol change, fit, channel, cut inputs, etc.) the
        # trace graph is redrawn at full range, so we must reset the bar too.
        view_data = relayout_data if triggered == 'graph' else None

        # Preserve the vacuum on zoom/pan; clear it on mol, channel, or fit changes.
        vt = None
        if triggered == 'graph' and vacuum_data is not None:
            phase = vacuum_data.get('phase')
            if phase == 'vacuum_ready':
                vt = (vacuum_data['t_start'], vacuum_data['t_end'])
            # 'left_selected': gold boundary already stored in hmm-selected-boundary;
            # no white overlay needed — sel_bnd was kept above for the gold line.
            vacuum_out = no_update
        else:
            vacuum_out = None

        fig, segs = _build_from_hmm_states(
            ch, mol, t_arr, view_data, sel_bnd,
            vacuum_t=vt,
            left_mode=left_mode, left_cut=left_cut, right_cut=right_cut,
        )
        # On zoom/pan the segments are unchanged — only the bar window shifts.
        # Returning no_update prevents update_color_windows from firing
        # unnecessarily, which avoids the graph blink caused by the server
        # round-trip patching graph.figure with identical shapes.
        segs_out = no_update if triggered == 'graph' else segs
        return fig, segs_out, bnd_out, vacuum_out, bseg_out

    # ════════════════════════════════════════════════════════════════════════
    # Callback 2 — INTERACT
    # Handles clicks on the state bar: Delete and Boundary-move selection.
    # Drag boundary move removed — use click-on-trace (hmm_trace_click) instead.
    # ════════════════════════════════════════════════════════════════════════
    @app.callback(
        Output('hmm-state-bar',          'figure',  allow_duplicate=True),
        Output('hmm-segments-store',     'data',    allow_duplicate=True),
        Output('hmm_tool',               'value',   allow_duplicate=True),
        Output('hmm-selected-boundary',  'data',    allow_duplicate=True),
        Output('hmm-vacuum-store',       'data',    allow_duplicate=True),
        Output('hmm-boundary-seg',       'data',    allow_duplicate=True),
        Input('hmm-state-bar',  'clickData'),
        State('hmm_tool',              'value'),
        State('i',                     'value'),
        State('tabs',                  'value'),
        State('hmm-channel',           'value'),
        State('graph',                 'relayoutData'),
        State('hmm-vacuum-store',      'data'),
        State('dwell-left-mode',       'value'),
        State('dwell-left-cut',        'value'),
        State('dwell-right-cut',       'value'),
        prevent_initial_call=True,
    )
    def hmm_bar_interact(click_data, tool, i, active_tab, channel, relayout_data,
                         vacuum_data, left_mode, left_cut, right_cut):
        if active_tab != 'HMM':
            raise PreventUpdate
        if not click_data:
            raise PreventUpdate

        hmm_states = app_mod.hmm_states
        ch = channel or 'fret_g'
        time_key = CHANNEL_TIME_KEY.get(ch, 'fret_g')
        t_arr = app_mod.time.get(time_key, [])
        if len(t_arr) == 0:
            raise PreventUpdate

        mol  = int(i)

        try:
            click_x = float(click_data['points'][0]['x'])
        except (KeyError, IndexError, TypeError):
            raise PreventUpdate

        # Always build fresh segments from hmm_states — no store dependency
        if ch not in hmm_states:
            raise PreventUpdate
        fresh_trace = np.asarray(hmm_states[ch][mol], dtype=object)
        fresh_segs  = trace_to_segments(fresh_trace, t_arr)
        fresh_segs  = merge_same_state_segments(fresh_segs, hmm_states[ch][mol])

        excluded = get_excluded_indices(
            fresh_segs, left_mode or 'cut',
            int(left_cut or 0), int(right_cut or 0),
        )

        vs, ve = _view_window(relayout_data, t_arr)

        # ── Boundary move: click on bar selects nearest boundary (gold) ──────
        if tool == 'Boundary move':
            if len(fresh_segs) < 2:
                raise PreventUpdate

            # ── Step 1: identify which segment was clicked ────────────────────
            # Primary: read pre-assigned segment index from scatter customdata.
            clicked_seg_idx = None
            try:
                cd_val = click_data['points'][0].get('customdata')
                if cd_val is not None:
                    raw_idx = int(cd_val[0])
                    if 0 <= raw_idx < len(fresh_segs):
                        clicked_seg_idx = raw_idx
            except (KeyError, IndexError, TypeError, ValueError):
                pass

            # Fallback: range-check (same geometry as the renderer)
            if clicked_seg_idx is None:
                for k, seg in enumerate(fresh_segs):
                    t_lo = float(seg['t_start'])
                    t_hi = (float(fresh_segs[k + 1]['t_start'])
                            if k + 1 < len(fresh_segs)
                            else float(seg['t_end']) + 1e-9)
                    if t_lo <= click_x < t_hi:
                        clicked_seg_idx = k
                        break

            # ── Step 2: pick left or right boundary based on which half was clicked
            if clicked_seg_idx is not None:
                seg = fresh_segs[clicked_seg_idx]
                seg_mid = (float(seg['t_start']) + float(seg['t_end'])) / 2.0

                if click_x <= seg_mid:
                    # Left half → select the boundary on the LEFT of this segment.
                    # If this is the first segment there is no left boundary, so
                    # fall back to its right boundary.
                    nearest_bidx = max(clicked_seg_idx - 1, 0)
                else:
                    # Right half → select the boundary on the RIGHT of this segment.
                    # If this is the last segment there is no right boundary, so
                    # fall back to its left boundary.
                    nearest_bidx = min(clicked_seg_idx, len(fresh_segs) - 2)
            else:
                # Final fallback: pure distance to boundary start times
                nearest_j = min(
                    range(1, len(fresh_segs)),
                    key=lambda j: abs(click_x - float(fresh_segs[j]['t_start']))
                )
                nearest_bidx = nearest_j - 1


            fig_sel = make_hmm_bar_figure(fresh_segs, vs, ve, nearest_bidx,
                                          excluded_indices=excluded)
            # clicked_seg_idx tells hmm_trace_click which segment's state to use as fill
            bar_seg = clicked_seg_idx if clicked_seg_idx is not None else -1
            return fig_sel, no_update, no_update, nearest_bidx, no_update, bar_seg

        # ── Delete ────────────────────────────────────────────────────────────
        if tool == 'Delete':
            # ── Step 1: identify the clicked segment ─────────────────────────
            # Primary: read segment index from the scatter point's customdata.
            # make_hmm_bar_figure() pre-assigns each scatter point to the
            # segment whose visual rectangle covers it, so this is exact.
            clicked_idx = None
            try:
                cd_val = click_data['points'][0].get('customdata')
                if cd_val is not None:
                    raw_idx = int(cd_val[0])
                    if 0 <= raw_idx < len(fresh_segs) and fresh_segs[raw_idx]['type'] == 'state':
                        clicked_idx = raw_idx
            except (KeyError, IndexError, TypeError, ValueError):
                pass

            # Fallback: range-check against visual extents (same geometry as
            # the renderer).  Handles clicks exactly on a boundary line where
            # the scatter point might fall in a gap.
            if clicked_idx is None:
                for k, seg in enumerate(fresh_segs):
                    if seg['type'] != 'state':
                        continue
                    t_lo = float(seg['t_start'])
                    t_hi = (float(fresh_segs[k + 1]['t_start'])
                            if k + 1 < len(fresh_segs)
                            else float(seg['t_end']) + 1e-9)
                    if t_lo <= click_x < t_hi:
                        clicked_idx = k
                        break

            # Last-resort fallback: nearest state-segment midpoint
            if clicked_idx is None:
                best_dist = float('inf')
                for k, seg in enumerate(fresh_segs):
                    if seg['type'] != 'state':
                        continue
                    mid = (float(seg['t_start']) + float(seg['t_end'])) / 2.0
                    d = abs(click_x - mid)
                    if d < best_dist:
                        best_dist = d
                        clicked_idx = k

            if clicked_idx is None:
                raise PreventUpdate

            del_seg = fresh_segs[clicked_idx]
            f0 = int(del_seg['f_start'])
            f1 = int(del_seg['f_end'])

            # ── Step 2: neighbour states straight from the segment list ───────
            left_state = None
            for k2 in range(clicked_idx - 1, -1, -1):
                s = fresh_segs[k2]
                if s['type'] == 'state':
                    left_state = int(s['state'])
                    break

            right_state = None
            for k2 in range(clicked_idx + 1, len(fresh_segs)):
                s = fresh_segs[k2]
                if s['type'] == 'state':
                    right_state = int(s['state'])
                    break

            # ── Step 3: fill the deleted region ──────────────────────────────
            arr = hmm_states[ch]
            # --- UNDO (Step 6b): snapshot before the Delete fill ---
            undo.push_hmm(hmm_states, ch, mol)
            if left_state is None and right_state is None:
                arr[mol, f0:f1 + 1] = None
            elif left_state is None:
                arr[mol, f0:f1 + 1] = right_state
            elif right_state is None:
                arr[mol, f0:f1 + 1] = left_state
            else:
                f_mid = (f0 + f1) // 2
                arr[mol, f0      : f_mid + 1] = left_state
                arr[mol, f_mid + 1: f1 + 1  ] = right_state

            if ch == 'fret_g':
                app_mod.hmm_fret_g = hmm_states['fret_g']

            fig, new_segs = _build_from_hmm_states(
                ch, mol, t_arr, relayout_data,
                left_mode=left_mode, left_cut=left_cut, right_cut=right_cut,
            )
            return fig, new_segs, 'Boundary move', -1, None, -1   # clear vacuum + boundary-seg

        # ── Add: single boundary click → INSERT a new state at that boundary ──
        # Detects the nearest boundary (same logic as Boundary move).
        # A brand-new state is created by carving a slice of frames from the
        # START of the right-side segment.  The right segment is NOT fully
        # replaced — only a portion of it becomes the new state, so the rest
        # stays unchanged.  The user presses a digit to choose which fitted
        # state the new segment belongs to, then can fine-tune its width with
        # Boundary move.
        if tool == 'Add':
            if len(fresh_segs) < 2:
                raise PreventUpdate

            nearest_j = min(
                range(1, len(fresh_segs)),
                key=lambda j: abs(click_x - float(fresh_segs[j]['t_start']))
            )
            right_seg = fresh_segs[nearest_j]
            if right_seg['type'] != 'state':
                raise PreventUpdate

            # Carve a slice from the right segment: ~1/3 of its length, capped
            # at 10 frames and at least 1 frame.
            right_len = int(right_seg['f_end']) - int(right_seg['f_start']) + 1
            n_new     = min(10, max(1, right_len // 3))

            f_start = int(right_seg['f_start'])
            f_end   = f_start + n_new - 1        # new state ends here
            t_start = float(t_arr[f_start])
            t_end   = float(t_arr[f_end])

            new_vacuum = {
                'phase':   'vacuum_ready',
                'f_start': f_start,
                'f_end':   f_end,
                't_start': t_start,
                't_end':   t_end,
                'ch':      ch,
            }
            vt = (t_start, t_end)
            fig = make_hmm_bar_figure(fresh_segs, vs, ve, -1, vacuum_t=vt,
                                      excluded_indices=excluded)
            return fig, no_update, no_update, -1, new_vacuum, -1

        # ── Change state: click on a state block → show vacuum overlay ──────
        if tool == 'Change state':
            clicked_idx = None
            # Primary: customdata
            try:
                cd_val = click_data['points'][0].get('customdata')
                if cd_val is not None:
                    raw_idx = int(cd_val[0])
                    if 0 <= raw_idx < len(fresh_segs) and fresh_segs[raw_idx]['type'] == 'state':
                        clicked_idx = raw_idx
            except (KeyError, IndexError, TypeError, ValueError):
                pass

            # Fallback: range check
            if clicked_idx is None:
                for k, seg in enumerate(fresh_segs):
                    if seg['type'] != 'state':
                        continue
                    t_lo = float(seg['t_start'])
                    t_hi = (float(fresh_segs[k + 1]['t_start'])
                            if k + 1 < len(fresh_segs)
                            else float(seg['t_end']) + 1e-9)
                    if t_lo <= click_x < t_hi:
                        clicked_idx = k
                        break

            if clicked_idx is None:
                raise PreventUpdate

            sel_seg = fresh_segs[clicked_idx]
            vacuum_data = {
                'phase':   'vacuum_ready',
                'f_start': int(sel_seg['f_start']),
                'f_end':   int(sel_seg['f_end']),
                't_start': float(sel_seg['t_start']),
                't_end':   float(sel_seg['t_end']),
                'ch':      ch,
            }
            vt = (vacuum_data['t_start'], vacuum_data['t_end'])
            fig = make_hmm_bar_figure(fresh_segs, vs, ve, -1, vacuum_t=vt,
                                      excluded_indices=excluded)
            # Don't update segments store — hmm_states unchanged yet
            return fig, no_update, no_update, no_update, vacuum_data, no_update

        raise PreventUpdate

    # ── Click on trace graph → move selected boundary ────────────────────────
    # Fires when user clicks the main Plotly figure while on HMM tab.
    # update_fig already raises PreventUpdate for this combination, so no
    # breakpoints are added and no full figure redraw occurs.
    @app.callback(
        Output('hmm-state-bar',          'figure', allow_duplicate=True),
        Output('hmm-segments-store',     'data',   allow_duplicate=True),
        Output('hmm-selected-boundary',  'data',   allow_duplicate=True),
        Output('hmm-boundary-seg',       'data',   allow_duplicate=True),
        Input('graph', 'clickData'),
        State('hmm-selected-boundary',  'data'),
        State('hmm-boundary-seg',       'data'),
        State('i',                      'value'),
        State('hmm-channel',            'value'),
        State('tabs',                   'value'),
        State('graph',                  'relayoutData'),
        State('dwell-left-mode',        'value'),
        State('dwell-left-cut',         'value'),
        State('dwell-right-cut',        'value'),
        prevent_initial_call=True,
    )
    def hmm_trace_click(click_data, sel_boundary, bar_seg,
                        i, channel, active_tab, relayout_data,
                        left_mode, left_cut, right_cut):
        """Move the selected HMM boundary to the clicked data point.

        Direction rule (segment-identity-based, NOT position-based):
          • bar_seg == bidx+1  → user clicked LEFT half of seg_right → expand seg_right LEFT
            Fill frames [f_click … f_boundary-1] with seg_right's state.
            Valid only when f_click < f_boundary (clicking inside seg_right = no-op).
          • bar_seg == bidx    → user clicked RIGHT half of seg_left → expand seg_left RIGHT
            Fill frames [f_boundary … f_click] with seg_left's state.
            Valid only when f_click >= f_boundary (clicking inside seg_left = no-op).
        """
        if active_tab != 'HMM':
            raise PreventUpdate
        if sel_boundary is None or int(sel_boundary) < 0:
            raise PreventUpdate
        if not click_data:
            raise PreventUpdate

        try:
            click_x = float(click_data['points'][0]['x'])
        except (KeyError, IndexError, TypeError):
            raise PreventUpdate

        ch       = channel or 'fret_g'
        time_key = CHANNEL_TIME_KEY.get(ch, 'fret_g')
        t_arr    = app_mod.time.get(time_key, [])
        if len(t_arr) == 0:
            raise PreventUpdate

        hmm_states = app_mod.hmm_states
        if ch not in hmm_states:
            raise PreventUpdate

        t_np = np.array(t_arr, dtype=float)
        mol  = int(i)

        # Build fresh segments from hmm_states — no store dependency
        fresh_trace = np.asarray(hmm_states[ch][mol], dtype=object)
        fresh_segs  = trace_to_segments(fresh_trace, t_arr)
        fresh_segs  = merge_same_state_segments(fresh_segs, hmm_states[ch][mol])

        bidx = int(sel_boundary)
        if bidx + 1 >= len(fresh_segs):
            raise PreventUpdate

        seg_left  = fresh_segs[bidx]
        seg_right = fresh_segs[bidx + 1]

        # Out-of-range guard: ignore clicks clearly outside the two adjacent segments
        if click_x > float(seg_right['t_end']) + 1e-9:
            raise PreventUpdate
        if click_x < float(seg_left['t_start']) - 1e-9:
            raise PreventUpdate

        f_click    = int(np.argmin(np.abs(t_np - click_x)))
        f_boundary = int(seg_right['f_start'])   # first frame of right segment

        # Determine expand direction from which segment was clicked on the bar.
        # bar_seg == bidx+1 → left half of seg_right was clicked → expand seg_right LEFT
        # bar_seg == bidx   → right half of seg_left was clicked → expand seg_left RIGHT
        # bar_seg == -1     → fallback: infer from click position (legacy behaviour)
        clicked_seg = int(bar_seg) if bar_seg is not None else -1

        arr = hmm_states[ch]

        # --- UNDO (Step 6a): snapshot this molecule's state row before the
        # boundary edit.  f_click == f_boundary is a no-op in every branch
        # below (raises PreventUpdate), so skip the snapshot in that case to
        # avoid wasting an undo slot.
        if f_click != f_boundary:
            undo.push_hmm(hmm_states, ch, mol)

        if clicked_seg == bidx + 1:
            # ── Clicked seg_right: boundary snaps to f_click ─────────────────
            if f_click == f_boundary:
                raise PreventUpdate
            elif f_click < f_boundary:
                # Move boundary LEFT — fill [f_click : f_boundary] with seg_right's state
                fill = (np.nan  if seg_right['type'] == 'pb'
                        else None if seg_right['type'] == 'no_prediction'
                        else int(seg_right['state']))
                arr[mol, f_click:f_boundary] = fill
            else:
                # Move boundary RIGHT — fill [f_boundary : f_click+1] with seg_left's state
                fill = (np.nan  if seg_left['type'] == 'pb'
                        else None if seg_left['type'] == 'no_prediction'
                        else int(seg_left['state']))
                arr[mol, f_boundary:f_click + 1] = fill

        elif clicked_seg == bidx:
            # ── Clicked seg_left: boundary snaps to f_click ──────────────────
            if f_click == f_boundary:
                raise PreventUpdate
            elif f_click >= f_boundary:
                # Move boundary RIGHT — fill [f_boundary : f_click+1] with seg_left's state
                fill = (np.nan  if seg_left['type'] == 'pb'
                        else None if seg_left['type'] == 'no_prediction'
                        else int(seg_left['state']))
                arr[mol, f_boundary:f_click + 1] = fill
            else:
                # Move boundary LEFT — fill [f_click : f_boundary] with seg_right's state
                fill = (np.nan  if seg_right['type'] == 'pb'
                        else None if seg_right['type'] == 'no_prediction'
                        else int(seg_right['state']))
                arr[mol, f_click:f_boundary] = fill

        else:
            # ── Fallback: position-based (bar_seg unknown) ───────────────────
            if f_click < f_boundary:
                fill = (np.nan  if seg_right['type'] == 'pb'
                        else None if seg_right['type'] == 'no_prediction'
                        else int(seg_right['state']))
                arr[mol, f_click:f_boundary] = fill
            elif f_click > f_boundary:
                fill = (np.nan  if seg_left['type'] == 'pb'
                        else None if seg_left['type'] == 'no_prediction'
                        else int(seg_left['state']))
                arr[mol, f_boundary:f_click + 1] = fill
            else:
                raise PreventUpdate

        if ch == 'fret_g':
            app_mod.hmm_fret_g = hmm_states['fret_g']

        fig, new_segs = _build_from_hmm_states(
            ch, mol, t_arr, relayout_data,
            left_mode=left_mode, left_cut=left_cut, right_cut=right_cut,
        )
        return fig, new_segs, -1, -1

    # ── Coloured state windows on the main trace graph ───────────────────────
    # Semi-transparent rectangles spanning each HMM segment, drawn on the
    # correct panel for each channel:
    #   bb      → top panel    (y4 domain)
    #   gg, rr  → second panel (y3 domain)
    #   fret_b  → third panel  (y2 domain)
    #   fret_g  → bottom panel (y1 domain)
    #
    # Triggered by both hmm-segments-store (any state change) AND tabs (so we
    # can clear shapes when the user leaves the HMM tab).
    # Uses Patch() so the full figure is never re-rendered.
    @app.callback(
        Output('graph', 'figure', allow_duplicate=True),
        Input('hmm-segments-store', 'data'),   # any HMM state / trace change
        Input('tabs', 'value'),                # tab switch → clear on non-HMM tabs
        Input('dwell-left-mode', 'value'),     # cut / dead → changes left exclusions
        Input('dwell-left-cut',  'value'),     # left N to gray out (cut mode)
        Input('dwell-right-cut', 'value'),     # right N to gray out
        State('i', 'value'),
        State('hmm-channel', 'value'),
        prevent_initial_call=True,
    )
    def update_color_windows(_segments, active_tab,
                             left_mode, left_cut, right_cut,
                             i, channel):
        """Overlay semi-transparent coloured rectangles on the intensity/FRET
        panel that corresponds to the **currently selected** HMM channel.
        Windows are removed when the user switches away from the HMM tab or
        when no HMM data exists for the selected channel.
        """
        p = Patch()
        p['layout']['shapes'] = []   # always start clean

        if active_tab != 'HMM':
            return p

        ch = channel or 'fret_g'

        # Map channel selector value → (Plotly y-domain ref, time-dict key)
        CHANNEL_PANEL = {
            'bb':     ('y4 domain', 'b'),
            'gg':     ('y3 domain', 'g'),
            'rr':     ('y3 domain', 'r'),
            'fret_b': ('y2 domain', 'fret_b'),
            'fret_g': ('y1 domain', 'fret_g'),
        }
        panel = CHANNEL_PANEL.get(ch)
        if panel is None:
            return p
        yref, time_key = panel

        source = app_mod.hmm_states.get(ch)
        if source is None:
            return p   # no HMM data for this channel → no windows

        mol   = int(i) if i is not None else 0
        t_arr = app_mod.time.get(time_key, [])
        if len(t_arr) == 0:
            return p

        try:
            trace = np.asarray(source[mol], dtype=object)
        except Exception:
            return p

        segs = trace_to_segments(trace, t_arr)
        excluded = get_excluded_indices(
            segs, left_mode or 'cut',
            int(left_cut  or 0),
            int(right_cut or 0),
        )
        shapes = []
        for k, seg in enumerate(segs):
            if seg['type'] == 'no_prediction':
                continue    # leave un-predicted regions un-coloured
            if seg['type'] == 'pb':
                fc = 'rgba(136,136,136,0.18)'
            else:
                is_excluded = k in excluded
                idx  = seg['state'] % len(STATE_COLORS)
                base = STATE_GRAYS[idx] if is_excluded else STATE_COLORS[idx]
                fc   = _hex_to_rgba(base, 0.18)

            # Place visual boundaries at frame midpoints so each data point
            # falls unambiguously inside its own colour block.
            # Without this, the orange window extends to t_arr[f_boundary] and
            # the data point there looks orange even though it belongs to blue.
            x0 = float(seg['t_start'])
            x1 = float(seg['t_end'])
            if k > 0:
                x0 = (float(segs[k - 1]['t_end']) + x0) / 2.0
            if k + 1 < len(segs):
                x1 = (x1 + float(segs[k + 1]['t_start'])) / 2.0

            shapes.append(dict(
                type='rect',
                xref='x',    # data coords shared by all x-axes
                yref=yref,   # e.g. 'y3 domain' → full height of that panel
                x0=x0, x1=x1,
                y0=0, y1=1,
                fillcolor=fc,
                line=dict(width=0),
                layer='below',
            ))

        p['layout']['shapes'] = shapes
        return p

    # ════════════════════════════════════════════════════════════════════════
    # Callback 4 — ADD-TOOL KEYBOARD HANDLER
    # When a vacuum is pending, listens for digit keys 0–9.
    # Assigns the clicked segment to the chosen state's fitted mean.
    # Ignores keys that are out of range (> number of fitted states).
    # ════════════════════════════════════════════════════════════════════════
    @app.callback(
        Output('hmm-state-bar',           'figure',   allow_duplicate=True),
        Output('hmm-segments-store',      'data',     allow_duplicate=True),
        Output('hmm-vacuum-store',        'data',     allow_duplicate=True),
        Output('hmm-fitted-states-table', 'columns',  allow_duplicate=True),
        Output('hmm-fitted-states-table', 'data',     allow_duplicate=True),
        Input('key_events', 'n_events'),
        State('key_events',       'event'),
        State('hmm-vacuum-store', 'data'),
        State('i',                'value'),
        State('hmm-channel',      'value'),
        State('tabs',             'value'),
        State('graph',            'relayoutData'),
        State('dwell-left-mode',  'value'),
        State('dwell-left-cut',   'value'),
        State('dwell-right-cut',  'value'),
        prevent_initial_call=True,
    )
    def hmm_add_keypress(n_events, event, vacuum, i, channel,
                         active_tab, relayout_data,
                         left_mode, left_cut, right_cut):
        """Assign the pending vacuum state to a fitted HMM state on digit keypress.

        Also handles Ctrl+Z (undo the last HMM state amendment).  The undo
        logic is merged into this callback on purpose: dash_extensions.enrich
        derives the allow_duplicate output suffix from the callback's input
        list, so a *separate* callback sharing the sole input
        'key_events.n_events' and the same outputs ('hmm-state-bar.figure',
        'hmm-segments-store.data') would produce identical output IDs and
        crash the whole callback system with "Duplicate callback outputs".
        """
        if active_tab != 'HMM':
            raise PreventUpdate

        # ── Ctrl+Z → undo the last HMM state amendment ───────────────────────
        if undo.is_ctrl_z(event):
            snapshot = undo.pop_hmm()
            if snapshot is None:
                raise PreventUpdate            # nothing left to undo

            scope = snapshot.get('scope')      # 'channel' or None (single-mol)

            if scope == 'channel':
                # ── Channel-scope undo: Merge States or HMM Clear ────────────
                restored_ch = undo.restore_hmm_channel(
                    app_mod.hmm_states, app_mod.hmm_means, snapshot
                )
                if restored_ch is None:
                    raise PreventUpdate
                if restored_ch == 'fret_g':
                    app_mod.hmm_fret_g = app_mod.hmm_states['fret_g']
                ch_u    = channel or 'fret_g'
                t_arr_u = app_mod.time.get(CHANNEL_TIME_KEY.get(ch_u, 'fret_g'), [])
                if len(t_arr_u) == 0:
                    raise PreventUpdate
                mol_u = int(i) if i is not None else 0
                fig_u, segs_u = _build_from_hmm_states(
                    ch_u, mol_u, t_arr_u, relayout_data,
                    left_mode=left_mode, left_cut=left_cut, right_cut=right_cut,
                )
                # Rebuild fitted-states table so it reflects the restored means
                means = app_mod.hmm_means.get(ch_u)
                if means is not None and len(means) > 0:
                    tbl_cols = [{'id': f'S{i}', 'name': f'S{i}'} for i in range(len(means))]
                    tbl_data = [{f'S{i}': round(float(m), 4) for i, m in enumerate(means)}]
                else:
                    tbl_cols, tbl_data = [], []
                return fig_u, segs_u, no_update, tbl_cols, tbl_data

            else:
                # ── Single-molecule undo: vacuum fill or boundary edit ────────
                restored_ch = undo.restore_hmm(app_mod.hmm_states, snapshot)
                if restored_ch is None:
                    raise PreventUpdate        # snapshot no longer applies
                if restored_ch == 'fret_g':
                    app_mod.hmm_fret_g = app_mod.hmm_states['fret_g']
                ch_u    = channel or 'fret_g'
                t_arr_u = app_mod.time.get(CHANNEL_TIME_KEY.get(ch_u, 'fret_g'), [])
                if len(t_arr_u) == 0:
                    raise PreventUpdate
                mol_u = int(i) if i is not None else 0
                fig_u, segs_u = _build_from_hmm_states(
                    ch_u, mol_u, t_arr_u, relayout_data,
                    left_mode=left_mode, left_cut=left_cut, right_cut=right_cut,
                )
                return fig_u, segs_u, no_update, no_update, no_update  # table unchanged

        # Only act when the vacuum is fully defined (both boundaries chosen)
        if vacuum is None or vacuum.get('phase') != 'vacuum_ready':
            raise PreventUpdate
        if not event or 'key' not in event:
            raise PreventUpdate

        key = event['key']
        if key not in '0123456789':
            raise PreventUpdate   # non-digit — keep waiting

        num = int(key)
        ch = vacuum.get('ch') or channel or 'fret_g'

        means = app_mod.hmm_means.get(ch)
        if means is None or num >= len(means):
            raise PreventUpdate   # out of range — ignore silently

        # Fill the vacuum frames with the chosen state index
        mol = int(i)
        f0  = int(vacuum['f_start'])
        f1  = int(vacuum['f_end'])
        if ch not in app_mod.hmm_states:
            raise KeyError(f"Channel '{ch}' has no HMM prediction — run HMM fitting first. Tell 🐎")
        arr = app_mod.hmm_states[ch]
        # --- UNDO (Step 6b): snapshot before the Add / Change-state fill ---
        undo.push_hmm(app_mod.hmm_states, ch, mol)
        arr[mol, f0:f1 + 1] = num   # store int state index directly
        if ch == 'fret_g':
            app_mod.hmm_fret_g = app_mod.hmm_states['fret_g']

        time_key = CHANNEL_TIME_KEY.get(ch, 'fret_g')
        t_arr    = app_mod.time.get(time_key, [])
        if len(t_arr) == 0:
            raise PreventUpdate

        # Rebuild bar without vacuum — assignment is now part of hmm_states
        fig, new_segs = _build_from_hmm_states(
            ch, mol, t_arr, relayout_data,
            left_mode=left_mode, left_cut=left_cut, right_cut=right_cut,
        )
        return fig, new_segs, None, no_update, no_update  # clear vacuum; table unchanged

    # ════════════════════════════════════════════════════════════════════════
    # Callback 5 — FITTED STATES SUMMARY TABLE
    # Read-only table at the bottom of the HMM Prediction column.
    # Reflects the current hmm_means for the selected channel; updates after
    # every fit or load, and whenever the channel dropdown changes.
    # ════════════════════════════════════════════════════════════════════════
    @app.callback(
        Output('hmm-fitted-states-table', 'columns'),
        Output('hmm-fitted-states-table', 'data'),
        Input('hmm_means',       'data'),    # updated by both fit and load callbacks
        Input('hmm-save-status', 'children'),# also fires on load (belt-and-suspenders)
        Input('hmm-channel',     'value'),
        prevent_initial_call=True,
    )
    def update_fitted_states_table(hmm_means_data, save_status, channel):
        """Populate the read-only fitted-states table from hmm_means.

        Orientation matches the 'Initial means' table: one column per state
        (header = S0, S1, …) and a single data row of mean values.
        """
        ch    = channel or 'fret_g'
        means = app_mod.hmm_means.get(ch)
        if means is None or len(means) == 0:
            return [], []

        cols = [{'id': f'S{i}', 'name': f'S{i}'} for i in range(len(means))]
        row  = {f'S{i}': round(float(m), 4) for i, m in enumerate(means)}
        return cols, [row]
