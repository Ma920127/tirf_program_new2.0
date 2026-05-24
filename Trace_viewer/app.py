import sys
import numpy as np
import os
import shutil
import time as rtime
from dash import Dash, dcc, html, Input, Output, State, callback_context, no_update, DiskcacheManager, callback
from dash.exceptions import PreventUpdate
from layout import make_app
import utils.HMM_slidebar as hmm_bar
from utils.HMM_slidebar import trace_to_segments, make_hmm_bar_figure, merge_same_state_segments
from utils.trace import update_trace, change_trace
from utils.selection import select_good_bad, select_colocalized, render_good_bad, render_colocalized
from utils.breakpoints import breakpoints_utils, sl_bkps, find_chp
from utils.blob import show_blob
from Gaussian_mixture.gmm import fit_gmm, draw_gmm, save_gmm
from utils.plotting import plot_fret_trace
from utils.draw import draw
from init_fig import init_fig
from loader import Loader 
from Hidden_Markov.hmm_fitter_new import HMM_fitter
from utils.dwell_manager import DwellManager
from utils.dwell_calculator import _extract_dwells
from utils.transition_counter import save_transition_txt
import utils.undo_manager as undo
from Rupture import Rupture
from utils.smoothing import uf, sa, mf, sg
import plotly.graph_objects as go

path = ''
fret_g = np.zeros(0)
fret_b = np.zeros(0)
rr = np.zeros(0)
gg = np.zeros(0)
gr = np.zeros(0) 
bb = np.zeros(0)
bg = np.zeros(0)
br = np.zeros(0)
time_r =np.zeros(0)
time_g =np.zeros(0)
time_b = np.zeros(0)
hmm_fret_g = np.full(1000, None, dtype=object)
tot_g = gg + gr
tot_b = bb + bg + br
select_list_g = np.zeros(0)
colocalized_list = np.zeros(0)
bmode = 1
N_traces = 0
total_frame = 0
idx='N/A'
new = 0
blobs = None
gmm = None
ch_label = 'fret_g'
fret_g_bkps= []
fret_b_bkps= []
b_bkps = []
g_bkps = []
r_bkps = []
bkps = {
        'fret_g' : fret_g_bkps,
        'fret_b' : fret_b_bkps,
        'b' : b_bkps,
        'g' : g_bkps,
        'r' : r_bkps,
    }
time = {
    'fret_b' : time_b,
    'fret_g' : time_g,
    'b' : time_b,
    'g' : time_g,
    'r' : time_r,
}
tot_dtime=[]

    
color=["#fff",'yellow']
rup_bkps = []
hmm_mask   = None
hmm_states = {}   # {channel: 2-D object array} — populated by Apply Filter / HMM fit
hmm_means  = {}   # {channel: 1-D float array}  — fitted means, populated after HMM fit


fig, fig_blob, fig2 = init_fig() 
app = make_app(fig, fig_blob, fig2)

# --- NEW CALLBACK: UI Toggle for Savitzky-Golay ---
@app.callback(
    [Output('poly-container', 'style'),
     Output('smooth', 'step')],
    Input('smooth_method', 'value')
)
def toggle_poly_input(method):
    if method == 'savgol':
        # Show poly order, force window step to 2 (keeps window size odd)
        return {'display': 'flex', 'flex-direction': 'row', 'align-items': 'center'}, 2
    # Hide poly order, regular step 1
    return {'display': 'none'}, 1
# --------------------------------------------------


@app.callback(
    Output('graph', 'figure'),
    Output('i','value'),
    Output('bkp','value'),
    Output('b_bkp','value'),
    Output('AR','value'),
    Output('N_traces','children'),
    Output('set_good','style'),
    Output('set_bad','style'),
    Output('set_colocalized','style'),
    Output('channel', 'options'),
    Output('chp_channel_0', 'options'),
    Output('chp_channel_1', 'options'),
    Output('confirm-reset', 'displayed'),
    Output('channel-error', 'is_open'),
    Output('graph', 'relayoutData'),
    Input('key_events', 'n_events'),
    Input('show', 'value'),
    Input('next', 'n_clicks'),
    Input('previous', 'n_clicks'),
    Input('tr_go', 'n_clicks'),
    Input('dtime','n_clicks'),
    Input('etime','n_clicks'),
    Input('graph', 'clickData'),
    Input('AR','value'),
    Input('save_bkps','n_clicks'),
    Input('load_bkps','n_clicks'),
    Input('loadp', 'n_clicks'),
    Input('set_good','n_clicks'),
    Input('set_bad','n_clicks'),
    Input('set_colocalized','n_clicks'),
    Input('select','n_clicks'),
    Input('scatter', 'value'),
    Input('smooth', 'value'),
    Input('smooth_method', 'value'),
    Input('polyorder', 'value'),
    Input('rescale', 'n_clicks'),
    Input("graph", "relayoutData"),
    Input('channel', 'value'),
    Input('chp_find_0','n_clicks'),
    Input('chp_find_1','n_clicks'),
    Input('confirm-reset', 'submit_n_clicks'),
    Input('rup-status-output', 'children'),
    Input('tabs', 'value'),
    State('i','value'),
    State('path','value'),
    State('chp_mode_0', 'value'),
    State('chp_comp_0', 'value'),
    State('chp_thres_0', 'value'),
    State('chp_channel_0', 'value'),
    State('chp_target_0', 'value'),
    State('chp_mode_1', 'value'),
    State('chp_comp_1', 'value'),
    State('chp_thres_1', 'value'),
    State('chp_channel_1', 'value'),
    State('chp_target_1', 'value'),
    State('key_events', 'event'),
    State('rup-channel', 'value')
    )

def update_fig(key_events, show, next, previous, go, dtime, etime, clickData, mode, save, load, loadp, good, bad, coloc, select, scatter, smooth, smooth_method, polyorder, rescale, relayout, channel, chp_find_0, chp_find_1, confirm_reset, rup_status, active_tab, i, path,
               chp_mode_0, chp_comp_0, chp_thres_0, chp_channel_0, chp_target_0, chp_mode_1, chp_comp_1, chp_thres_1, chp_channel_1, chp_target_1, event,rup_channel):
    changed_id = [p['prop_id'] for p in callback_context.triggered][0]
    global N_traces, fig, fig2, idx, total_frame, color, good_style, bad_style, bmode
    global new, time_b, N_traces, total_frame, tot_dtime
    global fret_g, fret_b, rr, gg, gr, bb, bg, br, time, tot_g, tot_b, hmm_fret_g
    global select_list_g, colocalized_list, bkps, ch_label, blobs
    if ('n_events' in changed_id) and (event['key'] not in ['q', 'w', 'z', 'x', 'c']):
        raise PreventUpdate()

    # Ctrl+Z is reserved for the dedicated undo callbacks — update_fig must
    # ignore it, otherwise the plain 'z' branch below would mark the trace
    # as Good.  The undo callbacks handle the actual revert.
    if ('n_events' in changed_id) and undo.is_ctrl_z(event):
        raise PreventUpdate()

    # On HMM tab, trace-graph clicks are handled exclusively by hmm_trace_click.
    # Skip update_fig so breakpoints are not added and the figure is not needlessly redrawn.
    if active_tab == 'HMM' and changed_id == 'graph.clickData':
        raise PreventUpdate()

    # Pure zoom / pan — Plotly.js already updates the viewport client-side.
    # Re-sending the full figure here would wipe the colour-window shapes that
    # update_color_windows applied via Patch(), causing them to disappear.
    if changed_id == 'graph.relayoutData':
        raise PreventUpdate()

    if fig['layout']['uirevision'] == False:
        fig['layout']['uirevision'] = True   
    
    # Map smooth_method directly to smooth_mode
    smooth_mode = smooth_method

    ##load path##
    if 'loadp' in changed_id:
        fret_g, fret_b, rr, gg, gr, bb, bg, br, time, tot_g, tot_b, N_traces, total_frame, bkps, select_list_g, colocalized_list, ch_label, blobs, hmm_fret_g = Loader(path).load_traces()
        tot_dtime = []
        i = 0
        new = 1
        undo.clear_all()   # new dataset — drop stale undo history

    ##rescale##
    if 'rescale' in changed_id:
        relayout = {'autosize' : True }
        fig['layout']['uirevision'] = False

    ##change trace##
    old_i = int(i) if i is not None else 0
    i, fig = change_trace(changed_id, event, i, N_traces, fig)
    new_i = int(i) if i is not None else 0

    # Navigation key pressed but index didn't change (e.g. 'q' at mol 0, 'w' at last mol).
    # Nothing will differ in the figure — bail early so the full figure is not pushed
    # back, which would wipe out the color-window Patch shapes applied by update_color_windows.
    if 'n_events' in changed_id and event.get('key') in ('q', 'w') and new_i == old_i:
        raise PreventUpdate()

    ##select good bad##
    select_list_g = select_good_bad(changed_id, event, i, select_list_g)
    colocalized_list = select_colocalized(changed_id, event, i, colocalized_list)

    good_style, bad_style = render_good_bad(i, select_list_g)
    colocalized_style = render_colocalized(i, colocalized_list)

    #save select##        
    if 'select' in changed_id:  
        np.save(path+r'/selected_g.npy', select_list_g)
        print('selected_g save')
        np.save(path+r'/colocalized_list.npy', colocalized_list)
     
    ## --- UNDO (Step 4a): snapshot bkps before breakpoints_utils mutates it.
    ## Only on the breakpoint-editing tabs; the Rupture/HMM tabs are excluded
    ## so Ctrl+Z stays a no-op there.  A graph click resolves its channel
    ## inside breakpoints_utils (from the curve number), so a click cannot be
    ## scoped to one channel here — snapshot the whole dict instead.
    if active_tab in ('Tools', 'Aois', 'GMM'):
        if 'graph.clickData' in changed_id:
            undo.push_bkps_all(bkps)
        elif ('dtime' in changed_id) or ('etime' in changed_id):
            undo.push_bkps_single(bkps, channel, i)
        elif 'confirm-reset' in changed_id:
            undo.push_bkps_all(bkps)
        elif mode in ('Clear All', 'Set All'):
            undo.push_bkps_all(bkps)
        elif mode == 'Clear':
            undo.push_bkps_single(bkps, channel, i)

    ##breakpoints## (Added polyorder)
    bkps, mode, confirm_reset_show, channel_error_show = breakpoints_utils(changed_id, clickData, mode, channel, i, time, bkps, smooth, smooth_mode, polyorder)
    
    ##save / load breakpoints##
    bkps = sl_bkps(changed_id, path, bkps, mode)

    ## --- UNDO (Step 4b): snapshot bkps before find_chp (auto-detect) mutates
    ## it.  'current trace' touches one channel/one trace → single snapshot;
    ## 'all traces'/'all good' touch many traces → whole-dict snapshot.
    if active_tab in ('Tools', 'Aois', 'GMM'):
        if 'chp_find_0' in changed_id:
            if chp_target_0 == 'current trace':
                undo.push_bkps_single(bkps, chp_channel_0, i)
            else:
                undo.push_bkps_all(bkps)
        elif 'chp_find_1' in changed_id:
            if chp_target_1 == 'current trace':
                undo.push_bkps_single(bkps, chp_channel_1, i)
            else:
                undo.push_bkps_all(bkps)

    # Auto-Detect Breakpoints (Added polyorder)
    bkps, channel_error_show = find_chp(changed_id, fret_g, fret_b, rr, gg, gr, bb, bg, br, i, time, select_list_g,
             chp_mode_0, chp_comp_0, chp_thres_0, chp_channel_0, chp_target_0, chp_mode_1, chp_comp_1, chp_thres_1, chp_channel_1, chp_target_1,
             bkps, smooth, smooth_mode, polyorder)

    # ---------------------------------------------------------
    # UI TRICK: HIDE RED DOTS ON RUPTURE / HMM TABS
    # On the Rupture tab the purple preview dots replace the red bkp dots.
    # On the HMM tab the colour-window overlays are shown instead — red dots
    # would overlap them and confuse the state view.
    if active_tab in ('Rupture', 'HMM'):
        empty_traces = [[] for _ in range(max(1, N_traces))]
        display_bkps = {
            'r': empty_traces,
            'g': empty_traces,
            'b': empty_traces,
            'fret_g': empty_traces,
            'fret_b': empty_traces
        }
    else:
        display_bkps = bkps  # Real dictionary -> Red dots draw normally!
    # ---------------------------------------------------------

    ##update trace## (Added polyorder)
    fig = update_trace(fig, relayout, i, scatter, fret_g, fret_b, rr, gg, gr, bb, bg, br, time, display_bkps, smooth, smooth_mode, show, polyorder=polyorder,
                       rup_bkps=rup_bkps,
                       channel=rup_channel,
                       active_tab=active_tab)

    ##Display Information##
    if np.any(np.array(bkps['fret_g'], dtype = object)):
        str_g_bkps = ', '.join(str(round(x[1], 2)) for x in bkps['fret_g'][i])
    else:
        str_g_bkps = ''
    
    if channel!= None:
        if np.any(np.array(bkps[channel], dtype = object)):    
            str_b_bkps = ', '.join(str(round(x[1], 2)) for x in bkps[channel][i])
        else:
            str_b_bkps = ''
    else:
        str_b_bkps = ''
    nnote='Total_traces: ' + str(N_traces)

    # Only re-emit `i` when the molecule actually changed — prevents Dash from
    # re-triggering render_hmm_bar (and other i-watchers) on zoom/pan events.
    # Only re-emit relayoutData on explicit rescale so the HMM bar isn't
    # woken up again after update_fig finishes, which would override the zoom.
    i_out = i if new_i != old_i else no_update
    relayout_out = relayout if 'rescale' in changed_id else no_update
    return fig, i_out, str_g_bkps, str_b_bkps, mode, nnote, good_style, bad_style, colocalized_style, ch_label, ch_label, ch_label, confirm_reset_show, channel_error_show, relayout_out


# ══════════════════════════════════════════════════════════════════════════════
# Undo (Ctrl+Z) — breakpoint edits  [Tools / Aois / GMM tabs]
# ══════════════════════════════════════════════════════════════════════════════
@app.callback(
    Output('graph', 'figure', allow_duplicate=True),
    Input('key_events', 'n_events'),
    State('key_events', 'event'),
    State('tabs', 'value'),
    State('i', 'value'),
    State('scatter', 'value'),
    State('smooth', 'value'),
    State('smooth_method', 'value'),
    State('polyorder', 'value'),
    State('show', 'value'),
    State('rup-channel', 'value'),
    prevent_initial_call=True,
)
def undo_bkps_callback(n_events, event, active_tab, i, scatter, smooth,
                       smooth_method, polyorder, show, rup_channel):
    """Ctrl+Z on the Tools / Aois / GMM tabs — revert the last breakpoint edit.

    HMM-tab undo is handled separately in utils/HMM_slidebar.py; this callback
    deliberately fires only on the breakpoint-editing tabs.
    """
    # Only act on Ctrl+Z, and only on the breakpoint-editing tabs.
    if not undo.is_ctrl_z(event):
        raise PreventUpdate
    if active_tab not in ('Tools', 'Aois', 'GMM'):
        raise PreventUpdate

    snapshot = undo.pop_bkps()
    if snapshot is None:
        raise PreventUpdate          # nothing left to undo

    global bkps, fig
    undo.restore_bkps(bkps, snapshot)

    # Re-render the trace graph with the reverted breakpoints.  relayout=None
    # keeps the current zoom (the figure carries uirevision=True).
    i = int(i) if i is not None else 0
    fig = update_trace(fig, None, i, scatter,
                       fret_g, fret_b, rr, gg, gr, bb, bg, br, time,
                       bkps, smooth, smooth_method, show,
                       polyorder=polyorder,
                       rup_bkps=rup_bkps,
                       channel=rup_channel,
                       active_tab=active_tab)
    return fig


@app.callback(
    Output('plot_status', 'children'),
    Input('plot_fret_g', 'n_clicks'),
    State('i', 'value'),
    State('path', 'value')
)
def plot_and_save_fret_g(n_clicks, current_index, path):
    if n_clicks is None or n_clicks == 0:
        raise PreventUpdate

    # Validate the current trace index.
    try:
        idx = int(current_index)
    except (ValueError, TypeError):
        return "Invalid trace index."
    
    global fret_g, time  # Assuming these globals are already defined and populated.
    try:
        trace = fret_g[idx]
    except IndexError:
        return f"Trace index {idx} is out of range."

    # Use the appropriate time array for fret_g (assuming stored in time['fret_g']).
    t = time['fret_g']
    
    # Define the base path (default to current directory if none provided).
    base_path = path if path else "."
    
    # Call the helper function to plot and save the figure.
    file_path = plot_fret_trace(t, trace, idx, base_path=base_path)
    
    return f"FRET_g trace {idx} plotted and saved to {file_path}"



@app.callback(
    Output('g_blob','figure'),
    Input('graph', 'hoverData'), 
    Input('aoi_max', 'value'),
    Input('i','value'),
    State('tabs', 'value'), 
    State('smooth', 'value'), 
    State('smooth_method', 'value'), # Changed from strided to smooth_method
    State('polyorder', 'value'),
    blocking = True  
    )
def show_blob_main(hoverData, aoi_max, i, tabs, smooth, smooth_method, polyorder): # <--- 2. Add it here
    global fig_blob, time
    
    if tabs == 'Aois':
        # 3. Instead of just calculating a 'strided' boolean, pass the exact 
        # smooth_method and polyorder straight into your show_blob function!
        fig_blob = show_blob(blobs, fig_blob, smooth, i, hoverData, time, aoi_max, smooth_method, polyorder)
    else:
        return fig_blob
        
    return fig_blob


# ══════════════════════════════════════════════════════════════════════════════
# 3.  Channel selector → show the correct filter panel
#     Also triggered by tab switch so it fires even with prevent_initial_callbacks
# ══════════════════════════════════════════════════════════════════════════════
@app.callback(
    Output('hmm-pb-fretg-panel',  'style'),
    Output('hmm-pb-fretb-panel',  'style'),
    Output('hmm-pb-single-panel', 'style'),
    Input('hmm-channel', 'value'),
    Input('hmm-pb-enable', 'value'),
    Input('tabs', 'value'),
)
def show_filter_panel(channel, pb_on, active_tab):
    hidden = {'display': 'none'}
    shown  = {'display': 'block', 'marginTop': '6px'}

    # Only show panels when on HMM tab AND filter is ON
    if not pb_on or active_tab != 'HMM':
        return hidden, hidden, hidden

    if channel == 'fret_g':
        return shown, hidden, hidden
    elif channel == 'fret_b':
        return hidden, shown, hidden
    else:                          # gg, bb, rr
        return hidden, hidden, shown


# ══════════════════════════════════════════════════════════════════════════════
# 4.  PB toggle → update "ON / OFF" label
# ══════════════════════════════════════════════════════════════════════════════
@app.callback(
    Output('hmm-pb-toggle-label', 'children'),
    Output('hmm-pb-toggle-label', 'style'),
    Input('hmm-pb-enable', 'value'),
)
def pb_label(on):
    base = {'fontSize': '11px', 'fontWeight': 'bold', 'marginLeft': '-4px',
            'fontFamily': 'Arial, sans-serif'}
    if on:
        return 'ON',  {**base, 'color': '#55A868'}
    return 'OFF', {**base, 'color': '#868e96'}


# ══════════════════════════════════════════════════════════════════════════════
# 4b. Selected-filter toggle → update "ON / OFF" label
# ══════════════════════════════════════════════════════════════════════════════
@app.callback(
    Output('hmm-selected-filter-label', 'children'),
    Output('hmm-selected-filter-label', 'style'),
    Input('hmm-selected-filter', 'value'),
)
def selected_filter_label(on):
    base = {'fontSize': '11px', 'fontWeight': 'bold', 'marginLeft': '-4px',
            'fontFamily': 'Arial, sans-serif'}
    if on:
        return 'ON',  {**base, 'color': '#4C72B0'}
    return 'OFF', {**base, 'color': '#868e96'}


# ══════════════════════════════════════════════════════════════════════════════
# 5.  RR sub-toggle → enable / disable the RR threshold input (fret_g panel)
# ══════════════════════════════════════════════════════════════════════════════
@app.callback(
    Output('hmm-pb-ming-rr', 'disabled'),
    Input('hmm-pb-use-rr', 'value'),
)
def toggle_rr_input_g(use_rr):
    return not use_rr


@app.callback(
    Output('hmm-pb-minb-rr', 'disabled'),
    Input('hmm-pb-use-rr-b', 'value'),
)
def toggle_rr_input_b(use_rr):
    return not use_rr


# ══════════════════════════════════════════════════════════════════════════════
# 6.  Apply / Clear filters  →  Selected filter + Photobleach filter
#     Both filters are configured by their toggles and applied together when
#     the user clicks "Apply Filter".  "Clear Filter" removes ALL NaN marks
#     (from both filters) while preserving fitted state values.
# ══════════════════════════════════════════════════════════════════════════════
@app.callback(
    Output('hmm-pb-status', 'children'),
    Input('hmm-pb-apply', 'n_clicks'),
    Input('hmm-pb-clear', 'n_clicks'),
    State('hmm-channel',          'value'),
    State('hmm-selected-filter',  'value'),
    State('hmm-pb-enable',        'value'),
    # fret_g thresholds
    State('hmm-pb-ming-totg', 'value'),
    State('hmm-pb-use-rr',    'value'),
    State('hmm-pb-ming-rr',   'value'),
    # fret_b thresholds
    State('hmm-pb-minb-totb', 'value'),
    State('hmm-pb-use-rr-b',  'value'),
    State('hmm-pb-minb-rr',   'value'),
    # single-channel threshold
    State('hmm-pb-min-single', 'value'),
    # path + smoothing from Tools tab
    State('path',          'value'),
    State('smooth',        'value'),
    State('smooth_method', 'value'),
    State('polyorder',     'value'),
)
def apply_pb_filter(
    n_apply, n_clear,
    channel, selected_filter, pb_on,
    min_totg, use_rr_g, min_rr_g,
    min_totb, use_rr_b, min_rr_b,
    min_single,
    path, smooth, smooth_method, polyorder,
):
    global hmm_mask, hmm_states, hmm_fret_g, N_traces, total_frame, select_list_g

    ctx = callback_context
    if not ctx.triggered:
        raise PreventUpdate
    triggered = ctx.triggered[0]['prop_id'].split('.')[0]

    # ── CLEAR — remove ALL NaN marks from both filters ───────────────────────
    if triggered == 'hmm-pb-clear':
        if N_traces == 0:
            return 'No data loaded.'
        if channel in hmm_states:
            arr = hmm_states[channel]
            for mol in range(arr.shape[0]):
                for f in range(arr.shape[1]):
                    v = arr[mol, f]
                    if v is not None:
                        try:
                            if np.isnan(float(v)):
                                arr[mol, f] = None
                        except (TypeError, ValueError):
                            pass
        # Reset mask to all-usable
        if hmm_mask is not None and hmm_mask.ndim == 2:
            hmm_mask = np.zeros(hmm_mask.shape, dtype=object)
        else:
            hmm_mask = np.zeros((N_traces, total_frame), dtype=object)
        return '✓ All filters cleared — NaN marks removed, predictions preserved.'

    if triggered != 'hmm-pb-apply':
        raise PreventUpdate

    if N_traces == 0 or total_frame == 0:
        return '✗ No data loaded yet.'
    if not pb_on and not selected_filter:
        return '✗ Both filters are OFF — enable at least one toggle first.'
    if pb_on and not path:
        return '✗ Photobleach filter ON but no folder path loaded.'

    # Determine array dimensions — load traces to get channel-specific shape
    try:
        w    = int(smooth)    if smooth    else 1
        poly = int(polyorder) if polyorder else 2
        hfit = HMM_fitter(path)
        hfit.load_traces(channel=channel)
    except Exception as e:
        return f'✗ Error loading traces: {str(e)}'

    N_ch, F_ch = hfit.Q.shape
    if channel not in hmm_states or hmm_states[channel].shape != (N_ch, F_ch):
        hmm_states[channel] = np.full((N_ch, F_ch), None, dtype=object)
    arr = hmm_states[channel]

    msgs = []

    # ── Selected filter — mark bad traces (select_g == -1) as fully NaN ─────
    if selected_filter:
        sel_marked = 0
        for mol in range(min(len(select_list_g), N_ch)):
            if int(select_list_g[mol]) == -1:
                arr[mol, :] = np.nan
                sel_marked += 1
        msgs.append(f'{sel_marked} bad traces marked')

    # ── Photobleach filter — mark PB frames as NaN ───────────────────────────
    if pb_on:
        thresholds = {
            'min_totg':  min_totg,  'use_rr':   use_rr_g,  'min_rr':   min_rr_g,
            'min_totb':  min_totb,  'use_rr_b': use_rr_b,  'min_rr_b': min_rr_b,
            'min_single': min_single,
        }
        try:
            mask = hfit.build_mask(channel, pb_on, thresholds,
                                   smooth_method or 'moving', w, poly)
        except Exception as e:
            return f'✗ PB filter error: {str(e)}'

        if mask is None:
            return '✗ build_mask returned None — check thresholds or data.'

        hmm_mask = mask
        N_mask, F_mask = mask.shape
        pb_count = 0
        for mol in range(N_mask):
            pb_start = None
            for f in range(F_mask):
                try:
                    if np.isnan(float(mask[mol, f])):
                        pb_start = f
                        break
                except (TypeError, ValueError):
                    pass
            if pb_start is not None:
                arr[mol, pb_start:] = np.nan
                pb_count += 1
            else:
                # No PB — clear any stale NaN in non-selected-filter frames
                for f in range(F_mask):
                    v = arr[mol, f]
                    if v is not None:
                        try:
                            if np.isnan(float(v)) and not (
                                selected_filter
                                and mol < len(select_list_g)
                                and int(select_list_g[mol]) == -1
                            ):
                                arr[mol, f] = None
                        except (TypeError, ValueError):
                            pass
        msgs.append(f'{pb_count} traces PB-filtered')
    else:
        hmm_mask = np.zeros((N_ch, F_ch), dtype=object)

    if channel == 'fret_g':
        hmm_fret_g = hmm_states['fret_g']

    return f'✓ Filter applied [{channel}]: ' + ', '.join(msgs) + '.'


# ══════════════════════════════════════════════════════════════════════════════
# 7.  HMM sidebar — callbacks live in utils/HMM_slidebar.py
# ══════════════════════════════════════════════════════════════════════════════

hmm_bar.register_callbacks(app, sys.modules[__name__])



@app.callback(
    Output('loadp',              'n_clicks'),
    Output('hmm-segments-store', 'data',   allow_duplicate=True),
    Output('hmm-state-bar',      'figure', allow_duplicate=True),
    Output('hmm_means',          'data',   allow_duplicate=True),
    Input('hmm_start', 'n_clicks'),
    State('smooth', 'value'),
    State('hmm_fit',      'value'),
    State('hmm_cov_type', 'value'),
    State('hmm_means',    'data'),
    State('hmm_epoch',    'value'),
    State('hmm_niter',    'value'),
    State('hmm-channel',  'value'),
    State('path',         'value'),
    State('smooth_method', 'value'),
    State('polyorder',     'value'),
    State('i',             'value'),
    State('hmm_reverse',   'value'),
    State('hmm_sticky_w',  'value'),
    blocking=True,
)
def HMM(start, w, fit, cov_type, means, epoch, n_iter, channel, path,
        smooth_method, polyorder, i, reverse, sticky_w):
    changed_id = [p['prop_id'] for p in callback_context.triggered][0]
    if 'hmm_start' not in changed_id:
        raise PreventUpdate()

    global hmm_mask

    # ── Parse initial means from the table ───────────────────────────────────
    # Values seed EM starting point; EM is free to refine them.
    # Number of valid entries determines k (number of states).
    fret_channels = {'fret_g', 'fret_b'}
    input_row = means[0] if means else {}
    init_means = []
    for idx in range(10):
        try:
            v = float(input_row[str(idx)])
        except (KeyError, TypeError, ValueError):
            continue
        if channel in fret_channels:
            if 0 <= v <= 1:
                init_means.append(v)
        else:       # gg, rr, bb — intensity values can exceed 1
            if v >= 0:
                init_means.append(v)

    if not init_means:
        raise PreventUpdate()
    init_means = np.array(init_means).reshape(-1, 1)
    print(init_means)

    poly = int(polyorder) if polyorder else 2
    hfit = HMM_fitter(path)
    hfit.load_traces(channel=channel)
    # Pass the live hmm_states[channel] so each trace's length equals
    # the number of non-NaN (non-PB) frames, capturing both the PB
    # filter and any manual boundary edits made on the slidebar.
    sticky = float(sticky_w) if sticky_w is not None else 1.0
    hfit.fitHMM(fit, w, init_means, epoch=epoch,
                covariance_type=cov_type, n_iter=n_iter,
                mask=hmm_mask,
                hd_states=hmm_states.get(channel),
                smooth_method=smooth_method or 'moving', polyorder=poly,
                sticky_w=sticky)
    hfit.cal_states()

    # ── Populate hmm_states[channel] from fitting result ─────────────────────
    N_fit    = hfit.Q.shape[0]
    F_fit    = hfit.Q.shape[1]
    old_arr  = hmm_states.get(channel)   # preserve NaN marks set by filters
    arr      = np.full((N_fit, F_fit), None, dtype=object)

    # Molecules that were actually fitted
    fitted_mols = {idx for idx, _ in getattr(hfit, 'pro_Q_indices', [])}

    for mol_idx, tr in enumerate(hfit.result_hd_states):
        if mol_idx in fitted_mols and tr is not None and len(tr) > 0:
            n = min(len(tr), F_fit)
            arr[mol_idx, :n] = np.asarray(tr[:n], dtype=object)
        elif old_arr is not None and mol_idx < old_arr.shape[0]:
            # Not fitted (e.g. fully NaN from Selected/PB filter, or unselected)
            # — restore the previous row so NaN marks are not lost.
            F_old = min(old_arr.shape[1], F_fit)
            arr[mol_idx, :F_old] = old_arr[mol_idx, :F_old]

    # Mark frames beyond each fitted trace's PB boundary as NaN
    for mol_idx, end_frame in getattr(hfit, 'pro_Q_indices', []):
        if end_frame < F_fit:
            arr[mol_idx, end_frame:] = np.nan

    hmm_means[channel]  = np.sort(hfit.mus.flatten())   # sorted: S0=lowest, S1=next, …
    hmm_states[channel] = arr                           # arr already holds int state indices
    if channel == 'fret_g':
        hmm_fret_g = arr

    # ── State reverse (optional) ─────────────────────────────────────────────
    # When the toggle is ON, flip the state ordering so that state 0 corresponds
    # to the HIGHEST mean and state (n-1) to the LOWEST.  The means array and
    # every integer index stored in arr are both remapped consistently.
    # This is purely a post-processing relabelling; NaN and None cells are kept.
    if reverse:
        n_states = len(hmm_means[channel])
        if n_states > 1:
            # Flip means: [low, …, high] → [high, …, low]
            hmm_means[channel] = hmm_means[channel][::-1].copy()

            # Remap state indices: old k → new (n_states-1-k).
            # Work from a snapshot (old_arr) so intermediate writes don't alias.
            old_arr = arr.copy()
            for s in range(n_states):
                arr[old_arr == s] = n_states - 1 - s

            hmm_states[channel] = arr
            if channel == 'fret_g':
                hmm_fret_g = arr

    # ── Build immediate slidebar preview for the current molecule ────────────
    mol       = int(i) if i is not None else 0
    t_arr     = hfit.time
    row       = arr[mol] if mol < N_fit else np.array([], dtype=object)
    segments  = trace_to_segments(np.asarray(row, dtype=object), t_arr)
    segments  = merge_same_state_segments(segments, row)
    view_start = float(t_arr[0])  if len(t_arr) else 0.0
    view_end   = float(t_arr[-1]) if len(t_arr) else 1.0
    bar_fig    = make_hmm_bar_figure(segments, view_start, view_end, -1)

    # ── Refresh means table with the fitted (and possibly reversed) values ───
    # Use hmm_means[channel] directly — already reversed if toggle was ON.
    fitted_vals = hmm_means[channel].tolist()
    new_row = {str(p): (round(float(fitted_vals[p]), 4) if p < len(fitted_vals) else -1)
               for p in range(10)}
    means_data = [new_row]

    return np.random.rand(1), segments, bar_fig, means_data



@app.callback(
    Output('gmm_hist','figure'),
    Output('gmm_means','data'),
    Input('gmm_fit','n_clicks'),
    Input('gmm_save','n_clicks'),
    Input('binsize','value'),
    Input('gmm_comps','value'),
    Input('gmm_cov_type', 'value'),
    State('gmm_means','data'),
    State('gmm_channel', 'value'),
    State('path','value'),
    )
def update_Hist(fit, save, binsize, n_comps, cov_type, means, channels, path):
    changed_id = [p['prop_id'] for p in callback_context.triggered][0]
    global fret_g, fret_b, select_list_g, fig2, gmm
   
        
    if 'gmm_comps' in changed_id:
        means = [dict(**{str(param): -1 for param in range(0, 10)})]


    if ('gmm_fit' in changed_id or 'gmm_comps' in changed_id or 'cov_type' in changed_id):
        init_means = []
        if channels == 'fret_g':
            FRET_list = fret_g
            select_list = select_list_g
        else:
            FRET_list = fret_b
            select_list = select_list_g

        for i in range (0, 10):
            try:
                float(means[0][str(i)])
            except:
                continue
            if 0 <= float(means[0][str(i)]) <=1:
                init_means.append(float(means[0][str(i)]))
        
        m, c, w, X, gmm = fit_gmm(FRET_list, select_list, init_means, cov_type, int(n_comps))
        fig2 = draw_gmm(fig2, m, c, w, X)
        
    if 'binsize' in changed_id:
        fig2.update_traces(xbins=dict(start=0,end=1,size = float(binsize)),selector=dict(name='gmm_hist'))

    if 'gmm_save' in changed_id:
        if gmm != None:
            save_gmm(gmm, path)

    

    return fig2, means

app.clientside_callback(
    """
    function(checked) {
        return checked ? "On: empty clears old bkps" : "Off: keep old bkps";
    }
    """,
    Output("vacuum-overwrite-hint", "children"),
    Input("switch-vacuum-overwrite", "value"),
)

@app.callback(
    Output('rup-status-output', 'children'),
    Input('btn-run-rupture', 'n_clicks'),
    Input('btn-rup-fit-all', 'n_clicks'),
    Input('btn-rup-clear', 'n_clicks'),
    Input('btn-rup-clear-all', 'n_clicks'),
    Input('btn-rup-save', 'n_clicks'),
    Input('rup-direction', 'value'),
    Input('rup-mingap', 'value'),
    State('switch-vacuum-overwrite', "value"),
    State('rup-channel', 'value'),
    State('smooth_method', 'value'),
    State('smooth', 'value'),
    State('rup-penalty', 'value'),
    State('rup-censor-mode', 'value'),
    State('rup-upper-thresh', 'value'),
    State('rup-lower-thresh', 'value'),
    State('i', 'value'),
    State('path', 'value')
)
def run_rpt_analysis(n_clicks_run, n_clicks_run_all, n_clicks_clear, n_clicks_clear_all,
                     n_clicks_save, rup_direction, rup_mingap,
                     vacuum_overwrite, rup_channel, smooth_method, smooth, penalty,
                     censor_mode, upper_thresh, lower_thresh, i, path):

    ctx = callback_context
    if not ctx.triggered:
        raise PreventUpdate

    triggered_id = ctx.triggered[0]['prop_id'].split('.')[0]

    global fret_g, fret_b, rr, gg, gr, bb, bg, br, time, rup_bkps, N_traces, bkps

    if not rup_bkps:
        rup_bkps = [[] for _ in range(max(1, N_traces))]

    if 'bkps' not in globals() or bkps is None:
        bkps = {}

    try: i = int(i)
    except: i = 0

    # -------------------------------------------------------------
    # CLEAR AND SAVE LOGIC
    # -------------------------------------------------------------
    if triggered_id == 'btn-rup-clear':
        if i < len(rup_bkps):
            rup_bkps[i] = []
        return f"Cleared predictions for Trace {i}"

    elif triggered_id == 'btn-rup-clear-all':
        rup_bkps = [[] for _ in range(max(1, N_traces))]
        return "Cleared predictions for ALL traces"

    elif triggered_id == 'btn-rup-save':
        if not path:
            return "Error: No folder path loaded."
        file_path = os.path.join(path, 'breakpoints.npz')

        if os.path.exists(file_path):
            try:
                loaded = np.load(file_path, allow_pickle=True)
                for key in loaded.files:
                    bkps[key] = loaded[key].tolist()
            except Exception as e:
                print(f"Failed to read existing npz: {e}")

        time_mapping = {
            'fret_g': 'fret_g', 'fret_b': 'fret_b',
            'gg': 'g', 'gr': 'g', 'rr': 'r',
            'bb': 'b', 'bg': 'b', 'br': 'b',
            'r': 'r', 'g': 'g', 'b': 'b'
        }
        time_key = time_mapping.get(rup_channel, 'fret_g')
        raw_time = time[time_key]

        # Ensure the channel list is sized to the current N_traces.
        # (The initial global bkps entries start as [] before any file is loaded;
        # after loading they are already [[] * N_traces].  We also resize here so
        # that a channel that was saved with a smaller dataset gets extended.)
        n_needed = max(1, N_traces)
        if rup_channel not in bkps or len(bkps[rup_channel]) < n_needed:
            existing = bkps.get(rup_channel, [])
            bkps[rup_channel] = list(existing) + [[] for _ in range(n_needed - len(existing))]

        # ← fixed indentation + vacuum overwrite logic
        for j in range(max(1, N_traces)):
            if j < len(rup_bkps):
                if len(rup_bkps[j]) > 0:
                    trace_list = []
                    for t in rup_bkps[j]:
                        idx = (np.abs(raw_time - t)).argmin()
                        trace_list.append((idx, t))
                    bkps[rup_channel][j] = trace_list
                else:
                    if vacuum_overwrite:
                        bkps[rup_channel][j] = []  # explicitly wipe
                    # else: skip → old data stays

        if os.path.exists(file_path):
            timestamp = rtime.strftime("%Y%m%d_%H%M%S")
            backup_path = os.path.join(path, f'breakpoints_{timestamp}.npz')
            shutil.copy(file_path, backup_path)

        save_dict = {k: np.array(v, dtype=object) for k, v in bkps.items()}
        np.savez(file_path, **save_dict)

        rup_bkps = [[] for _ in range(max(1, N_traces))]
        return f"SUCCESS: Saved to Tools Tab! Purple dots cleared."

    # -------------------------------------------------------------
    # FIT ALL TRACES
    # -------------------------------------------------------------
    elif triggered_id == 'btn-rup-fit-all':
        fitted_count = 0
        skipped_count = 0
        signal_dict = {
            'fret_g': fret_g, 'fret_b': fret_b,
            'rr': rr, 'gg': gg, 'gr': gr, 'bb': bb, 'bg': bg, 'br': br,
            'r': rr, 'g': gg, 'b': bb
        }
        time_mapping = {
            'fret_g': 'fret_g', 'fret_b': 'fret_b',
            'gg': 'g', 'gr': 'g', 'rr': 'r',
            'bb': 'b', 'bg': 'b', 'br': 'b',
            'r': 'r', 'g': 'g', 'b': 'b'
        }
        time_key = time_mapping.get(rup_channel, 'fret_g')
        raw_time = time[time_key]

        try: lag = int(smooth)
        except: lag = 1
        try: min_gap = int(rup_mingap)
        except: min_gap = 0

        for j in range(max(1, N_traces)):
            while len(rup_bkps) <= j:
                rup_bkps.append([])

            if len(rup_bkps[j]) > 0:
                skipped_count += 1
                continue

            if rup_channel not in signal_dict:
                continue

            channel_data = signal_dict[rup_channel]
            if channel_data is None or j >= len(channel_data):
                continue

            raw_signal = channel_data[j]
            if len(raw_signal) == 0:
                continue

            if len(raw_signal) != len(raw_time):
                min_len = min(len(raw_signal), len(raw_time))
                raw_signal = raw_signal[:min_len]
                local_time = raw_time[:min_len]
            else:
                local_time = raw_time

            if smooth_method == 'moving': sm_signal, sm_time = uf(raw_signal, lag), local_time
            elif smooth_method == 'median': sm_signal, sm_time = mf(raw_signal, lag), local_time
            elif smooth_method == 'savgol': sm_signal, sm_time = sg(raw_signal, lag, polyorder=2), local_time
            elif smooth_method == 'strided': sm_signal, sm_time = sa(raw_signal, lag), sa(local_time, lag)
            else: sm_signal, sm_time = raw_signal, local_time

            try:
                detector = Rupture(sm_signal)
                detector.config['pen'] = penalty
                time_bkps = detector.analyze_trace(
                    sm_time=sm_time, min_gap=min_gap, direction=rup_direction,
                    censor_mode=censor_mode, upper_thresh=upper_thresh, lower_thresh=lower_thresh
                )
                rup_bkps[j] = time_bkps
                fitted_count += 1
            except Exception as e:
                print(f"Rupture fit failed for trace {j}: {e}. Tell 🐎")
                continue

        return f"Fit All Complete! Predicted {fitted_count} new traces. Kept {skipped_count} current manual predictions."

    # -------------------------------------------------------------
    # DETECT CURRENT TRACE
    # -------------------------------------------------------------
    signal_dict = {
        'fret_g': fret_g, 'fret_b': fret_b,
        'rr': rr, 'gg': gg, 'gr': gr, 'bb': bb, 'bg': bg, 'br': br,
        'r': rr, 'g': gg, 'b': bb
    }
    time_mapping = {
        'fret_g': 'fret_g', 'fret_b': 'fret_b',
        'gg': 'g', 'gr': 'g', 'rr': 'r',
        'bb': 'b', 'bg': 'b', 'br': 'b',
        'r': 'r', 'g': 'g', 'b': 'b'
    }

    if rup_channel not in signal_dict:
        return f"Error: Channel '{rup_channel}' not supported."

    channel_data = signal_dict[rup_channel]

    if channel_data is None or len(channel_data) == 0:
        return f"Error: No data loaded for channel '{rup_channel}'."
    if i >= len(channel_data):
        return f"Error: Trace index {i} out of bounds (Max is {len(channel_data)-1})."

    raw_signal = channel_data[i]
    time_key = time_mapping.get(rup_channel, 'fret_g')
    raw_time = time[time_key]

    if len(raw_signal) == 0: return f"Error: The actual signal for '{rup_channel}' is empty."
    if len(raw_time) == 0: return f"Error: The time array for '{rup_channel}' is empty."

    if len(raw_signal) != len(raw_time):
        min_len = min(len(raw_signal), len(raw_time))
        raw_signal = raw_signal[:min_len]
        raw_time = raw_time[:min_len]

    try: lag = int(smooth)
    except: lag = 1
    try: min_gap = int(rup_mingap)
    except: min_gap = 0

    if smooth_method == 'moving': sm_signal, sm_time = uf(raw_signal, lag), raw_time
    elif smooth_method == 'median': sm_signal, sm_time = mf(raw_signal, lag), raw_time
    elif smooth_method == 'savgol': sm_signal, sm_time = sg(raw_signal, lag, polyorder=2), raw_time
    elif smooth_method == 'strided': sm_signal, sm_time = sa(raw_signal, lag), sa(raw_time, lag)
    else: sm_signal, sm_time = raw_signal, raw_time

    if len(sm_time) == 0 or len(sm_signal) == 0:
        return f"Error: Smoothing compressed the array to 0 points! Lower your lag."

    try:
        detector = Rupture(sm_signal)
        detector.config['pen'] = penalty
        time_bkps = detector.analyze_trace(
            sm_time=sm_time, min_gap=min_gap, direction=rup_direction,
            censor_mode=censor_mode, upper_thresh=upper_thresh, lower_thresh=lower_thresh
        )

        while len(rup_bkps) <= i:
            rup_bkps.append([])

        rup_bkps[i] = time_bkps
        return f"Success: {len(time_bkps)} points on {rup_channel}"

    except Exception as e:
        import traceback
        print(traceback.format_exc())
        return f"Error: {str(e)}"

# ══════════════════════════════════════════════════════════════════════════════
# Rupture — Fit-All progress bar
#
# The bar is shown via a *clientside* callback so it appears in the browser
# instantly when the button is clicked, before the server has even started
# processing.  A server-side callback hides it as soon as the result arrives
# (i.e. when rup-status-output gets any new value).
# ══════════════════════════════════════════════════════════════════════════════

app.clientside_callback(
    """
    function(n_clicks) {
        if (n_clicks) {
            return {display: 'block'};
        }
        return window.dash_clientside.no_update;
    }
    """,
    Output('rup-fit-progress-container', 'style'),
    Input('btn-rup-fit-all', 'n_clicks'),
    prevent_initial_call=True,
)

@app.callback(
    Output('rup-fit-progress-container', 'style', allow_duplicate=True),
    Input('rup-status-output', 'children'),
    prevent_initial_call=True,
)
def _hide_fit_all_progress(_status):
    """Hide the Fit-All progress bar once any rupture action finishes."""
    return {'display': 'none'}


# ══════════════════════════════════════════════════════════════════════════════
# Dwell Manager — boundary UI callbacks
# ══════════════════════════════════════════════════════════════════════════════

# Left boundary: enable/disable the correct input when radio switches mode
@app.callback(
    Output('dwell-left-cut',      'disabled'),
    Output('dwell-left-cut',      'value'),
    Output('dwell-left-deadtime', 'disabled'),
    Output('dwell-left-deadtime', 'value'),
    Input('dwell-left-mode', 'value'),
    prevent_initial_call=True,
)
def toggle_left_boundary_mode(mode):
    if mode == 'cut':
        # Cut states active — enable cut input, disable & clear dead time
        return False, 0, True, 0
    else:
        # Dead time active — disable & clear cut input, enable dead time
        return True, 0, False, 0



# ══════════════════════════════════════════════════════════════════════════════
# HMM Clear — wipe predicted state values (keep NaN/PB marks) for all molecules
# ══════════════════════════════════════════════════════════════════════════════
@app.callback(
    Output('hmm-segments-store',     'data',    allow_duplicate=True),
    Output('hmm-state-bar',          'figure',  allow_duplicate=True),
    Output('hmm-fitted-states-table','columns', allow_duplicate=True),
    Output('hmm-fitted-states-table','data',    allow_duplicate=True),
    Input('hmm_clear', 'n_clicks'),
    State('hmm-channel',  'value'),
    State('i',            'value'),
    State('graph',        'relayoutData'),
    prevent_initial_call=True,
)
def hmm_clear_callback(n_clicks, channel, i, relayout_data):
    global hmm_states, hmm_means
    if not n_clicks:
        raise PreventUpdate
    if channel not in hmm_states:
        raise PreventUpdate

    # ── UNDO: snapshot the whole channel before wiping ────────────────────────
    undo.push_hmm_channel(hmm_states, hmm_means, channel)

    # Clear the fitted means for this channel so the table empties
    hmm_means.pop(channel, None)

    arr = hmm_states[channel]
    # Set every non-NaN value to None (clear predictions, keep PB markers)
    for mol in range(arr.shape[0]):
        for f in range(arr.shape[1]):
            v = arr[mol, f]
            if v is not None:
                try:
                    if not np.isnan(float(v)):
                        arr[mol, f] = None
                except (TypeError, ValueError):
                    arr[mol, f] = None

    # Rebuild bar for current molecule
    mol      = int(i) if i is not None else 0
    time_key = hmm_bar.CHANNEL_TIME_KEY.get(channel, 'fret_g')
    t_arr    = time.get(time_key, [])
    if len(t_arr) == 0:
        return [], hmm_bar.EMPTY_BAR_FIGURE, [], []

    row      = np.asarray(arr[mol], dtype=object) if mol < len(arr) else np.array([], dtype=object)
    segments = trace_to_segments(row, t_arr)
    segments = merge_same_state_segments(segments, row)

    view_start = float(t_arr[0])
    view_end   = float(t_arr[-1])
    if relayout_data:
        if 'xaxis.range[0]' in relayout_data and 'xaxis.range[1]' in relayout_data:
            view_start = float(relayout_data['xaxis.range[0]'])
            view_end   = float(relayout_data['xaxis.range[1]'])

    bar_fig = make_hmm_bar_figure(segments, view_start, view_end, -1)
    return segments, bar_fig, [], []


# ══════════════════════════════════════════════════════════════════════════════
# Merge States — collapse states whose fitted means are within threshold
# ══════════════════════════════════════════════════════════════════════════════
@app.callback(
    Output('hmm_means',           'data',   allow_duplicate=True),
    Output('hmm-state-bar',       'figure', allow_duplicate=True),
    Output('hmm-segments-store',  'data',   allow_duplicate=True),
    Output('hmm-merge-status',    'children'),
    Input('hmm-merge-btn', 'n_clicks'),
    State('hmm-merge-threshold', 'value'),
    State('hmm-channel',         'value'),
    State('i',                   'value'),
    State('graph',               'relayoutData'),
    prevent_initial_call=True,
)
def hmm_merge_callback(n_clicks, threshold, channel, i, relayout_data):
    global hmm_states, hmm_means, hmm_fret_g
    if not n_clicks:
        raise PreventUpdate
    if channel not in hmm_means or channel not in hmm_states:
        return no_update, no_update, no_update, '✗ No HMM prediction — run fitting first.'

    # ── UNDO: snapshot the whole channel before merging ───────────────────────
    undo.push_hmm_channel(hmm_states, hmm_means, channel)

    mus = hmm_means[channel]          # sorted 1-D numpy array (low → high)
    k   = len(mus)
    thr = float(threshold) if threshold is not None else 0.05

    # ── Group consecutive means within threshold ──────────────────────────────
    # Reference = first (lowest) element of the current group — avoids chaining.
    groups        = []
    current_group = [0]
    for idx in range(1, k):
        if abs(mus[idx] - mus[current_group[0]]) <= thr:
            current_group.append(idx)
        else:
            groups.append(current_group)
            current_group = [idx]
    groups.append(current_group)

    n_after = len(groups)
    if n_after == k:
        return (no_update, no_update, no_update,
                f'No states merged (threshold={thr}). All {k} states are well-separated.')

    # ── Build old→new index mapping and new means ─────────────────────────────
    mapping   = np.full(k, -1, dtype=int)
    new_means = []
    for new_idx, grp in enumerate(groups):
        new_means.append(float(np.mean(mus[grp])))
        for old_idx in grp:
            mapping[old_idx] = new_idx

    # ── Remap hmm_states[channel] (2-D object array) ──────────────────────────
    # Read from original arr, write to new_arr — prevents aliasing when
    # new_idx < old_idx.  Iterate over old state indices (≤10), not pixels.
    arr     = hmm_states[channel]
    new_arr = arr.copy()
    for old_idx in range(k):
        new_idx = int(mapping[old_idx])
        if old_idx != new_idx:
            new_arr[arr == old_idx] = new_idx
    hmm_states[channel] = new_arr
    if channel == 'fret_g':
        hmm_fret_g = new_arr   # keep global mirror in sync

    # ── Update global means ───────────────────────────────────────────────────
    hmm_means[channel] = np.array(new_means, dtype=float)

    # ── Rebuild segments + bar for the current molecule ───────────────────────
    mol      = int(i) if i is not None else 0
    time_key = hmm_bar.CHANNEL_TIME_KEY.get(channel, 'fret_g')
    t_arr    = time.get(time_key, [])
    if len(t_arr) == 0:
        bar_fig  = no_update
        segments = no_update
    else:
        row      = np.asarray(new_arr[mol], dtype=object) if mol < len(new_arr) else np.array([], dtype=object)
        segments = trace_to_segments(row, t_arr)
        segments = merge_same_state_segments(segments, row)
        view_start = float(t_arr[0])
        view_end   = float(t_arr[-1])
        if relayout_data:
            if 'xaxis.range[0]' in relayout_data and 'xaxis.range[1]' in relayout_data:
                view_start = float(relayout_data['xaxis.range[0]'])
                view_end   = float(relayout_data['xaxis.range[1]'])
        bar_fig = make_hmm_bar_figure(segments, view_start, view_end, -1)

    # ── Refresh hmm_means table (triggers update_fitted_states_table) ─────────
    new_row = {str(p): (round(new_means[p], 4) if p < len(new_means) else -1)
               for p in range(10)}
    status  = f'✓ Merged: {k} → {n_after} states  (threshold={thr})'
    print(f'[merge] {status}')
    return [new_row], bar_fig, segments, status


# ══════════════════════════════════════════════════════════════════════════════
# Save hmm.npz
# ══════════════════════════════════════════════════════════════════════════════
@app.callback(
    Output('hmm-save-status', 'children'),
    Input('hmm-save-npz', 'n_clicks'),
    State('path', 'value'),
    prevent_initial_call=True,
)
def hmm_save_npz(n_clicks, path):
    if not n_clicks:
        raise PreventUpdate
    if not path:
        return '✗ No folder path loaded.'

    global hmm_mask
    # Generate timestamp once so hmm.npz backup and transition.txt backup
    # both carry the identical suffix (e.g. hmm_20260518_143200.npz +
    # transition_20260518_143200.txt).
    ts = rtime.strftime('%Y%m%d_%H%M%S')
    saved, ts_used = HMM_fitter(path).save_hmm(
        hmm_states_dict=hmm_states,
        hmm_means_dict=hmm_means,
        hmm_mask=hmm_mask,
        timestamp=ts,
    )

    if not saved:
        return '✗ No fitted HMM data found. Run HMM fitting first.'

    # Save companion transition.txt with the same backup timestamp
    t_suffix = ''
    try:
        t_written = save_transition_txt(path, hmm_states, hmm_means, ts_used)
        t_suffix = ' + transition.txt' if t_written else ''
    except Exception as e:
        print(f'transition.txt save error: {e}')
        t_suffix = ' (transition.txt failed)'

    return f'✓ Saved [{", ".join(saved)}] → hmm.npz{t_suffix}'


# ══════════════════════════════════════════════════════════════════════════════
# Save dwell.npz
# ══════════════════════════════════════════════════════════════════════════════
@app.callback(
    Output('hmm-save-status', 'children', allow_duplicate=True),
    Input('hmm-save-dwell',    'n_clicks'),
    State('path',              'value'),
    State('hmm-channel',       'value'),
    State('dwell-left-mode',     'value'),
    State('dwell-left-cut',      'value'),
    State('dwell-left-deadtime', 'value'),
    State('dwell-right-cut',     'value'),
    prevent_initial_call=True,
)
def hmm_save_dwell(n_clicks, path, channel, left_mode, left_cut, dead_time, right_cut):
    if not n_clicks:
        raise PreventUpdate
    if not path:
        return '✗ No folder path loaded.'
    if not channel or channel not in hmm_states:
        return '✗ No HMM prediction found — run HMM fitting first.'

    t_key = hmm_bar.CHANNEL_TIME_KEY.get(channel, 'fret_g')
    t_arr = np.asarray(time.get(t_key, []))
    if len(t_arr) == 0:
        return '✗ Time array not found.'

    left_cut_n  = int(left_cut)    if left_cut  is not None else 0
    right_cut_n = int(right_cut)   if right_cut is not None else 0
    dead_time_s = float(dead_time) if dead_time is not None else 0.0
    mode        = left_mode if left_mode in ('cut', 'dead') else 'cut'

    dwell_dict, events_dict = _extract_dwells(
        hmm_states[channel], t_arr,
        left_mode=mode,
        left_cut=left_cut_n,
        dead_time_s=dead_time_s,
        right_cut=right_cut_n,
    )

    if not dwell_dict:
        return '✗ No dwell times found after applying cuts.'

    dm = DwellManager(load_path=path)   # load existing dwell.npz to preserve other channels
    dm.data[channel]   = dwell_dict
    dm.events[channel] = events_dict
    dm.save(path)

    total      = sum(len(v) for v in dwell_dict.values())
    states_str = ', '.join(f'S{s}={len(v)}' for s, v in sorted(dwell_dict.items()))
    return f'✓ Saved dwell.npz — {total} dwells  [{states_str}]'


@app.callback(
    Output('hmm-save-status', 'children', allow_duplicate=True),
    Output('hmm_means',       'data',     allow_duplicate=True),
    Input('hmm-load-npz', 'n_clicks'),
    State('path',         'value'),
    State('hmm-channel',  'value'),
    prevent_initial_call=True,
)
def hmm_load_callback(n_clicks, path, channel):
    """Load hmm.npz → restore hmm_states global.

    The bar and colour windows update automatically because render_hmm_bar
    listens to hmm-save-status, which this callback outputs to.
    No need to manually rebuild segments or figure here.
    """
    if not path:
        return '✗ No folder path loaded.', no_update

    global hmm_fret_g, hmm_states, hmm_means

    data = HMM_fitter(path).load_hmm()
    if data is None:
        return '✗ hmm.npz not found — save first.', no_update

    # ── Restore globals ───────────────────────────────────────────────────────
    hmm_states = {}
    for ch, d in data.items():
        raw_states = d['hd_states']
        m = d.get('means')
        means_arr = (np.asarray(m, dtype=np.float64).flatten()
                     if (m is not None and len(m)) else np.array([]))

        if raw_states.dtype.kind in ('i', 'u'):
            # ── New format: int16 indices → reconstruct object array ─────────
            # -1 = no prediction (None), -2 = photobleached (NaN), 0..k-1 = state
            obj_arr = np.full(raw_states.shape, None, dtype=object)
            pb_mask = (raw_states == -2)
            if pb_mask.any():
                obj_arr[pb_mask] = np.nan
            for si in range(len(means_arr)):
                state_mask = (raw_states == si)
                if state_mask.any():
                    obj_arr[state_mask] = si   # store int index, not float mean
            hmm_states[ch] = obj_arr
        else:
            # ── Old format: float object array — convert to int indices ───────
            obj_arr = np.full(raw_states.shape, None, dtype=object)
            for idx in np.ndindex(raw_states.shape):
                v = raw_states[idx]
                if v is None:
                    pass   # stays None
                else:
                    try:
                        fv = float(v)
                        if np.isnan(fv):
                            obj_arr[idx] = np.nan
                        elif len(means_arr) > 0:
                            obj_arr[idx] = int(np.argmin(np.abs(means_arr - fv)))
                    except (TypeError, ValueError):
                        pass
            hmm_states[ch] = obj_arr

    if 'fret_g' in hmm_states:
        hmm_fret_g = hmm_states['fret_g']

    for ch, d in data.items():
        m = d.get('means')
        if m is not None and len(m):
            hmm_means[ch] = np.sort(np.asarray(m, dtype=float).flatten())

    ch_str = ', '.join(data.keys())
    # Writing to hmm-save-status triggers render_hmm_bar and update_fitted_states_table.
    # hmm_means store (Initial Means table) is intentionally NOT updated on load —
    # the user's manually entered seed values should be preserved.
    return f'✓ Loaded [{ch_str}] from hmm.npz', no_update


server = app.server 
if __name__ == '__main__':
   app.run_server(debug = True)
