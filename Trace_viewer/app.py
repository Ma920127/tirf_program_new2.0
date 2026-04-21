import numpy as np
import os
import shutil
import time as rtime
from dash import Dash, dcc, html, Input, Output, State, callback_context, no_update, DiskcacheManager, callback
from dash.exceptions import PreventUpdate
from layout import make_app
from utils.trace import update_trace, change_trace
from utils.selection import select_good_bad, select_colocalized, render_good_bad, render_colocalized
from utils.breakpoints import breakpoints_utils, sl_bkps, find_chp
from utils.blob import show_blob
from Gaussian_mixture.gmm import fit_gmm, draw_gmm, save_gmm
from utils.plotting import plot_fret_trace
from utils.draw import draw
from utils.calculate_dtime import calculate_FRET, calculate_conv, gaussian
from init_fig import init_fig
from loader import Loader 
from Hidden_Markov.hmm_fitter_new import HMM_fitter
from Rupture import Rupture
from utils.smoothing import uf, sa, mf, sg
import plotly.graph_objects as go
import traceback

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
hmm_fret_g = np.zeros(1000)
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
    Input('rupture','n_clicks'),
    Input('set_good','n_clicks'),
    Input('set_bad','n_clicks'),
    Input('set_colocalized','n_clicks'),
    Input('select','n_clicks'),
    Input('scatter', 'value'),
    Input('smooth', 'value'),
    Input('smooth_method', 'value'), # Changed from 'strided'
    Input('polyorder', 'value'),     # Added this input
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

def update_fig(key_events, show, next, previous, go, dtime, etime, clickData, mode, save, load, loadp, rupture, good, bad, coloc, select, scatter, smooth, smooth_method, polyorder, rescale, relayout, channel, chp_find_0, chp_find_1, confirm_reset, rup_status, active_tab, i, path, 
               chp_mode_0, chp_comp_0, chp_thres_0, chp_channel_0, chp_target_0, chp_mode_1, chp_comp_1, chp_thres_1, chp_channel_1, chp_target_1, event,rup_channel):
    changed_id = [p['prop_id'] for p in callback_context.triggered][0]
    global N_traces, fig, fig2, idx, total_frame, color, good_style, bad_style, bmode
    global new, time_b, N_traces, total_frame, tot_dtime
    global fret_g, fret_b, rr, gg, gr, bb, bg, br, time, tot_g, tot_b, hmm_fret_g
    global select_list_g, colocalized_list, bkps, ch_label, blobs
    if ('n_events' in changed_id) and (event['key'] not in ['q', 'w', 'z', 'x', 'c']):
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

    ##rescale##
    if 'rescale' in changed_id:
        relayout = {'autosize' : True }
        fig['layout']['uirevision'] = False

    ##change trace##
    i, fig = change_trace(changed_id, event, i, N_traces, fig)
    
    ##select good bad##
    select_list_g = select_good_bad(changed_id, event, i, select_list_g)
    colocalized_list = select_colocalized(changed_id, event, i, colocalized_list)

    good_style, bad_style = render_good_bad(i, select_list_g)
    colocalized_style = render_colocalized(i, colocalized_list)

    #save select##        
    if 'select' in changed_id:  
        np.save(path+r'/selected_g.npy', select_list_g)
        np.save(path+r'/colocalized_list.npy', colocalized_list)
     
    ##breakpoints## (Added polyorder)
    bkps, mode, confirm_reset_show, channel_error_show = breakpoints_utils(changed_id, clickData, mode, channel, i, time, bkps, smooth, smooth_mode, polyorder)
    
    ##save / load breakpoints##
    bkps = sl_bkps(changed_id, path, bkps, mode)

    # Auto-Detect Breakpoints (Added polyorder)
    bkps, channel_error_show = find_chp(changed_id, fret_g, fret_b, rr, gg, gr, bb, bg, br, i, time, select_list_g, 
             chp_mode_0, chp_comp_0, chp_thres_0, chp_channel_0, chp_target_0, chp_mode_1, chp_comp_1, chp_thres_1, chp_channel_1, chp_target_1,
             bkps, smooth, smooth_mode, polyorder)

    # ---------------------------------------------------------
    # UI TRICK: HIDE RED DOTS ON RUPTURE TAB
    # ---------------------------------------------------------
    # Replace 'tab-rupture' with whatever the actual "value" of your Rupture tab is in layout.py
    if active_tab == 'Rupture': 
        # Create a "fake" dictionary so trace.py doesn't crash looking for 'g'
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
    fig = update_trace(fig, relayout, i, scatter, fret_g, fret_b, rr, gg, gr, bb, bg, br, time, hmm_fret_g, display_bkps, smooth, smooth_mode, show, polyorder=polyorder,
                       rup_bkps = rup_bkps, 
                       channel = rup_channel,
                       active_tab = active_tab)

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

    return fig, i, str_g_bkps, str_b_bkps, mode, nnote, good_style, bad_style, colocalized_style, ch_label, ch_label, ch_label, confirm_reset_show, channel_error_show, relayout



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




@app.callback(
    Output('loadp', 'n_clicks'),
    Input('hmm_start', 'n_clicks'),
    State('smooth', 'value'),
    State('hmm_fit','value'),
    State('hmm_fix','value'),
    State('hmm_plot','value'),
    State('hmm_cov_type', 'value'),
    State('hmm_means', 'data'),
    State('hmm_epoch', 'value'),
    State('hmm_niter', 'value'),
    State('path','value'),
    blocking = True  
    )
def HMM(start, w, fit, fix, plot, cov_type, means, epoch, n_iter, path):
    changed_id = [p['prop_id'] for p in callback_context.triggered][0]
    if 'hmm_start' in changed_id:
        init_means = []
        for i in range (0, 10):
            try:
                float(means[0][str(i)])
            except:
                continue
            if 0 <= float(means[0][str(i)]) <=1:
                init_means.append(float(means[0][str(i)]))
        init_means = np.array(init_means).reshape(-1, 1)
        print(init_means)
        if np.any(init_means):
            hfit = HMM_fitter(path)
            hfit.load_traces()
            hfit.fitHMM(fit, w, init_means, fix_means = fix, epoch = epoch, covariance_type = cov_type, n_iter = n_iter)
            hfit.cal_states(plot = plot)    
        return np.random.rand(1)
    else:
        raise PreventUpdate() 



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

        if rup_channel not in bkps:
            bkps[rup_channel] = [[] for _ in range(max(1, N_traces))]

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
            except:
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

server = app.server 
if __name__ == '__main__':
   app.run_server(debug = True)
