from .smoothing import uf, sa, mf, sg
import numpy as np
import plotly.graph_objects as go

def update_trace(fig, relayout, i, scatter, fret_g, fret_b, rr, gg, gr, bb, bg, br, time, bkps, lag, smooth_mode, show, polyorder=2, rup_bkps=None, channel='r', active_tab='Aois'):
    """
    Update the plot traces in the figure for the given trace index 'i' using the specified smoothing.

    Parameters:
        fig: Plotly figure to be updated.
        relayout: Relayout data from graph interactions (e.g., zoom).
        i (int): Current trace index.
        scatter (int): 0 for 'lines', 1 for 'markers'.
        fret_g, fret_b, rr, gg, gr, bb, bg, br: Arrays of signal data.
        time (dict): Dictionary containing time arrays for channels.
        hmm_fret_g: HMM trace data.
        bkps (dict): Dictionary of breakpoints for each channel.
        lag (int): Smoothing window size.
        smooth_mode (str): 'moving', 'strided', 'median', or 'savgol'.
        show (list): List of trace names to hide.
        polyorder (int): Polynomial order for Savitzky-Golay filter.
    
    Returns:
        fig: The updated Plotly figure.
    """
    # ---------------------------
    # 1. Map the smoothing function based on user selection
    # ---------------------------
    if smooth_mode == 'moving':
        sm = lambda t, l: uf(t, l)
    elif smooth_mode == 'strided':
        sm = lambda t, l: sa(t, l)
    elif smooth_mode == 'median':
        sm = lambda t, l: mf(t, l)
    elif smooth_mode == 'savgol':
        sm = lambda t, l: sg(t, l, polyorder=polyorder)
    else:
        sm = lambda t, l: uf(t, l)

    # For breakpoints: strided downsamples the array, so we must divide the index by lag.
    # For moving, median, and savgol, the array length stays the same.
    idx_factor = lag if smooth_mode == 'strided' else 1

    # Define mode mapping: 0 -> lines; 1 -> markers.
    mode_dict = {0: 'lines', 1: 'markers'}

    # Try to extract the histogram range from the relayout data.
    try:
        hist_range = (relayout['xaxis.range[0]'], relayout['xaxis.range[1]'])
    except:
        hist_range = (0, np.inf)

    # ---------------------------
    # Update Blue Channel (fret_b) and associated signals.
    # ---------------------------
    if np.any(fret_b):
        # Pre-calculate smoothed arrays to improve performance
        sm_time_b = sm(time['b'], lag) if smooth_mode == 'strided' else time['b']
        sm_fret_b = sm(fret_b[i], lag)
        sm_bb = sm(bb[i], lag)
        sm_bg = sm(bg[i], lag)
        sm_br = sm(br[i], lag)
        sm_tot_b = sm(bb[i] + bg[i] + br[i], lag)

        fig.update_traces(x=sm_time_b, y=sm_fret_b, mode=mode_dict[scatter], visible=('FRET BG' not in show), selector=dict(name='fret_b'))
        fig.update_traces(x=sm_time_b, y=sm_bb, mode=mode_dict[scatter], visible=('BB' not in show), selector=dict(name='bb'))
        fig.update_traces(x=sm_time_b, y=sm_bg, mode=mode_dict[scatter], visible=('BG' not in show), selector=dict(name='bg'))
        fig.update_traces(x=sm_time_b, y=sm_br, mode=mode_dict[scatter], visible=('BR' not in show), selector=dict(name='br'))
        fig.update_traces(x=sm_time_b, y=sm_tot_b, mode=mode_dict[scatter], line=dict(dash="longdash", width=2), visible=('Tot B' not in show), selector=dict(name='tot_b'))
        
        # Breakpoints mapping
        b_bkps_idx = [(x[0] // idx_factor) for x in bkps['b'][i]]
        fig.update_traces(x=sm_time_b[b_bkps_idx], y=sm_bb[b_bkps_idx], mode='markers', selector=dict(name='b_bkps'))
        
        fret_b_bkps_idx = [(x[0] // idx_factor) for x in bkps['fret_b'][i]]
        fig.update_traces(x=sm_time_b[fret_b_bkps_idx], y=sm_fret_b[fret_b_bkps_idx], mode='markers', selector=dict(name='fret_b_bkps'))
        
        # Histogram mapping
        hfilt_b = (sm_time_b > hist_range[0]) * (sm_time_b < hist_range[1])
        fig.update_traces(y=sm_fret_b[hfilt_b], selector=dict(name='Histogram_b'))
        
        if 'Tot B' in show:
            fig.update_layout(yaxis4=dict(range=(0, np.max(np.concatenate((sm_bb, sm_bg, sm_br))))))
        else:
            fig.update_layout(yaxis4=dict(range=(0, np.max(sm_bb + sm_bg + sm_br))))
    else:
        selectors = ['fret_b', 'bb', 'bg', 'br', 'tot_b', 'b_bkps', 'fret_b_bkps']
        clear_trace(fig, selectors)

    # ---------------------------
    # Update Green Channel (fret_g) and associated signals.
    # ---------------------------
    if np.any(fret_g):
        sm_time_g = sm(time['g'], lag) if smooth_mode == 'strided' else time['g']
        sm_fret_g = sm(fret_g[i], lag)
        sm_gg = sm(gg[i], lag)
        sm_gr = sm(gr[i], lag)
        sm_tot_g = sm(gg[i] + gr[i], lag)

        fig.update_traces(x=sm_time_g, y=sm_fret_g, mode=mode_dict[scatter], visible=('FRET GR' not in show), selector=dict(name='fret_g'))
        fig.update_traces(x=sm_time_g, y=sm_gg, mode=mode_dict[scatter], visible=('GG' not in show), selector=dict(name='gg'))
        fig.update_traces(x=sm_time_g, y=sm_gr, mode=mode_dict[scatter], visible=('GR' not in show), selector=dict(name='gr'))
        fig.update_traces(x=sm_time_g, y=sm_tot_g, mode=mode_dict[scatter], line=dict(dash="longdash", width=2), visible=('Tot G' not in show), selector=dict(name='tot_g'))
        
        g_bkps_idx = [(x[0] // idx_factor) for x in bkps['g'][i]]
        fig.update_traces(x=sm_time_g[g_bkps_idx], y=sm_gg[g_bkps_idx], mode='markers', selector=dict(name='g_bkps'))
        
        fret_g_bkps_idx = [(x[0] // idx_factor) for x in bkps['fret_g'][i]]
        fig.update_traces(x=sm_time_g[fret_g_bkps_idx], y=sm_fret_g[fret_g_bkps_idx], mode='markers', selector=dict(name='fret_g_bkps'))
        
        hfilt_g = (sm_time_g > hist_range[0]) * (sm_time_g < hist_range[1])
        fig.update_traces(y=sm_fret_g[hfilt_g], selector=dict(name='Histogram_g'))
        
        if 'Tot G' in show:
            if 'RR' in show:
                fig.update_layout(yaxis3=dict(range=(0, np.max(np.concatenate((gg[i], gr[i]))))))
            else:
                fig.update_layout(yaxis3=dict(range=(0, np.max(np.concatenate((gg[i], gr[i], rr[i]))))))
        else:
            if 'RR' in show:
                fig.update_layout(yaxis3=dict(range=(0, np.max(gg[i] + gr[i]))))
            else:
                fig.update_layout(yaxis3=dict(range=(0, np.max(np.concatenate((gg[i] + gr[i], rr[i]))))))
    else:
        selectors = ['fret_g', 'gg', 'gr', 'tot_g', 'g_bkps', 'fret_g_bkps']
        clear_trace(fig, selectors)

    # ---------------------------
    # Update Red Channel (rr) signals.
    # ---------------------------
    if np.any(rr):
        sm_time_r = sm(time['r'], lag) if smooth_mode == 'strided' else time['r']
        sm_rr = sm(rr[i], lag)

        fig.update_traces(x=sm_time_r, y=sm_rr, mode=mode_dict[scatter], visible=('RR' not in show), selector=dict(name='rr'))
        
        r_bkps_idx = [(x[0] // idx_factor) for x in bkps['r'][i]]
        fig.update_traces(x=sm_time_r[r_bkps_idx], y=sm_rr[r_bkps_idx], mode='markers', selector=dict(name='r_bkps'))
        
        if not np.any(fret_g):
            fig.update_layout(yaxis3=dict(range=(0, np.max(rr[i]))))
    else:
        selectors = ['rr', 'r_bkps']
        clear_trace(fig, selectors)

    # ---------------------------
    # Update x-axis range based on time data.
    # ---------------------------
    if np.any(fret_b) or np.any(fret_g) or np.any(rr):
        fig.update_layout(xaxis1=dict(
            range=(min(time['g'][0], time['b'][0], time['r'][0]),
                   max(time['g'][-1], time['b'][-1], time['r'][-1]))
        ))

    # ---------------------------------------------------------
    # --- DRAW RUPTURE PREVIEW DOTS ---
    # Put this near the end of your update_trace function
    # ---------------------------------------------------------
    # 1. Clear any old preview traces so they don't infinitely stack
    fig.data = tuple(t for t in fig.data if t.name != 'Rpt Preview')

    # 2. Safely cast index
    current_i = int(i)

    # 2. ONLY draw the dots if the user is actively on the Rupture tab!
    # (Check exactly what the 'value' of your Rupture tab is in layout.py)
    if active_tab == 'Rupture':  
        if rup_bkps is not None and current_i < len(rup_bkps):
            current_times = rup_bkps[current_i]
            
            if len(current_times) > 0:                

                signal_map = {'fret_g': fret_g, 'fret_b': fret_b, 
                'rr': rr, 'gg': gg, 'gr': gr, 'bb': bb, 'bg': bg, 'br': br,
                'r': rr, 'g': gg, 'b': bb}
                time_mapping = {'fret_g': 'fret_g', 'fret_b': 'fret_b',
                    'gg': 'g', 'gr': 'g', 'rr': 'r', 
                    'bb': 'b', 'bg': 'b', 'br': 'b',
                    'r': 'r', 'g': 'g', 'b': 'b'}
                
                # THIS is the magic dictionary that moves the dots to the right panel!
                yaxis_map = {
                    'fret_g': 'y1', 'fret_b': 'y2', 
                    'gg': 'y3', 'gr': 'y3', 'rr': 'y3', 
                    'bb': 'y4', 'bg': 'y4', 'br': 'y4',
                    'r': 'y3', 'g': 'y3', 'b': 'y4'
                }
                
                if channel in signal_map:
                    y_arr = signal_map[channel][current_i]
                    t_key = time_mapping.get(channel, 'fret_g')
                    t_arr = time[t_key]
                    y_ax = yaxis_map.get(channel, 'y1')

                    # 4. Apply smoothing to align with visual line
                    if smooth_mode == 'moving': sm_time, sm_sig = t_arr, uf(y_arr, lag)
                    elif smooth_mode == 'median': sm_time, sm_sig = t_arr, mf(y_arr, lag)
                    elif smooth_mode == 'savgol': sm_time, sm_sig = t_arr, sg(y_arr, lag, polyorder)
                    elif smooth_mode == 'strided': sm_time, sm_sig = sa(t_arr, lag), sa(y_arr, lag)
                    else: sm_time, sm_sig = t_arr, y_arr

                    # 5. Map the exact coordinates
                    x_points, y_points = [], []
                    for t_val in current_times:
                        idx = np.argmin(np.abs(sm_time - t_val))
                        x_points.append(sm_time[idx])
                        y_points.append(sm_sig[idx])

                    # 6. Draw it using Scattergl!
                    fig.add_trace(go.Scattergl(
                        x=x_points, 
                        y=y_points,
                        mode='markers',
                        marker=dict(color='purple', size=14, symbol='circle-open', line=dict(width=3)),
                        name='Rpt Preview',
                        yaxis=y_ax,
                        showlegend=False
                    ))

    
    return fig










def clear_trace(fig, selectors):
    """
    Clear the data of traces specified by selectors.
    """
    for s in selectors:
        fig.update_traces(x=[], y=[], selector=dict(name=s))

def change_trace(changed_id, event, i, N_traces, fig):
    """
    Change the current trace index based on navigation events.

    Parameters:
        changed_id: Identifier of the triggering component.
        event: Event information (e.g., key press).
        i (int): Current trace index.
        N_traces (int): Total number of traces.
        fig: The Plotly figure.
    
    Returns:
        Tuple (new trace index, updated figure).
    """
    if ('next' in changed_id) or (('n_events' in changed_id) and event['key'] == 'w'):
        if i < N_traces - 1:
            i += 1
            fig.layout.shapes = []
    elif ('previous' in changed_id) or (('n_events' in changed_id) and event['key'] == 'q'):
        if i > 0:
            i -= 1
            fig.layout.shapes = []
    elif 'tr_go' in changed_id:
        try:
            i = int(i) if int(i) < N_traces else 0
        except:
            i = 0
        fig.layout.shapes = []

    fig.update_layout(annotations=[dict(
        text = f'Trace {i}',
       xref="paper", yref="paper",
        x=0.5, y=1.02,  
        xanchor='center', yanchor='top',
        showarrow=False,
        font=dict(size=20, family='Arial')
    )])

    return i, fig
