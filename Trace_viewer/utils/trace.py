from .smoothing import uf, sa
import numpy as np

def update_trace(fig, relayout, i, scatter, fret_g, fret_b, rr, gg, gr, bb, bg, br, time, hmm_fret_g, bkps, lag, smooth_mode, show):
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
        smooth_mode (str): Either 'moving' or 'strided' smoothing.
        show (list): List of trace names to hide.
    
    Returns:
        fig: The updated Plotly figure.
    """
    # Initialize temporary arrays for smoothed time data.
    uf_time_b = np.zeros(10)
    uf_time_g = np.zeros(10)
    uf_time_r = np.zeros(10)

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
        if smooth_mode == 'moving':
            uf_time_b = uf(time['b'], lag)
            fig.update_traces(
                x=uf_time_b, y=uf(fret_b[i], lag),
                mode=mode_dict[scatter],
                visible=('FRET BG' not in show),
                selector=dict(name='fret_b')
            )
            fig.update_traces(
                x=uf_time_b, y=uf(bb[i], lag),
                mode=mode_dict[scatter],
                visible=('BB' not in show),
                selector=dict(name='bb')
            )
            fig.update_traces(
                x=uf_time_b, y=uf(bg[i], lag),
                mode=mode_dict[scatter],
                visible=('BG' not in show),
                selector=dict(name='bg')
            )
            fig.update_traces(
                x=uf_time_b, y=uf(br[i], lag),
                mode=mode_dict[scatter],
                visible=('BR' not in show),
                selector=dict(name='br')
            )
            fig.update_traces(
                x=uf_time_b, y=uf(bb[i] + bg[i] + br[i], lag),
                mode=mode_dict[scatter],
                line=dict(dash="longdash", width=2),
                visible=('Tot B' not in show),
                selector=dict(name='tot_b')
            )
            fig.update_traces(
                x=uf_time_b[[x[0] for x in bkps['b'][i]]],
                y=uf(bb[i], lag)[[y[0] for y in bkps['b'][i]]],
                mode='markers',
                selector=dict(name='b_bkps')
            )
            fig.update_traces(
                x=uf_time_b[[x[0] for x in bkps['fret_b'][i]]],
                y=uf(fret_b[i], lag)[[y[0] for y in bkps['fret_b'][i]]],
                mode='markers',
                selector=dict(name='fret_b_bkps')
            )
            hfilt_b = (uf_time_b > hist_range[0]) * (uf_time_b < hist_range[1])
            fig.update_traces(
                y=uf(fret_b[i], lag)[hfilt_b],
                selector=dict(name='Histogram_b')
            )
            if 'Tot B' in show:
                fig.update_layout(yaxis4=dict(range=(0, np.max(np.concatenate((uf(bb[i], lag), uf(bg[i], lag), uf(br[i], lag)))))))
            else:
                fig.update_layout(yaxis4=dict(range=(0, np.max((uf(bb[i], lag) + uf(bg[i], lag) + uf(br[i], lag))))))
        elif smooth_mode == 'strided':
            sa_time_b = sa(time['b'], lag)
            fig.update_traces(
                x=sa_time_b, y=sa(fret_b[i], lag),
                mode=mode_dict[scatter],
                visible=('FRET BG' not in show),
                selector=dict(name='fret_b')
            )
            fig.update_traces(
                x=sa_time_b, y=sa(bb[i], lag),
                mode=mode_dict[scatter],
                visible=('BB' not in show),
                selector=dict(name='bb')
            )
            fig.update_traces(
                x=sa_time_b, y=sa(bg[i], lag),
                mode=mode_dict[scatter],
                visible=('BG' not in show),
                selector=dict(name='bg')
            )
            fig.update_traces(
                x=sa_time_b, y=sa(br[i], lag),
                mode=mode_dict[scatter],
                visible=('BR' not in show),
                selector=dict(name='br')
            )
            fig.update_traces(
                x=sa_time_b, y=sa(bb[i] + bg[i] + br[i], lag),
                mode=mode_dict[scatter],
                line=dict(dash="longdash", width=2),
                visible=('Tot B' not in show),
                selector=dict(name='tot_b')
            )
            fig.update_traces(
                x=sa_time_b[[(x[0] // lag) for x in bkps['b'][i]]],
                y=sa(bb[i], lag)[[(y[0] // lag) for y in bkps['b'][i]]],
                mode='markers',
                selector=dict(name='b_bkps')
            )
            fig.update_traces(
                x=sa_time_b[[(x[0] // lag) for x in bkps['fret_b'][i]]],
                y=sa(fret_b[i], lag)[[(y[0] // lag) for y in bkps['fret_b'][i]]],
                mode='markers',
                selector=dict(name='fret_b_bkps')
            )
            hfilt_b = (sa_time_b > hist_range[0]) * (sa_time_b < hist_range[1])
            fig.update_traces(
                y=sa(fret_b[i], lag)[hfilt_b],
                selector=dict(name='Histogram_b')
            )
            if 'Tot B' in show:
                fig.update_layout(yaxis4=dict(range=(0, np.max(np.concatenate((sa(bb[i], lag), sa(bg[i], lag), sa(br[i], lag)))))))
            else:
                fig.update_layout(yaxis4=dict(range=(0, np.max((sa(bb[i], lag) + sa(bg[i], lag) + sa(br[i], lag))))))
    else:
        # Clear blue channel traces if no data available.
        selectors = ['fret_b', 'bb', 'bg', 'br', 'tot_b', 'b_bkps', 'fret_b_bkps']
        clear_trace(fig, selectors)

    # ---------------------------
    # Update Green Channel (fret_g) and associated signals.
    # ---------------------------
    if np.any(fret_g):
        if smooth_mode == 'moving':
            uf_time_g = uf(time['g'], lag)
            fig.update_traces(
                x=uf_time_g, y=uf(fret_g[i], lag),
                mode=mode_dict[scatter],
                visible=('FRET GR' not in show),
                selector=dict(name='fret_g')
            )
            fig.update_traces(
                x=uf_time_g, y=uf(gg[i], lag),
                mode=mode_dict[scatter],
                visible=('GG' not in show),
                selector=dict(name='gg')
            )
            fig.update_traces(
                x=uf_time_g, y=uf(gr[i], lag),
                mode=mode_dict[scatter],
                visible=('GR' not in show),
                selector=dict(name='gr')
            )
            fig.update_traces(
                x=uf_time_g, y=uf(gg[i] + gr[i], lag),
                mode=mode_dict[scatter],
                line=dict(dash="longdash", width=2),
                visible=('Tot G' not in show),
                selector=dict(name='tot_g')
            )
            fig.update_traces(
                x=uf_time_g[[x[0] for x in bkps['g'][i]]],
                y=uf(gg[i], lag)[[y[0] for y in bkps['g'][i]]],
                mode='markers',
                selector=dict(name='g_bkps')
            )
            fig.update_traces(
                x=uf_time_g[[x[0] for x in bkps['fret_g'][i]]],
                y=uf(fret_g[i], lag)[[y[0] for y in bkps['fret_g'][i]]],
                mode='markers',
                selector=dict(name='fret_g_bkps')
            )
            hfilt_g = (uf_time_g > hist_range[0]) * (uf_time_g < hist_range[1])
            fig.update_traces(
                y=uf(fret_g[i], lag)[hfilt_g],
                selector=dict(name='Histogram_g')
            )
        elif smooth_mode == 'strided':
            sa_time_g = sa(time['g'], lag)
            fig.update_traces(
                x=sa_time_g, y=sa(fret_g[i], lag),
                mode=mode_dict[scatter],
                visible=('FRET GR' not in show),
                selector=dict(name='fret_g')
            )
            fig.update_traces(
                x=sa_time_g, y=sa(gg[i], lag),
                mode=mode_dict[scatter],
                visible=('GG' not in show),
                selector=dict(name='gg')
            )
            fig.update_traces(
                x=sa_time_g, y=sa(gr[i], lag),
                mode=mode_dict[scatter],
                visible=('GR' not in show),
                selector=dict(name='gr')
            )
            fig.update_traces(
                x=sa_time_g, y=sa(gg[i] + gr[i], lag),
                mode=mode_dict[scatter],
                line=dict(dash="longdash", width=2),
                visible=('Tot G' not in show),
                selector=dict(name='tot_g')
            )
            fig.update_traces(
                x=sa_time_g[[(x[0] // lag) for x in bkps['g'][i]]],
                y=sa(gg[i], lag)[[(y[0] // lag) for y in bkps['g'][i]]],
                mode='markers',
                selector=dict(name='g_bkps')
            )
            fig.update_traces(
                x=sa_time_g[[(x[0] // lag) for x in bkps['fret_g'][i]]],
                y=sa(fret_g[i], lag)[[(y[0] // lag) for y in bkps['fret_g'][i]]],
                mode='markers',
                selector=dict(name='fret_g_bkps')
            )
            hfilt_g = (sa_time_g > hist_range[0]) * (sa_time_g < hist_range[1])
            fig.update_traces(
                y=sa(fret_g[i], lag)[hfilt_g],
                selector=dict(name='Histogram_g')
            )
        # Adjust y-axis for green channel based on additional signals.
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
        if smooth_mode == 'moving':
            uf_time_r = uf(time['r'], lag)
            fig.update_traces(
                x=uf_time_r, y=uf(rr[i], lag),
                mode=mode_dict[scatter],
                visible=('RR' not in show),
                selector=dict(name='rr')
            )
            fig.update_traces(
                x=uf_time_r[[x[0] for x in bkps['r'][i]]],
                y=uf(rr[i], lag)[[y[0] for y in bkps['r'][i]]],
                mode='markers',
                selector=dict(name='r_bkps')
            )
        elif smooth_mode == 'strided':
            sa_time_r = sa(time['r'], lag)
            fig.update_traces(
                x=sa_time_r, y=sa(rr[i], lag),
                mode=mode_dict[scatter],
                visible=('RR' not in show),
                selector=dict(name='rr')
            )
            fig.update_traces(
                x=sa_time_r[[(x[0] // lag) for x in bkps['r'][i]]],
                y=sa(rr[i], lag)[[(y[0] // lag) for y in bkps['r'][i]]],
                mode='markers',
                selector=dict(name='r_bkps')
            )
        if not np.any(fret_g):
            fig.update_layout(yaxis3=dict(range=(0, np.max(rr[i]))))
    else:
        selectors = ['rr', 'r_bkps']
        clear_trace(fig, selectors)

    # ---------------------------
    # Update HMM trace.
    # ---------------------------
    if np.any(hmm_fret_g[i]):
        fig.update_traces(
            x=uf(time['g'], lag)[:hmm_fret_g[i].shape[0]],
            y=hmm_fret_g[i].reshape(-1),
            visible=('HMM' not in show),
            selector=dict(name='HMM_fret_g')
        )
    else:
        fig.update_traces(x=[], y=[], selector=dict(name='HMM_fret_g'))

    # ---------------------------
    # Update x-axis range based on time data.
    # ---------------------------
    if np.any(fret_b) or np.any(fret_g) or np.any(rr):
        fig.update_layout(xaxis1=dict(
            range=(min(time['g'][0], time['b'][0], time['r'][0]),
                   max(time['g'][-1], time['b'][-1], time['r'][-1]))
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
