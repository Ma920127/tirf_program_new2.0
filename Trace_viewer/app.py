import numpy as np

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



fig, fig_blob, fig2 = init_fig() 
app = make_app(fig, fig_blob, fig2)

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
    Input('strided', 'value'),
    Input('rescale', 'n_clicks'),
    Input("graph", "relayoutData"),
    Input('channel', 'value'),
    Input('chp_find_0','n_clicks'),
    Input('chp_find_1','n_clicks'),
    Input('confirm-reset', 'submit_n_clicks'),
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
    )


def update_fig(key_events, show, next, previous, go, dtime, etime, clickData, mode, save, load, loadp, rupture, good, bad, coloc, select, scatter, smooth, strided, rescale, relayout, channel, chp_find_0, chp_find_1, confirm_reset, i, path, 
               chp_mode_0, chp_comp_0, chp_thres_0, chp_channel_0, chp_target_0, chp_mode_1, chp_comp_1, chp_thres_1, chp_channel_1, chp_target_1, event):
    changed_id = [p['prop_id'] for p in callback_context.triggered][0]
    global N_traces, fig, fig2, idx, total_frame, color, good_style, bad_style, bmode
    global new, time_b, N_traces, total_frame, tot_dtime
    global fret_g, fret_b, rr, gg, gr, bb, bg, br, time, tot_g, tot_b, hmm_fret_g
    global select_list_g, colocalized_list, bkps, ch_label, blobs
    if ('n_events' in changed_id) and (event['key'] not in ['q', 'w', 'z', 'x', 'c']):
        raise PreventUpdate()

    if fig['layout']['uirevision'] == False:
        fig['layout']['uirevision'] = True   
    
    strided_dict = ['moving', 'strided']
    smooth_mode = strided_dict[int(strided)]

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
     
    ##breakpoints##
    bkps, mode, confirm_reset_show, channel_error_show = breakpoints_utils(changed_id, clickData, mode, channel, i, time, bkps, smooth, smooth_mode)
    
    ##save / load breakpoints##
    bkps = sl_bkps(changed_id, path, bkps, mode)

    bkps, channel_error_show = find_chp(changed_id, fret_g, fret_b, rr, gg, gr, bb, bg, br, i, time, select_list_g, 
             chp_mode_0, chp_comp_0, chp_thres_0, chp_channel_0, chp_target_0, chp_mode_1, chp_comp_1, chp_thres_1, chp_channel_1, chp_target_1,
             bkps, smooth, smooth_mode)

    
    # if 'rupture' in changed_id:
        
    #     rup=Rupture(fret[i])
    #     tot_bkps[i]=rup.det_bkps()      
    #     fig.layout.shapes=[]
    #     fig = draw(fig,tot_bkps,i,time_gr,dead_time,color,total_frame)
        
    
    ##update trace##
    fig = update_trace(fig, relayout, i, scatter, fret_g, fret_b, rr, gg, gr, bb, bg, br, time, hmm_fret_g, bkps, smooth, smooth_mode, show)

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
    State('strided', 'value'),
    blocking = True  
    )
def show_blob_main(hoverData, aoi_max, i, tabs, smooth, strided):
    global fig_blob, time
    if tabs == 'Aois' :
        fig_blob = show_blob(blobs, fig_blob, smooth, i, hoverData, time, aoi_max, strided)
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

    
        
server = app.server 
if __name__ == '__main__':
   app.run_server(debug = False)













