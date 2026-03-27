import numpy as np
import plotly.graph_objects as go
from dash import Dash, dcc, callback_context, no_update
import dash_bootstrap_components as dbc
import plotly.express as px
from aoi_utils import draw_blobs, move_blobs, load_path, cal_blob_intensity, cal_FRET_utils, save_config, load_config, save_aoi_utils, load_aoi_utils
from dash_extensions.enrich import DashProxy, html, Output, Input, State, FileSystemCache, Trigger, CycleBreakerTransform, CycleBreakerInput
from dash.exceptions import PreventUpdate
import subprocess
import dash_daq as daq
import logging
import pickle


logging.getLogger('werkzeug').setLevel(logging.ERROR)

thres, mpath, redchi, average_frame, minf, maxf, channel, radius, leakage_g, leakage_b, f_lag, lag_b, snap_time_g, snap_time_b, red_intensity, red_time, red, green_intensity, green_time, green, fit, fit_b, GFP_plot, ps, ow = load_config(1)


anchor = 0
image_g = np.zeros((1, 512, 512))
image_r = np.zeros((1, 512, 512))
image_b = np.zeros((1, 512, 512))
coord_list = np.zeros(0)
blob_list = []
rem_hist = []
rem_hist_blob = []
org_size = 1
dr = radius
loader = None
blob_disable = True
image_datas = None
fsc = FileSystemCache("cache_dir")
fsc.set("load_progress", 0)
fsc.set("progress", 0)
fsc.set("cal_progress", 0)
fsc.set("fret_progress", 0)
fsc.set("stage", 'Idle')
fsc.set('mode', 'manual')


frame_slider = dcc.Slider(0, image_g.shape[0]-1, 1,
            value=0,
            updatemode='drag',
            tooltip={"placement": "bottom", "always_visible": False},
            marks=None,
            id='frame_slider')

fig = px.imshow(image_g[0], color_continuous_scale='gray', zmin=minf, zmax=maxf)

fig.update_layout(
    xaxis=dict(
        showline=True,
        range=(0, image_g.shape[2]),
        autorange =  False,
    ),
    
    yaxis=dict(
        showline=True,
        range = (image_g.shape[1], 0),
        autorange =  False
    ),
    width=1024, 
    height=1024,
    autosize = False,
    uirevision = True,
    dragmode = 'pan'

)

fig.add_scatter(x=[], y=[],
            mode ='markers',
            marker_symbol= 'square-open',
            marker = dict(
            color ='rgba(135, 206, 250, 0.5)',
            size = 2 * radius + 1,
            line = dict(
                color='MediumPurple',
                width=1
            )
        )
        ,name = 'blobs_r')

fig.add_scatter(x=[], y=[],
            mode = 'markers',
            marker_symbol = 'square-open',
            marker = dict(
            color = 'rgba(135, 206, 250, 0.5)',
            size = 2 * radius + 1,
            line = dict(
                color='MediumPurple',
                width=1
            )
        )
        ,name = 'blobs_g')

fig.add_scatter(x=[], y=[],
            mode='markers',
            marker_symbol= 'square-open',
            marker=dict(
            color='rgba(135, 206, 250, 0.5)',
            size = 2 * radius + 1,
            line = dict(
                color='MediumPurple',
                width=1
            )
        )
        ,name = 'blobs_b')


app = DashProxy(__name__, external_stylesheets=[dbc.themes.ZEPHYR, dbc.icons.BOOTSTRAP], prevent_initial_callbacks=True, transforms=[CycleBreakerTransform()])

app.layout = html.Div([

    dbc.Row([
        dbc.Col([
            dcc.Graph(figure=fig, id="graph", style={'width': 1024, 'height': 1024}, config = {'scrollZoom': True, 'modebar_remove' : ['box select', 'lasso select']} ),
            html.Div([html.Div(frame_slider, style={'width': 900}), 
            dcc.Input(value = 0, id="anchor", type="text", placeholder="", style={'textAlign': 'center'}, size='3', debounce = 1)
            ],style={"padding" : 5, 'display': 'flex','flex-direction': 'row'}),
        ]),

        dbc.Col([
            dcc.Tabs(id='tabs-example-1', value='tab-1', children=[

            ##BLOB TAB##               
                dcc.Tab(label='Blob', children=[          
                    html.Div([
                        html.Div('Path : '),
                        dcc.Interval(id="interval", interval=500),
                        dcc.Input(value= None, id="path", type="text", placeholder="", style={'textAlign': 'left'}, size= '50', persistence = True),
                        html.Button(children = html.I(className="bi bi-cpu"), id='auto'),
                        dcc.Loading(id="loading1", type='circle' ,children = html.Button('Load', id='loadp')),
                        html.Button('Open', id='openp'),
                    ], style={'padding': 5, 'display': 'flex', 'flex-direction': 'row'}),

                    html.Div([
                        html.Div('Mapping Path : '),
                        dcc.Input(id="mpath", value= mpath, type="text", placeholder="", style={'textAlign': 'left'}, size= '30', persistence = True),
                        html.Div('Plot: ', style={"margin-left": "10px", "margin-right": "10px"}),
                        daq.ToggleSwitch(id = 'plot_circle', value = 0, color = 'green'), 
                        html.Div('Reverse', style={"margin-left": "10px"}),
                        daq.ToggleSwitch(id = 'reverse', value = 0, color = 'green')
                    ], style={'padding': 5, 'display': 'flex', 'flex-direction': 'row'}),


                ##
                    html.Div([
                        html.Button('Blob and Fit', id='blob', disabled = True),
                        html.Div('Thres', style={"margin-left": "10px"}),
                        dcc.Input(value = thres, id="thres", type="number", step = 1, placeholder="", style={'textAlign': 'center', "margin-left": "10px", 'width': '40px'}, persistence = True),
                        html.Div('Average', style={"margin-left": "10px"}),
                        dcc.Input(value = redchi, id="average_frame", type="number", placeholder="", step = 1, min = 1, style={'textAlign': 'center', "margin-left": "8px", 'width': '60px'}, persistence = True),
                        html.Div('Ratio', style={"margin-left": "10px"}),
                        dcc.Input(value = average_frame, id="ratio_thres", type="number", step = 0.1, placeholder="", style={'textAlign': 'center', "margin-left": "8px", 'width': '60px'}, persistence = True),
                        html.Div('Radius', style={"margin-left": "10px"}),
                        dcc.Input(value = radius, id = "radius", type="number", placeholder="", style={'textAlign': 'center', 'width': '40px', "margin-left": "8px"}),
                       
                    ],style={'padding': 5, 'display': 'flex','flex-direction': 'row'}),

                ##           
                    html.Div([
                        html.Div('Channel'),
                        html.Div(dcc.Dropdown(['green', 'red', 'blue'], 'green', clearable = False, id = 'channel'), style = {'width': '80px', "margin-left": "10px", "margin-right": "10px"}),
                        html.Div('Range'),
                        dcc.Input(value = minf, id="minf", type="number", step = 1, placeholder="minf", persistence = True, style={'textAlign': 'center', "margin-left": "10px", 'width': '80px'}),
                        dcc.Input(value = maxf, id="maxf", type="number", step = 1, placeholder="maxf", persistence = True, style={'textAlign': 'center', 'width': '80px'}),
                        html.Button('Autoscale', id='autoscale', disabled = False),
                        html.Button('Cal Intensity', id='cal_intensity', disabled = True),
                    ], style={'padding': 5, 'display': 'flex','flex-direction': 'row'}),   
                ##

                    html.Div([
                        html.Div('AOI tools:'),
                        dbc.RadioItems(id="aoi_mode", 
                            options=[
                                {"label": html.I(className = 'bi bi-eraser', style = {'font-size' : 30, 'textAlign': 'center'}, title = 'Remove'), "value": 0, 'title' : 'Remove'},
                                {"label": html.I(className = 'bi bi-plus-circle', style = {'font-size' : 30, 'textAlign': 'center'}, title = 'Add'), "value": 1, 'disabled' : True},
                                {"label": html.I(className = 'bi bi-arrow-counterclockwise', style = {'font-size' : 30, 'textAlign': 'center'}, title = 'Undo'), "value": 2},
                                {"label": html.I(className = 'bi bi-save2', style = {'font-size' : 30, 'textAlign': 'center'}, title = 'save'), "value": 3},
                                {"label": html.I(className = 'bi bi-upload', style = {'font-size' : 30, 'textAlign': 'center'}, title = 'load'), "value": 4},
                                {"label": html.I(className = 'bi bi-cart-x', style = {'font-size' : 30, 'textAlign': 'center'}, title = 'clear'), "value": 5},
                            ], value = 0, style={"margin-left": "10px"}, inline=True),
                        html.Div(0, id = 'aoi_num', style={"margin-left": "10px"}),
                    ], style={'padding': 5, 'display': 'flex','flex-direction': 'row'}),  

                    html.Div([
                        html.Div('Move AOIs:', style = {'textAlign': 'center', "margin-right": "30px"}),
                        html.I(className="bi bi-arrow-left-square", id='left', style = {'font-size' : 30, "margin-right": "30px"}),
                        html.I(className="bi bi-arrow-right-square", id='right', style = {'font-size' : 30, "margin-right": "30px"}),
                        html.I(className="bi bi-arrow-down-square", id='down', style = {'font-size' : 30, "margin-right": "30px"}),
                        html.I(className="bi bi-arrow-up-square", id='up', style = {'font-size' : 30, "margin-right": "10px"}),
                        html.Div(dcc.Dropdown(['channel_r', 'channel_g', 'channel_b'], 'channel_r', clearable = False, id = 'selector', persistence = True), style = {'width': '110px', "margin-right": "10px"}), 
                        dcc.Input(value = 1, id="move_step", type="number", step = 1, style={"margin-right": "10px", 'width': '50px'}),
                        dbc.Button('Fit', id = 'fit_gauss', outline = True, color="primary", className="me-1"),   
                    ], style={'padding': 5, 'display': 'flex','flex-direction': 'row'}),   

                    html.Div([
                        html.Div('Load progress'),
                        dbc.Progress(id="load_progress", value = 0, color = 'success', label = '0'),
                        html.Div('Blob progress'),
                        dbc.Progress(id="blob_progress", value = 0, color = 'success', label = '0'),
                        html.Div('Intensity progress'),
                        dbc.Progress(id="int_progress", value = 0, color = 'success', label = '0'),  
                        html.Div('FRET progress'),
                        dbc.Progress(id="fret_progress", value = 0, color = 'success', label = '0'), 
                    ], style = {'padding': 10, 'width': '600px'}),

                    html.Div([
                        html.Div('', style = {'width': '400px'}, id = 'log'),   
                    ], style = {'padding': 10, 'width': '600px'})
                ]),

            ##FRET TAB##
                dcc.Tab(label='FRET', children=[
                    html.Div(id='FRET_mode', children=[
                        ##common TAB##
                        dbc.Card([
                            dbc.CardBody([  
                                html.Div([
                                    dcc.Loading(id="loading2", type='circle', children =  dbc.Button(id='FRET', n_clicks = 0, outline=True, color="dark", className = "bi bi-play-fill")),
                                    html.Div('Preserve selected', style={"margin-left": "20px", "margin-right": "10px"}),
                                    daq.ToggleSwitch(
                                            id = 'ps',
                                            value = ps,
                                            color = 'green'
                                        ), 
                                    html.Div('Folder', style={"margin-left": "20px", "margin-right": "10px"}),
                                    dcc.Input(value = int(ow), id = 'ow', type = 'number', step = 1, placeholder="", style={'textAlign': 'center', 'width': '40px'})
                                ],style={'padding': 10, 'display': 'flex','flex-direction': 'row', 'align-items' : 'center'} ) 
                            ])
                        ], color="light", outline=True, style = {'width': '750px', 'padding' : 5}),
                        ##gr TAB##
                        dbc.Card([
                            dbc.CardHeader("green-red FRET"),
                            dbc.CardBody([  
                                html.Div([
                                    html.Div('Red', style={"margin-left": "20px", "margin-right": "10px"}),
                                    daq.ToggleSwitch(
                                        id = 'red',
                                        value = red,
                                        color = 'green'
                                    ),  
                                    html.Div('Fit', style={"margin-left": "20px", "margin-right": "10px"}),
                                    daq.ToggleSwitch(
                                    id='fit',
                                    value = fit,
                                    color = 'green'
                                    ),  
                                    html.Div('GFP plot', style={"margin-left": "20px", "margin-right": "10px"}),
                                    daq.ToggleSwitch(
                                    id='gfp_plot',
                                    value = GFP_plot,
                                    color = 'green'
                                    ),  
                                ],style={'padding': 10, 'display': 'flex','flex-direction': 'row'}),

                                html.Div([
                                    html.Div('leakage'),
                                    dcc.Input(value = leakage_g, id="leakage_g", type="number", placeholder="", style={'textAlign': 'center', "margin-left": "10px", 'width': '80px'}),
                                    html.Div('lag', style={"margin-left": "10px"}),
                                    dcc.Input(value = f_lag, id="f_lag", type="number", placeholder="", style={'textAlign': 'center', "margin-left": "10px", 'width': '60px'}),
                                    html.Div('red intensity', style={"margin-left": "10px"}),
                                    dcc.Input(value = red_intensity, id="red_intensity", type="number", step = 100, placeholder="", style={'textAlign': 'center', "margin-left": "10px", 'width': '80px'}),
                                ],style={'padding': 10, 'display': 'flex','flex-direction': 'row'}),

                                html.Div([
                                    html.Div("Snap time"),
                                    html.Div(dcc.RangeSlider(0, snap_time_g[1], value = snap_time_g, tooltip={"placement": "bottom", "always_visible": True}, id = 'snap_time_g'), style = {'width': '600px'}),
                                ],style={'padding': 10, 'display': 'flex','flex-direction': 'row'}),

                                html.Div([
                                    html.Div("Red time"),
                                    html.Div(dcc.RangeSlider(0, red_time[1], value = red_time, tooltip={"placement": "bottom", "always_visible": True}, id = 'red_time'), style = {'width': '600px'})
                                ],style={'padding': 10, 'display': 'flex','flex-direction': 'row'}),
                            ])
                        ], color="light", outline=True, style = {'width': '750px', 'padding' : 5}),
                        ##bg TAB##
                        dbc.Card([
                            dbc.CardHeader("blue-green FRET"),
                            dbc.CardBody([  
                                html.Div([
                                    html.Div('Green', style={"margin-left": "20px", "margin-right": "10px"}),
                                    daq.ToggleSwitch(
                                        id = 'green',
                                        value = green,
                                        color = 'green'
                                    ),  
                                    html.Div('Fit', style={"margin-left": "20px", "margin-right": "10px"}),
                                    daq.ToggleSwitch(
                                    id='fit_b',
                                    value = fit_b,
                                    color = 'green'
                                    ),  
                                ],style={'padding': 10, 'display': 'flex','flex-direction': 'row'}),

                                html.Div([
                                    html.Div('leakage'),
                                    dcc.Input(value = leakage_b, id="leakage_b", type="number", placeholder="", style={'textAlign': 'center', "margin-left": "10px", 'width': '80px'}),
                                    html.Div('lag', style={"margin-left": "10px"}),
                                    dcc.Input(value = lag_b, id="lag_b", type="number", placeholder="", style={'textAlign': 'center', "margin-left": "10px", 'width': '60px'}),
                                    html.Div('green intensity', style={"margin-left": "10px"}),
                                    dcc.Input(value = green_intensity, id="green_intensity", type="number", step = 100, placeholder="", style={'textAlign': 'center', "margin-left": "10px", 'width': '80px'}),
                                ],style={'padding': 10, 'display': 'flex','flex-direction': 'row'}),


                                html.Div([
                                    html.Div("Snap time"),
                                    html.Div(dcc.RangeSlider(0, max(snap_time_b[1], 20), value = snap_time_b, tooltip={"placement": "bottom", "always_visible": True}, id = 'snap_time_b'), style = {'width': '600px'}),
                                ],style={'padding': 10, 'display': 'flex','flex-direction': 'row'}),

                                html.Div([
                                    html.Div("Green time"),
                                    html.Div(dcc.RangeSlider(0, max(green_time[1], 20), value = green_time, tooltip={"placement": "bottom", "always_visible": True}, id = 'green_time'), style = {'width': '600px'})
                                ],style={'padding': 10, 'display': 'flex','flex-direction': 'row'}),
                            ])
                        ], color="light", outline=True, style = {'width': '750px', 'padding' : 5}),



                    ]),
                ##
                ])


            ], style = {'width': '600px', 'padding' : 5}),


            html.Div([
                dbc.RadioItems(id="configs", className="btn-group", inputClassName="btn-check", labelClassName="btn btn-outline-primary", labelCheckedClassName="active",
                options=[
                    {"label": "Config 1", "value": 1},
                    {"label": "Config 2", "value": 2},
                    {"label": "Config 3", "value": 3},
                    {"label": "Config 4", "value": 4},
                ], value = 1, style={'width' : 500}, labelStyle={'width':'100%'}),

                html.Button('Save Config', id='savec', className = "btn btn-outline-primary"),

            ],style={'padding': 20, 'display': 'flex','flex-direction': 'row'})

        ]),    
            
    ],  align="center")

])       



@app.callback(Output('graph', 'figure'), 
            Output('graph', 'clickData'),
            Output('anchor', 'value'),
            Output('blob', 'disabled'),
            Output('cal_intensity', 'disabled'),
            Output('frame_slider', 'value'),
            Output('frame_slider', 'max'),
            Output("snap_time_g", "max"),
            Output("red_time", "max"),
            Output("snap_time_b", "max"),
            Output("green_time", "max"),
            Output('aoi_mode', "value"),
            Output('aoi_num', "children"),
            Output("loadp", "title"),
            Output("FRET", "outline"),
            Output("auto", "n_clicks"),
            Input('graph', 'clickData'),
            Input("graph", "relayoutData"),
            Input('blob', 'n_clicks'),
            Input('up', 'n_clicks'),
            Input('down', 'n_clicks'),
            Input('left', 'n_clicks'),
            Input('right', 'n_clicks'),
            Input('frame_slider', 'value'),
            Input('anchor', 'value'),
            Input('average_frame', 'value'),
            Input('loadp', 'n_clicks'),
            Input('minf', 'value'),
            Input('maxf', 'value'),
            Input('reverse', 'value'),
            Input('channel', 'value'),
            Input('cal_intensity', 'n_clicks'),
            Input("openp", "n_clicks"),
            Input("configs", 'value'),
            Input("aoi_mode", 'value'),
            State('ratio_thres', 'value'),
            State('radius', 'value'),
            State('selector', 'value'),
            State('move_step', 'value'),
            State('path', 'value'),
            State('mpath', 'value'),
            State("plot_circle", 'value'),
            # State('thres', 'value'),
            State("snap_time_g", "value"),
            State("red_time", "value"),
            State("auto", "n_clicks"),
            
)

def update_fig(clickData, relayout, blob, up, down, left, right, frame, anchor, average_frame, loadp, minf, maxf, reverse, channel, cal_intensity, openp, configs, aoi_mode, ratio_thres, radius, selector, move_step, path, mpath, plot, thres, snap_time, red_time, auto):

    global  rem_hist, rem_hist_blob, org_size, bac_mode, dr, coord_list, blob_list, fig, image_g, image_r, image_b, loader, blob_disable, image_datas, fsc

    changed_id = [p['prop_id'] for p in callback_context.triggered][0]

    if ('blob' in changed_id): 
        fsc.set("progress", 0)
        loader.gen_dimg(anchor = anchor, mpath = mpath, maxf = maxf, minf = minf, laser = channel, average_frame = average_frame)
        blob_list = loader.det_blob(plot = plot, fsc = fsc, thres = thres, r = radius, ratio_thres = float(ratio_thres))
        coord_list = []
        for b in blob_list:
            coord_list.append(b.get_coord())
        coord_list = np.array(coord_list)
        fig = draw_blobs(fig, coord_list, dr, reverse)
        fsc.set("stage", 'Blobing Finished')

    for move_button in ['up', 'down', 'left', 'right']:
        if (move_button in changed_id):
            coord_list = move_blobs(coord_list, selector, int(move_step), changed_id)
            fig = draw_blobs(fig, coord_list, dr, reverse)

    if ('loadp' in changed_id):
        fsc.set("load_progress", '0')
        loader, image_g, image_r, image_b, image_datas = load_path(thres, path, fsc)
        blob_disable = False
        frame = 0
        fsc.set("stage", 'Image Loaded')

    if ('openp' in changed_id):
        subprocess.Popen(f'explorer "{path}"')

    if ('cal_intensity' in changed_id):
        fsc.set("cal_progress", 0)
        cal_blob_intensity(loader, coord_list, path, image_datas, maxf, minf, fsc)
        fsc.set("stage", 'Intensity Calculated')

    if 'graph' in changed_id:
        if isinstance(relayout, dict):
            try:
                size = np.round(512 / (relayout['xaxis.range[1]'] - relayout['xaxis.range[0]']), 2) 
                if size != org_size:
                    org_size = size
                    dr = radius * size
                    if coord_list.any(): 
                        fig = draw_blobs(fig, coord_list, dr, reverse)
            except:
                pass

        if isinstance(clickData, dict):
                if clickData["points"][0]["curveNumber"] in [1, 2, 3]:
                    if aoi_mode == 0:
                        remove_id = clickData["points"][0]["pointNumber"]
                        rem_hist.append(coord_list[remove_id])
                        rem_hist_blob.append(blob_list[remove_id])
                        coord_list = np.delete(coord_list, remove_id, 0) 
                        blob_list.pop(remove_id)
                        fig = draw_blobs(fig, coord_list, dr, reverse)
                        


    #undo oi
    if aoi_mode == 2:
        aoi_mode = 0
        if len(rem_hist) > 0:
            coord_list = np.concatenate((coord_list, rem_hist[-1].reshape(1, 12)), axis = 0)
            blob_list.append(rem_hist_blob[-1])
            rem_hist.pop()
            rem_hist_blob.pop()
            fig = draw_blobs(fig, coord_list, dr, reverse)

    #save aoi
    if aoi_mode == 3:
        aoi_mode = 0
        save_aoi_utils(blob_list, path + r'\\aoi.dat')
        # np.save(path + r'\\aoi.npy', coord_list)
        print("saved_aoi")
      
    #load aoi
    if aoi_mode == 4:
        aoi_mode = 0
        blob_list = load_aoi_utils(path + r'\\aoi.dat')
        coord_list = []
        for b in blob_list:
            coord_list.append(b.get_coord())
        coord_list = np.array(coord_list) 
        #coord_list = np.load(path + r'\\aoi.npy')
        fig = draw_blobs(fig, coord_list, dr, reverse)
        print("loaded_aoi")
    
    #clear aoi
    if aoi_mode == 5:
        aoi_mode = 0
        blob_list = []
        coord_list = np.zeros(0)
        fig = draw_blobs(fig, coord_list, dr, reverse)
        print("cleared_aoi")
    


    channel_dict = {
        'green' : image_g,
        'red' : image_r,
        'blue' : image_b
    }
    
    if 'channel' in changed_id:
        if frame > channel_dict[channel].shape[0]:
            frame = 0


    if 'anchor.value' in changed_id:
        if int(anchor) < channel_dict[channel].shape[0]:
            frame = int(anchor)
    fig.update_traces(zmax = maxf, zmin = minf, selector = dict(type = 'heatmap')) 

    if 'reverse.value' in changed_id:
        if int(reverse) == 0:
            fig['layout']['coloraxis']['colorscale'] = 'gray'
            fig = draw_blobs(fig, coord_list, dr, reverse)
        else:
            fig['layout']['coloraxis']['colorscale'] = 'gray_r'
            fig = draw_blobs(fig, coord_list, dr, reverse)


    end = min(channel_dict[channel].shape[0], int(frame) + int(average_frame))
    start = max(0, end - int(average_frame))
    fig['data'][0]['z'] = np.average(channel_dict[channel][start: end], axis = 0)

    fig['layout']['coloraxis']['cmax'] = maxf
    fig['layout']['coloraxis']['cmin'] = minf
    slider_max = channel_dict[channel].shape[0] 
    snap_g_max = max(channel_dict['green'].shape[0]-1, snap_time_g[1]) 
    r_max = max(channel_dict['red'].shape[0]-1, red_time[1])
    snap_b_max = max(channel_dict['blue'].shape[0]-1, snap_time_b[1]) 
    g_max = max(channel_dict['green'].shape[0]-1, green_time[1])
    anchor = min(int(frame), channel_dict[channel].shape[0])
    aoi_num = coord_list.shape[0]

    if fsc.get("mode") != 'auto':
        auto_state = no_update
    else: 
        auto_state = auto + 1

    return fig, None, anchor, blob_disable, blob_disable, anchor, slider_max, snap_g_max, r_max, snap_b_max, g_max, aoi_mode, aoi_num, None, True, auto_state




#Load Config
@app.callback(
        output = [
        # Output('thres', 'value'),
        Output('mpath', 'value'),
        Output('average_frame', 'value'),
        Output('ratio_thres', 'value'),
        Output('minf', 'value'),
        Output('maxf', 'value'),
        Output('channel', 'value'),
        Output('radius', 'value'),
        Output('leakage_g', 'value'),
        Output('leakage_b', 'value'),
        Output('f_lag', 'value'),
        Output('lag_b', 'value'),
        Output('snap_time_g', 'value'),
        Output('snap_time_b', 'value'),
        Output('red_intensity', 'value'),
        Output('red_time', 'value'),
        Output('red', 'value'),
        Output('green_intensity', 'value'),
        Output('green_time', 'value'),
        Output('green', 'value'),
        Output('fit', 'value'),
        Output('fit_b', 'value'),
        Output('gfp_plot', 'value'),
        Output('ps', 'value'),
        Output('ow', 'value')],

        inputs = dict(
        configs = Input('configs', 'value'),
        savec = Input('savec', 'n_clicks'),
        autoscale = Input('autoscale', 'n_clicks'),
        config_data = dict(
        # thres = Input('thres', 'value'),
        mpath = Input('mpath', 'value'),
        average_frame = Input('average_frame', 'value'),
        ratio_thres = Input('ratio_thres', 'value'),
        minf = Input('minf', 'value'),
        maxf = Input('maxf', 'value'),
        channel = Input('channel', 'value'),
        radius = Input('radius', 'value'),
        leakage_g = Input('leakage_g', 'value'),
        leakage_b = Input('leakage_b', 'value'),
        f_lag = Input('f_lag', 'value'),
        lag_b = Input('lag_b', 'value'),
        snap_time_g = Input('snap_time_g', 'value'),
        snap_time_b = Input('snap_time_b', 'value'),
        red_intensity = Input('red_intensity', 'value'),
        red_time = Input('red_time', 'value'),
        red = Input('red', 'value'),
        green_intensity = Input('green_intensity', 'value'),
        green_time = Input('green_time', 'value'),
        green = Input('green', 'value'),
        fit = Input('fit', 'value'),
        fit_b = Input('fit_b', 'value'),
        gfp_plot = Input('gfp_plot', 'value'),
        preserve_selected = Input('ps', 'value'),
        overwrite = Input('ow', 'value'))
        )
)
def load_config_callback(configs, savec, autoscale, config_data):
    global fig

    changed_id = [p['prop_id'] for p in callback_context.triggered][0]

    if 'autoscale' in changed_id:
        maxf = np.round(np.max(fig['data'][0]['z']))
        minf = np.round(np.min(fig['data'][0]['z']))
        config_data['maxf'] = maxf
        config_data['minf'] = minf
        return list(config_data.values())


    if 'savec' in changed_id:
        save_config(configs, config_data)
        raise PreventUpdate

    if 'configs' in changed_id:
        config_data = load_config(int(configs))
        if config_data == None:
            raise PreventUpdate
        return config_data
    else:
        raise PreventUpdate



#Show Progress
@app.callback(Output("load_progress", "value"), 
              Output("load_progress", "label"),
              Output("blob_progress", "value"), 
              Output("blob_progress", "label"),
              Output("int_progress", "value"), 
              Output("int_progress", "label"),
              Output("fret_progress", "value"), 
              Output("fret_progress", "label"),
              Trigger("interval", "n_intervals")
)
def update_progress(self):
    
    load_prog = fsc.get("load_progress")
    blob_prog = fsc.get("progress")
    cal_prog = fsc.get("cal_progress")
    fret_prog = fsc.get("fret_progress")

    if load_prog == None:
        fsc.set("load_progress", 0)
        load_prog = 0
        
    load_prog = float(load_prog) * 100
    load_prog_label = f"{load_prog:.0f} %" if load_prog > 5 else ""

    if blob_prog == None:
        fsc.set("progress", 0)
        blob_prog = 0
    blob_prog = float(blob_prog) * 100
    blob_prog_label = f"{blob_prog:.0f} %" if blob_prog > 5 else ""

    if cal_prog == None:
        fsc.set("cal_progress", 0)
        cal_prog = 0
    cal_prog = float(cal_prog) * 100
    cal_prog_label = f"{cal_prog:.0f} %" if cal_prog > 5 else ""

    if fret_prog == None:
        fsc.set("fret_progress", 0)
        fret_prog = 0
    fret_prog = float(fret_prog) * 100
    fret_prog_label = f"{fret_prog:.0f} %" if fret_prog > 5 else ""

    return load_prog, load_prog_label, blob_prog, blob_prog_label, cal_prog, cal_prog_label, fret_prog, fret_prog_label



@app.callback(Output("log", "children"),
              Trigger("interval", "n_intervals")
)
def update_log(self):
   
    value = fsc.get("stage")
    if value == None:
        fsc.set("stage", "Idle")
    return value


@app.callback(Output("FRET", "title"), 
              Trigger("FRET", "n_clicks"), 
              State("path", "value"),
              State("ps", "value"),
              State('ow', 'value'),
              State("leakage_g", "value"),
              State("leakage_b", "value"),
              State("f_lag", "value"),
              State("lag_b", "value"),
              State("red", "value"),
              State("green", "value"),
              State("fit", "value"),
              State("fit_b", "value"),
              State("gfp_plot", "value"),
              State("snap_time_g", "value"),
              State("snap_time_b", "value"),
              State("red_time", "value"),
              State("green_time", "value"),
              State("red_intensity", "value"),
              State("green_intensity", "value"))
def cal_FRET(FRET, path, ps, ow, leakage_g, leakage_b, f_lag, lag_b, red, green, fit, fit_b, gfp_plot, snap_time_g, snap_time_b, red_time, green_time, red_intensity, green_intensity):
    global fsc

    fsc.set("fret_progress", 0)
    cal_FRET_utils(path, ps, ow, snap_time_g, snap_time_b, red, red_time, red_intensity, green, green_time, green_intensity, leakage_g, leakage_b, f_lag, lag_b, fit, fit_b, gfp_plot, fsc)
    print('Done')
    fsc.set("stage", 'Idle')

    return None


@app.callback(Output("loadp", 'n_clicks'), 
              Output('blob', 'n_clicks'),
              Output('cal_intensity', 'n_clicks'),
              Output('FRET', 'n_clicks'),
              CycleBreakerInput("auto", "n_clicks"))
def auto(auto):
    stage = fsc.get("stage")
    fsc.set('mode', 'auto')

    if stage == 'Idle':
        fsc.set("stage", 'Loading Image')
        return -1, no_update, no_update, no_update
    
    elif stage == 'Image Loaded':
        fsc.set("stage", 'Blobing and Fitting')
        return no_update, -1, no_update, no_update
    
    elif stage == 'Blobing Finished':
        fsc.set("stage", 'Calculating intensity')
        return no_update, no_update, -1, no_update
    
    elif stage == 'Intensity Calculated':
        fsc.set("stage", 'Calculating FRET')
        fsc.set('mode', 'manual')
        return no_update, no_update, no_update, -1
    
    else:
        fsc.set('mode', 'manual')
        return no_update, no_update, no_update, no_update




server = app.server 
if __name__ == '__main__':
   app.run_server(debug = False)
