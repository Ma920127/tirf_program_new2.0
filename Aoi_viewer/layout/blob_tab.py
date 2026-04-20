from dash import dcc, html
import dash_bootstrap_components as dbc
import dash_daq as daq

def get_blob_tab(config):
    return html.Div([
        html.Div([
            html.Div('Path : '),
            dcc.Interval(id="interval", interval=500),
            dcc.Input(
                value=None, id="path", type="text",
                placeholder="", style={'textAlign': 'left'},
                size='50', persistence=True
            ),
            html.Button(children=html.I(className="bi bi-cpu"), id='auto'),
            dcc.Loading(id="loading1", type='circle',
                        children=html.Button('Load', id='loadp')),
            html.Button('Open', id='openp'),
        ], style={'padding': 5, 'display': 'flex', 'flex-direction': 'row'}),
        html.Div([
            html.Div('Mapping Path : '),
            dcc.Input(
                id="mpath", value = config['mpath'], type="text",
                placeholder="", style={'textAlign': 'left'},
                size='30', persistence=True
            ),
            html.Div('Plot: ', style={"margin-left": "10px", "margin-right": "10px"}),
            daq.ToggleSwitch(id='plot_circle', value=False, color='green'),
            html.Div('Reverse', style={"margin-left": "10px"}),
            daq.ToggleSwitch(id='reverse', value=False, color='green')
        ], style={'padding': 5, 'display': 'flex', 'flex-direction': 'row'}),
        html.Div([
            html.Button('Blob and Fit', id='blob', disabled=True),
            html.Div('Thres', style={"margin-left": "10px"}),
            dcc.Input(
                value = config['thres'], id="thres", type="number", step=1,
                placeholder="", style={'textAlign': 'center', "margin-left": "10px", 'width': '40px'},
                persistence=True
            ),
            html.Div('Average', style={"margin-left": "10px"}),
            dcc.Input(
                value = config['average_frame'], id="average_frame", type="number",
                placeholder="", step=1, min=1,
                style={'textAlign': 'center', "margin-left": "8px", 'width': '60px'},
                persistence=True
            ),
            html.Div('Ratio', style={"margin-left": "10px"}),
            dcc.Input(
                value = config['ratio_thres'], id="ratio_thres", type="number", step=0.05,
                placeholder="", style={'textAlign': 'center', "margin-left": "8px", 'width': '60px'},
                persistence=True
            ),
            html.Div('Radius', style={"margin-left": "10px"}),
            dcc.Input(
                value = config['radius'], id="radius", type="number",
                placeholder="", style={'textAlign': 'center', 'width': '40px', "margin-left": "8px"},
                persistence=True
            ),
            html.Div('Min_D', style={"margin-left": "10px"}),
            
            # 🌟 AMENDMENT 1: Change value=6 to value=config.get('min_distance', 6)
            dcc.Input(
                value=config.get('min_distance', 8), 
                id="min_distance", type="number", step=1, 
                style={'textAlign': 'center', "margin-left": "8px", 'width': '40px'}, persistence=True
            ),
        ], style={'padding': 5, 'display': 'flex', 'flex-direction': 'row'}),

        html.Div([
            html.Div('Channel'),
            html.Div(
                dcc.Dropdown(['green', 'red', 'blue'], 'green', clearable=False, id='channel'),
                style={'width': '80px', "margin-left": "10px", "margin-right": "10px"}
            ),
            html.Div('Range'),
            dcc.Input(
                value = config['minf'], id="minf", type="number", step=1,
                placeholder="minf", persistence=True,
                style={'textAlign': 'center', "margin-left": "10px", 'width': '80px'}
            ),
            dcc.Input(
                value = config['maxf'], id="maxf", type="number", step=1,
                placeholder="maxf", persistence=True,
                style={'textAlign': 'center', 'width': '80px'}
            ),
            html.Button('Autoscale', id='autoscale', disabled=False),
            html.Button('Cal Intensity', id='cal_intensity', disabled=True),   
        ], style={'padding': 5, 'display': 'flex', 'flex-direction': 'row'}),

        html.Div([
            html.Button('Calculate Drift', id='cal_drift', disabled=False),
            html.Button('Load Drift', id='load_drift', disabled=False),
            html.Div('Interval', style={"margin-left": "10px"}),
            dcc.Input(
                value=200, id="per_n", type="number", min=1, step=1,
                placeholder="per_n", persistence=True,
                style={'textAlign': 'center', "margin-left": "10px", 'width': '80px'}
            ),
            html.Div('Pairing Threshold', style={"margin-left": "10px"}),
            dcc.Input(
                value=1.5, id="pairing_threshold", type="number", min=0.1, step=0.1,
                placeholder="pairing_threshold", persistence=True,
                style={'textAlign': 'center', "margin-left": "10px", 'width': '60px'}
            ),
        ], style={'padding': 5, 'display': 'flex', 'flex-direction': 'row'}),
        html.Div([
            html.Div('AOI tools:'),
            dbc.RadioItems(
                id="aoi_mode",
                options=[
                    {"label": html.I(className='bi bi-eraser',
                                       style={'font-size': 30, 'textAlign': 'center'},
                                       title='Remove'),
                     "value": 0, 'title': 'Remove'},
                    {"label": html.I(className='bi bi-plus-circle',
                                       style={'font-size': 30, 'textAlign': 'center'},
                                       title='Add'),
                     "value": 1, 'disabled': True},
                    {"label": html.I(className='bi bi-arrow-counterclockwise',
                                       style={'font-size': 30, 'textAlign': 'center'},
                                       title='Undo'),
                     "value": 2},
                    {"label": html.I(className='bi bi-save2',
                                       style={'font-size': 30, 'textAlign': 'center'},
                                       title='save'),
                     "value": 3},
                    {"label": html.I(className='bi bi-upload',
                                       style={'font-size': 30, 'textAlign': 'center'},
                                       title='load'),
                     "value": 4},
                    {"label": html.I(className='bi bi-cart-x',
                                       style={'font-size': 30, 'textAlign': 'center'},
                                       title='clear'),
                     "value": 5},
                ],
                value=0, style={"margin-left": "10px"}, inline=True
            ),
            html.Div(0, id='aoi_num', style={"margin-left": "10px"}),
        ], style={'padding': 5, 'display': 'flex', 'flex-direction': 'row'}),
        html.Div([
            html.Div('Move AOIs:', style={'textAlign': 'center', "margin-right": "30px"}),
            html.I(className="bi bi-arrow-left-square", id='left', style={'font-size': 30, "margin-right": "30px"}),
            html.I(className="bi bi-arrow-right-square", id='right', style={'font-size': 30, "margin-right": "30px"}),
            html.I(className="bi bi-arrow-down-square", id='down', style={'font-size': 30, "margin-right": "30px"}),
            html.I(className="bi bi-arrow-up-square", id='up', style={'font-size': 30, "margin-right": "10px"}),
            html.Div(
                dcc.Dropdown(['channel_r', 'channel_g', 'channel_b'], 'channel_r', clearable=False, id='selector', persistence=True),
                style={'width': '110px', "margin-right": "10px"}
            ),
            dcc.Input(value=1, id="move_step", type="number", step=1, style={"margin-right": "10px", 'width': '50px'}),
            dbc.Button('Fit', id='fit_gauss', outline=True, color="primary", className="me-1"),
        ], style={'padding': 5, 'display': 'flex', 'flex-direction': 'row'}),
        html.Div([
            html.Div('Load progress'),
            dbc.Progress(id="load_progress", value=0, color='success', label='0'),
            html.Div('Blob progress'),
            dbc.Progress(id="blob_progress", value=0, color='success', label='0'),
            html.Div('Intensity progress'),
            dbc.Progress(id="int_progress", value=0, color='success', label='0'),
            html.Div('FRET progress'),
            dbc.Progress(id="fret_progress", value=0, color='success', label='0'),
        ], style={'padding': 10, 'width': '600px'}),
        html.Div([
            html.Div('', style={'width': '400px'}, id='log'),
        ], style={'padding': 10, 'width': '600px'})
    ])
