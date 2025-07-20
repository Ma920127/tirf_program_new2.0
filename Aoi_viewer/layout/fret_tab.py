from dash import dcc, html
import dash_bootstrap_components as dbc
import dash_daq as daq

def get_fret_tab(config):
    return html.Div([
        html.Div(id='FRET_mode', children=[
            dbc.Card([
                dbc.CardBody([
                    html.Div([
                        dcc.Loading(
                            id="loading2",
                            type='circle',
                            children=dbc.Button(id='FRET', n_clicks=0, outline=True, color="dark", className="bi bi-play-fill")
                        ),
                        html.Div('Preserve selected', style={"margin-left": "20px", "margin-right": "10px"}),
                        daq.ToggleSwitch(id='ps', value = config['preserve_selected'], color='green'),
                        html.Div('Folder', style={"margin-left": "20px", "margin-right": "10px"}),
                        dcc.Input(value = config['overwrite'], id='ow', type='number', step=1, placeholder="", style={'textAlign': 'center', 'width': '40px'})
                    ], style={'padding': 10, 'display': 'flex', 'flex-direction': 'row', 'align-items': 'center'})
                ])
            ], color="light", outline=True, style={'width': '750px', 'padding': 5}),
            dbc.Card([
                dbc.CardHeader("green-red FRET"),
                dbc.CardBody([
                    html.Div([
                        html.Div('Red', style={"margin-left": "20px", "margin-right": "10px"}),
                        daq.ToggleSwitch(id='red', value = config['red'], color='green'),
                        html.Div('Fit', style={"margin-left": "20px", "margin-right": "10px"}),
                        daq.ToggleSwitch(id='fit', value = config['fit'], color='green'),
                        html.Div('GFP plot', style={"margin-left": "20px", "margin-right": "10px"}),
                        daq.ToggleSwitch(id='gfp_plot', value= config['gfp_plot'] , color='green'),
                        html.Div('GFP hist', style={"margin-left": "20px", "margin-right": "10px"}),
                        daq.ToggleSwitch(id='gfp_hist', value= False , color='green'),
                    ], style={'padding': 10, 'display': 'flex', 'flex-direction': 'row'}),
                    html.Div([
                        html.Div('leakage'),
                        dcc.Input(value = config['leakage_g'], id="leakage_g", type="number", placeholder="", style={'textAlign': 'center', "margin-left": "10px", 'width': '80px'}),
                        html.Div('lag', style={"margin-left": "10px"}),
                        dcc.Input(value = config['f_lag'], id="f_lag", type="number", placeholder="", style={'textAlign': 'center', "margin-left": "10px", 'width': '60px'}),
                        html.Div('red intensity', style={"margin-left": "10px"}),
                        dcc.Input(value = config['red_intensity'], id="red_intensity", type="number", step=100, placeholder="", style={'textAlign': 'center', "margin-left": "10px", 'width': '80px'}),
                    ], style={'padding': 10, 'display': 'flex', 'flex-direction': 'row'}),
                    html.Div([
                        html.Div("Snap time"),
                        html.Div(dcc.RangeSlider(0, 20, step = 1, marks = None, value = config['snap_time_b'], tooltip={"placement": "bottom", "always_visible": True}, id='snap_time_g'), style={'width': '600px'}),
                    ], style={'padding': 10, 'display': 'flex', 'flex-direction': 'row'}),
                    html.Div([
                        html.Div("Red time"),
                        html.Div(dcc.RangeSlider(0, 20, step = 1, marks = None, value = config['red_time'], tooltip={"placement": "bottom", "always_visible": True}, id='red_time'), style={'width': '600px'})
                    ], style={'padding': 10, 'display': 'flex', 'flex-direction': 'row'}),
                ])
            ], color="light", outline=True, style={'width': '750px', 'padding': 5}),
            dbc.Card([
                dbc.CardHeader("blue-green FRET"),
                dbc.CardBody([
                    html.Div([
                        html.Div('Green', style={"margin-left": "20px", "margin-right": "10px"}),
                        daq.ToggleSwitch(id='green', value = config['green'], color='green'),
                        html.Div('Fit', style={"margin-left": "20px", "margin-right": "10px"}),
                        daq.ToggleSwitch(id='fit_b', value = config['fit_b'], color='green'),
                    ], style={'padding': 10, 'display': 'flex', 'flex-direction': 'row'}),
                    html.Div([
                        html.Div('leakage'),
                        dcc.Input(value = config['leakage_b'], id="leakage_b", type="number", placeholder="", style={'textAlign': 'center', "margin-left": "10px", 'width': '80px'}),
                        html.Div('lag', style={"margin-left": "10px"}),
                        dcc.Input(value = config['lag_b'], id="lag_b", type="number", placeholder="", style={'textAlign': 'center', "margin-left": "10px", 'width': '60px'}),
                        html.Div('green intensity', style={"margin-left": "10px"}),
                        dcc.Input(value = config['green_intensity'], id="green_intensity", type="number", step=100, placeholder="", style={'textAlign': 'center', "margin-left": "10px", 'width': '80px'}),
                    ], style={'padding': 10, 'display': 'flex', 'flex-direction': 'row'}),
                    html.Div([
                        html.Div("Snap time"),
                        html.Div(dcc.RangeSlider(0, 20, step = 1, marks = None, value = config['snap_time_b'], tooltip={"placement": "bottom", "always_visible": True}, id='snap_time_b'), style={'width': '600px'}),
                    ], style={'padding': 10, 'display': 'flex', 'flex-direction': 'row'}),
                    html.Div([
                        html.Div("Green time"),
                        html.Div(dcc.RangeSlider(0, 20, step = 1, marks = None, value = config['green_time'], tooltip={"placement": "bottom", "always_visible": True}, id='green_time'), style={'width': '600px'})
                    ], style={'padding': 10, 'display': 'flex', 'flex-direction': 'row'}),
                ])
            ], color="light", outline=True, style={'width': '750px', 'padding': 5}),
        ])
    ])
