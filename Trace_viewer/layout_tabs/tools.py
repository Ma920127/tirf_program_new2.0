from dash import dcc, html
import dash_bootstrap_components as dbc
import dash_daq as daq
from dash_extensions.enrich import dash_table

def tools_tab():
    return dcc.Tab(
        label='Tools',
        value='Tools',
        children=[
            # Row 1: Trace Visibility, Smoothing, and Mode Controls
            html.Div(
                children=[
                    html.Div('Hide Trace:', style={"margin-left": "60px", "padding": 5, "white-space": "nowrap"}),
                    html.Div(
                        children=[
                            dcc.Checklist(
                                ['BB', 'BG', 'BR', 'GG', 'GR', 'RR', 'FRET BG', 'FRET GR', 'Tot B', 'Tot G', 'HMM'],
                                ['Tot B', 'Tot G'],
                                inline=True,
                                inputStyle={"margin-left": "10px"},
                                id='show',
                                persistence=True
                            )
                        ],
                        style={'padding': 5, 'white-space': 'nowrap', 'flex-shrink': 0}
                    ),
                    
                    # --- NEW SMOOTHING CONTROLS ---
                    html.Div('Method:', style={"margin-left": "20px"}),
                    dcc.Dropdown(
                        id='smooth_method',
                        options=[
                            {'label': 'Moving Avg (uf)', 'value': 'moving'},
                            {'label': 'Strided Avg (sa)', 'value': 'strided'},
                            {'label': 'Median (mf)', 'value': 'median'},
                            {'label': 'Savgol (sg)', 'value': 'savgol'}
                        ],
                        value='moving',
                        clearable=False,
                        searchable=False,
                        style={"margin-left": "5px", "width": "160px"},
                        persistence=True
                    ),
                    html.Div('Window:', style={"margin-left": "10px"}),
                    dcc.Input(
                        id='smooth', type='number', value=1, min=1, step=1,
                        persistence=True,
                        style={"margin-left": "5px", "width": "60px"}
                    ),
                    html.Div(
                        id='poly-container',
                        children=[
                            html.Div('Order:', style={"margin-left": "10px"}),
                            dcc.Input(
                                id='polyorder', type='number', value=2, min=1,
                                persistence=True,
                                style={"margin-left": "5px", "width": "50px"}
                            ),
                        ],
                        style={'display': 'none'}  # Controlled by app.py callback
                    ),
                    # ------------------------------

                    html.Div('Scatter:', style={"margin-left": "20px", "padding": 5}),
                    daq.ToggleSwitch(
                        id='scatter', value=0, color='green',
                        style={"margin-left": "10px", "padding": 5}
                    ),
                    
                    # Note: Original 'Strided Smooth' toggle removed since it's now in the Method Dropdown above
                    
                    html.Div(
                        children=[
                            dcc.RadioItems(
                                ['Add', 'Remove', 'Except', "Clear", "Clear All", "Set All", "Reset"],
                                'Add',
                                id='AR',
                                labelStyle={'display': 'inline-block', "marginTop": "5px"}
                            )
                        ],
                        style={'padding': 30}
                    ),
                   
                ],
                style={'display': 'flex', 'flex-direction': 'row', 'flex-wrap': 'wrap', 'align-items': 'center', 'width': '100%'}
            ),

            # Row 2: Navigation Buttons
            html.Div(
                children=[
                    html.Button('Set Dead time', id='dtime'),
                    html.Button('Set End time', id='etime'),
                    html.Button('Previous', id='previous', accessKey='q'),
                    html.Button('Next', id='next', accessKey='w'),
                    html.Button('Good', id='set_good'),
                    html.Button('Bad', id='set_bad'),
                    html.Button('Colocalized', id='set_colocalized'),
                    html.Button('Save Selected', id='select'),
                    dcc.ConfirmDialog(
                        id='confirm-reset',
                        message='All breakpoint will be deleted, are you sure you want to continue?'
                    ),
                    dbc.Popover(
                        [dbc.PopoverHeader("Please select channel first")],
                        id="channel-error",
                        is_open=False,
                        target="etime",
                    ),
                ],
                style={"margin-left": "60px", "padding": 5, "flex": 1.5}
            ),

            # Row 3: Trace Index and Go Control
            html.Div(
                children=[
                    dcc.Input(id="bkp", type="text", placeholder="", style={'textAlign': 'center'}, size='20'),
                    dcc.Input(id="b_bkp", type="text", placeholder="", style={'textAlign': 'center'}, size='20'),
                    dcc.Input(value=0, id="i", type='number', placeholder="", min = 0,
                              style={'textAlign': 'center', 'width': '80px'}, persistence='True'),
                    html.Button('Go', id='tr_go'),
                    dcc.Loading(
                        id="loading1", type="default",
                        children=html.Div('Total_traces: ' + str(0), id='N_traces',
                                          style={"margin-left": "10px"})
                    )
                ],
                style={'padding': 5, "margin-left": "60px", 'display': 'flex', 'flex-direction': 'row'}
            ),

            # Row 4: Breakpoints and Channel Dropdown
            html.Div(
                children=[
                    html.Button('Save breakpoints', id='save_bkps'),
                    html.Button('Load breakpoints', id='load_bkps'),
                    html.Button('Rupture', id='rupture', disabled=True),
                    html.Button('Rescale', id='rescale'),
                    dcc.Dropdown([], None, clearable=False, searchable=False,
                                 style={'width': '100px'}, id='channel')
                ],
                style={'padding': 5, "margin-left": "60px", 'display': 'flex', 'flex-direction': 'row'}
            ),

            # Row 5: Path and Load Button
            html.Div(
                children=[
                    html.Div('Path:'),
                    dcc.Input(id="path", type="text", placeholder="",
                              style={'textAlign': 'left'}, size='50', persistence='True'),
                    html.Button('Load', id='loadp')
                ],
                style={'padding': 5, "margin-left": "60px", 'display': 'flex', 'flex-direction': 'row'}
            ),

            # Row 6: Plot FRET_g Button and Status
            html.Div(
                children=[
                    html.Button('Plot FRET_g', id='plot_fret_g', n_clicks=0),
                    html.Div(id='plot_status', style={'margin-left': '10px'})
                ],
                style={'margin-left': '60px', 'padding': 5, 'display': 'flex', 'flex-direction': 'row'}
            ),

            # Row 7: Find Breakpoints (Group 0)
            html.Div(
                children=[
                    dbc.InputGroup(
                        [
                            dbc.InputGroupText("Find the"),
                            dcc.Dropdown(
                                ['first', 'second', 'previous'],
                                'previous',
                                id='chp_mode_0',
                                clearable=False,
                                searchable=False,
                                style={'textAlign': 'center', 'width': '100px'},
                                persistence='True'
                            ),
                            dbc.InputGroupText("value that is "),
                            dcc.Dropdown(
                                ['bigger', 'smaller'],
                                'smaller',
                                id='chp_comp_0',
                                clearable=False,
                                searchable=False,
                                style={'textAlign': 'center', 'width': '100px'},
                                persistence='True'
                            ),
                            dbc.InputGroupText("than "),
                            dbc.Input(
                                value=0.5, type="number",
                                id='chp_thres_0',
                                size=5,
                                step=0.05,
                                style={'textAlign': 'center', 'width': '80px'},
                                persistence='True'
                            ),
                            dbc.InputGroupText("in "),
                            dcc.Dropdown(
                                [],
                                'fret_g',
                                id='chp_channel_0',
                                clearable=False,
                                searchable=False,
                                style={'width': '100px'},
                                persistence='True',
                                persisted_props=['value', 'options']
                            ),
                            dbc.InputGroupText("for "),
                            dcc.Dropdown(
                                ['current trace', 'all traces', 'all good'],
                                'current trace',
                                id='chp_target_0',
                                clearable=False,
                                searchable=False,
                                style={'textAlign': 'center', 'width': '150px'},
                                persistence='True'
                            ),
                            dbc.Button("Find", id="chp_find_0", n_clicks=0),
                            
                        ],
                        size='small'
                    ),
                ],
                style={'padding': 5, "margin-left": "60px", 'display': 'flex', 'flex-direction': 'row', 'width': '40%'}
            ),

            # Row 8: Find Breakpoints (Group 1)
            html.Div(
                children=[
                    dbc.InputGroup(
                        [
                            dbc.InputGroupText("Find the"),
                            dcc.Dropdown(
                                ['first', 'second', 'previous'],
                                'previous',
                                id='chp_mode_1',
                                clearable=False,
                                searchable=False,
                                style={'textAlign': 'center', 'width': '100px'},
                                persistence='True'
                            ),
                            dbc.InputGroupText("value that is "),
                            dcc.Dropdown(
                                ['bigger', 'smaller'],
                                'smaller',
                                id='chp_comp_1',
                                clearable=False,
                                searchable=False,
                                style={'textAlign': 'center', 'width': '100px'},
                                persistence='True'
                            ),
                            dbc.InputGroupText("than "),
                            dbc.Input(
                                value=0.5, type="number",
                                id='chp_thres_1',
                                size=5,
                                step=0.05,
                                style={'textAlign': 'center', 'width': '80px'},
                                persistence='True',
                                persisted_props=['value', 'options']
                            ),
                            dbc.InputGroupText("in "),
                            dcc.Dropdown(
                                [],
                                'fret_b',
                                id='chp_channel_1',
                                clearable=False,
                                searchable=False,
                                style={'width': '100px'},
                                persistence='True'
                            ),
                            dbc.InputGroupText("for "),
                            dcc.Dropdown(
                                ['current trace', 'all traces', 'all good'],
                                'current trace',
                                id='chp_target_1',
                                clearable=False,
                                searchable=False,
                                style={'textAlign': 'center', 'width': '150px'},
                                persistence='True'
                            ),
                            dbc.Button("Find", id="chp_find_1", n_clicks=0),
                        ],
                        size='small'
                    )
                ],
                style={'padding': 5, "margin-left": "60px", 'display': 'flex', 'flex-direction': 'row', 'width': '40%'}
            )
        ],
        style={'width': '95%', 'height': '20%', 'padding': 5}
    )






# def tools_tab():
#     return dcc.Tab(
#         label='Tools',
#         value='Tools',
#         children=[
#             # Row 1: Trace Visibility, Smoothing, and Mode Controls
#             html.Div(
#                 children=[
#                     html.Div('Hide Trace:', style={"margin-left": "60px", "padding": 5}),
#                     html.Div(
#                         children=[
#                             dcc.Checklist(
#                                 ['BB', 'BG', 'BR', 'GG', 'GR', 'RR', 'FRET BG', 'FRET GR', 'Tot B', 'Tot G', 'HMM'],
#                                 ['Tot B', 'Tot G'],
#                                 inline=True,
#                                 inputStyle={"margin-left": "10px"},
#                                 id='show',
#                                 persistence=True
#                             )
#                         ],
#                         style={'padding': 5}
#                     ),
#                     html.Div('Smoothing', style={"margin-left": "10px"}),
#                     dcc.Input(
#                         1, type='number', min=1, step=1, id='smooth',
#                         persistence='True',
#                         style={"margin-left": "10px", "width": "50px"}
#                     ),
#                     html.Div('Scatter:', style={"margin-left": "20px", "padding": 5}),
#                     daq.ToggleSwitch(
#                         id='scatter', value=0, color='green',
#                         style={"margin-left": "10px", "padding": 5}
#                     ),
#                     html.Div('Strided Smooth:', style={"margin-left": "20px", "padding": 5}),
#                     daq.ToggleSwitch(
#                         id='strided', value=0, color='green',
#                         style={"margin-left": "10px", "padding": 5}
#                     ),
#                     html.Div(
#                         children=[
#                             dcc.RadioItems(
#                                 ['Add', 'Remove', 'Except', "Clear", "Clear All", "Set All", "Reset"],
#                                 'Add',
#                                 id='AR',
#                                 labelStyle={'display': 'inline-block', "marginTop": "5px"}
#                             )
#                         ],
#                         style={'padding': 5, "margin-left": "30%"}
#                     ),
                   
#                 ],
#                 style={'display': 'flex', 'flex-direction': 'row', 'align-items': 'center'}
#             ),

#             # Row 2: Navigation Buttons
#             html.Div(
#                 children=[
#                     html.Button('Set Dead time', id='dtime'),
#                     html.Button('Set End time', id='etime'),
#                     html.Button('Previous', id='previous', accessKey='q'),
#                     html.Button('Next', id='next', accessKey='w'),
#                     html.Button('Good', id='set_good'),
#                     html.Button('Bad', id='set_bad'),
#                     html.Button('Colocalized', id='set_colocalized'),
#                     html.Button('Save Selected', id='select'),
#                     dcc.ConfirmDialog(
#                         id='confirm-reset',
#                         message='All breakpoint will be deleted, are you sure you want to continue?'
#                     ),
#                     dbc.Popover(
#                         [dbc.PopoverHeader("Please select channel first")],
#                         id="channel-error",
#                         is_open=False,
#                         target="etime",
#                     ),
#                 ],
#                 style={"margin-left": "60px", "padding": 5, "flex": 1.5}
#             ),

#             # Row 3: Trace Index and Go Control
#             html.Div(
#                 children=[
#                     dcc.Input(id="bkp", type="text", placeholder="", style={'textAlign': 'center'}, size='20'),
#                     dcc.Input(id="b_bkp", type="text", placeholder="", style={'textAlign': 'center'}, size='20'),
#                     dcc.Input(value=0, id="i", type='number', placeholder="", min = 0,
#                               style={'textAlign': 'center', 'width': '80px'}, persistence='True'),
#                     html.Button('Go', id='tr_go'),
#                     dcc.Loading(
#                         id="loading1", type="default",
#                         children=html.Div('Total_traces: ' + str(0), id='N_traces',
#                                           style={"margin-left": "10px"})
#                     )
#                 ],
#                 style={'padding': 5, "margin-left": "60px", 'display': 'flex', 'flex-direction': 'row'}
#             ),

#             # Row 4: Breakpoints and Channel Dropdown
#             html.Div(
#                 children=[
#                     html.Button('Save breakpoints', id='save_bkps'),
#                     html.Button('Load breakpoints', id='load_bkps'),
#                     html.Button('Rupture', id='rupture', disabled=True),
#                     html.Button('Rescale', id='rescale'),
#                     dcc.Dropdown([], None, clearable=False, searchable=False,
#                                  style={'width': '100px'}, id='channel')
#                 ],
#                 style={'padding': 5, "margin-left": "60px", 'display': 'flex', 'flex-direction': 'row'}
#             ),

    #         # Row 5: Path and Load Button
    #         html.Div(
    #             children=[
    #                 html.Div('Path:'),
    #                 dcc.Input(id="path", type="text", placeholder="",
    #                           style={'textAlign': 'left'}, size='50', persistence='True'),
    #                 html.Button('Load', id='loadp')
    #             ],
    #             style={'padding': 5, "margin-left": "60px", 'display': 'flex', 'flex-direction': 'row'}
    #         ),

    #         # Row 6: Plot FRET_g Button and Status
    #         html.Div(
    #             children=[
    #                 html.Button('Plot FRET_g', id='plot_fret_g', n_clicks=0),
    #                 html.Div(id='plot_status', style={'margin-left': '10px'})
    #             ],
    #             style={'margin-left': '60px', 'padding': 5, 'display': 'flex', 'flex-direction': 'row'}
    #         ),

    #         # Row 7: Find Breakpoints (Group 0)
    #         html.Div(
    #             children=[
    #                 dbc.InputGroup(
    #                     [
    #                         dbc.InputGroupText("Find the"),
    #                         dcc.Dropdown(
    #                             ['first', 'second', 'previous'],
    #                             'previous',
    #                             id='chp_mode_0',
    #                             clearable=False,
    #                             searchable=False,
    #                             style={'textAlign': 'center', 'width': '100px'},
    #                             persistence='True'
    #                         ),
    #                         dbc.InputGroupText("value that is "),
    #                         dcc.Dropdown(
    #                             ['bigger', 'smaller'],
    #                             'smaller',
    #                             id='chp_comp_0',
    #                             clearable=False,
    #                             searchable=False,
    #                             style={'textAlign': 'center', 'width': '100px'},
    #                             persistence='True'
    #                         ),
    #                         dbc.InputGroupText("than "),
    #                         dbc.Input(
    #                             value=0.5, type="number",
    #                             id='chp_thres_0',
    #                             size=5,
    #                             step=0.05,
    #                             style={'textAlign': 'center', 'width': '80px'},
    #                             persistence='True'
    #                         ),
    #                         dbc.InputGroupText("in "),
    #                         dcc.Dropdown(
    #                             [],
    #                             'fret_g',
    #                             id='chp_channel_0',
    #                             clearable=False,
    #                             searchable=False,
    #                             style={'width': '100px'},
    #                             persistence='True',
    #                             persisted_props=['value', 'options']
    #                         ),
    #                         dbc.InputGroupText("for "),
    #                         dcc.Dropdown(
    #                             ['current trace', 'all traces', 'all good'],
    #                             'current trace',
    #                             id='chp_target_0',
    #                             clearable=False,
    #                             searchable=False,
    #                             style={'textAlign': 'center', 'width': '150px'},
    #                             persistence='True'
    #                         ),
    #                         dbc.Button("Find", id="chp_find_0", n_clicks=0),
                            
    #                     ],
    #                     size='small'
    #                 ),
    #             ],
    #             style={'padding': 5, "margin-left": "60px", 'display': 'flex', 'flex-direction': 'row', 'width': '40%'}
    #         ),

    #         # Row 8: Find Breakpoints (Group 1)
    #         html.Div(
    #             children=[
    #                 dbc.InputGroup(
    #                     [
    #                         dbc.InputGroupText("Find the"),
    #                         dcc.Dropdown(
    #                             ['first', 'second', 'previous'],
    #                             'previous',
    #                             id='chp_mode_1',
    #                             clearable=False,
    #                             searchable=False,
    #                             style={'textAlign': 'center', 'width': '100px'},
    #                             persistence='True'
    #                         ),
    #                         dbc.InputGroupText("value that is "),
    #                         dcc.Dropdown(
    #                             ['bigger', 'smaller'],
    #                             'smaller',
    #                             id='chp_comp_1',
    #                             clearable=False,
    #                             searchable=False,
    #                             style={'textAlign': 'center', 'width': '100px'},
    #                             persistence='True'
    #                         ),
    #                         dbc.InputGroupText("than "),
    #                         dbc.Input(
    #                             value=0.5, type="number",
    #                             id='chp_thres_1',
    #                             size=5,
    #                             step=0.05,
    #                             style={'textAlign': 'center', 'width': '80px'},
    #                             persistence='True',
    #                             persisted_props=['value', 'options']
    #                         ),
    #                         dbc.InputGroupText("in "),
    #                         dcc.Dropdown(
    #                             [],
    #                             'fret_b',
    #                             id='chp_channel_1',
    #                             clearable=False,
    #                             searchable=False,
    #                             style={'width': '100px'},
    #                             persistence='True'
    #                         ),
    #                         dbc.InputGroupText("for "),
    #                         dcc.Dropdown(
    #                             ['current trace', 'all traces', 'all good'],
    #                             'current trace',
    #                             id='chp_target_1',
    #                             clearable=False,
    #                             searchable=False,
    #                             style={'textAlign': 'center', 'width': '150px'},
    #                             persistence='True'
    #                         ),
    #                         dbc.Button("Find", id="chp_find_1", n_clicks=0),
    #                     ],
    #                     size='small'
    #                 )
    #             ],
    #             style={'padding': 5, "margin-left": "60px", 'display': 'flex', 'flex-direction': 'row', 'width': '40%'}
    #         )
    #     ],
    #     style={'width': '95%', 'height': '20%', 'padding': 5}
    # )
