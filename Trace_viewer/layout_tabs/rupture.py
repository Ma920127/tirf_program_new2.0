from dash import dcc, html
import dash_bootstrap_components as dbc

def rupture_tab():
    return dcc.Tab(label='Rupture', value='Rupture', children=[
        html.Div([
            dbc.Row([
                
                # ==========================================
                # LEFT COLUMN: RUPTURE SETTINGS & BUTTONS
                # ==========================================
                dbc.Col([
                    html.H5("Rupture Parameters", className="mb-3 font-weight-bold"),
                    
                    # ROW 1: Target Channel and Penalty
                    dbc.Row([
                        dbc.Col([
                            dbc.Label("Target Channel"),
                            dcc.Dropdown(
                                id='rup-channel',
                                options=[{'label': 'FRET_g', 'value': 'fret_g'}, 
                                         {'label': 'FRET_b', 'value': 'fret_b'},
                                         {'label': 'RR', 'value': 'r'},
                                         {'label': 'GG', 'value': 'g'},
                                         {'label': 'BB', 'value': 'b'}
                                         ], 
                                value='fret_g',
                                clearable=False
                            )
                        ], width=5), 
                        
                        dbc.Col([
                            dbc.Label("Penalty (λ)"),
                            dcc.Input(
                                id='rup-penalty', 
                                type='number', 
                                value=50000, 
                                className="form-control" 
                            )
                        ], width=5),
                    ], className="mb-3"), 

                    # ROW 2: Direction and Min Gap
                    dbc.Row([
                        dbc.Col([
                            dbc.Label("Direction Filter", className="d-block"), 
                            dcc.RadioItems(
                                id='rup-direction',
                                options=[
                                    {'label': ' Both ', 'value': 'both'},
                                    {'label': ' Upward ', 'value': 'up'},
                                    {'label': ' Downward ', 'value': 'down'}
                                ],
                                value='both',
                                inline=True,
                                inputStyle={"margin-right": "5px", "margin-left": "10px"}
                            )
                        ], width=7),
                        
                        dbc.Col([
                            dbc.Label("Min Gap (frames)"),
                            dcc.Input(
                                id='rup-mingap', 
                                type='number', 
                                value=0, 
                                min=0, 
                                step=1,
                                className="form-control"
                            )
                        ], width=4), 
                    ], className="mb-4"),

                    html.Hr(className="my-4"), # Subtle horizontal line before buttons

                    # ROW 3: Action Buttons (Detect / Fit All)
                    dbc.Row([
                        dbc.Col([
                            html.Button(
                                "DETECT CURRENT", 
                                id="btn-run-rupture", 
                                className="btn btn-primary w-100", 
                                style={"color": "white", "fontWeight": "bold"}
                            ),
                        ], width=4), 

                        dbc.Col([
                            html.Button(
                                "FIT ALL TRACES", 
                                id="btn-rup-fit-all", 
                                className="btn btn-success w-100", 
                                style={"color": "white", "fontWeight": "bold"}
                            ),
                        ], width=4),
                    ], className="mb-2"),

                    # ROW 4: Bottom Buttons (Clear)
                    dbc.Row([
                        dbc.Col([
                            html.Button(
                                "CLEAR", 
                                id="btn-rup-clear", 
                                className="btn btn-warning w-100", 
                                style={"color": "white", "fontWeight": "bold"}
                            )
                        ], width=4),
                        
                        dbc.Col([
                            html.Button(
                                "CLEAR ALL", 
                                id="btn-rup-clear-all", 
                                className="btn btn-danger w-100", 
                                style={"color": "white", "fontWeight": "bold"}
                            )
                        ], width=4),
                    ], className="mb-3"), 

                    # ROW 5: Save Button
                    dbc.Row([
                        dbc.Col([
                            html.Button(
                                "SAVE PREDICTIONS & SEND TO TOOLS TAB", 
                                id="btn-rup-save", 
                                className="btn btn-success w-100", 
                                style={"color": "white", "fontWeight": "bold"}
                            )
                        ], width=8), 
                    ], className="mb-3"),

                    # Status Output Text
                    dbc.Row([
                        dbc.Col([
                            html.Div(id='rup-status-output', className="mt-2 text-danger font-weight-bold")
                        ], width=12)
                    ])
                ], width=7, style={"paddingRight": "30px"}), 


                # ==========================================
                # RIGHT COLUMN: CENSOR DETECTION
                # ==========================================
                dbc.Col([
                    html.H5("Censor Detection", className="mb-3 font-weight-bold", style={"color": "#4a4a4a"}),
                    
                    dbc.Row([
                        dbc.Col([
                            dbc.Label("Censor Mode (If no rupture found)"),
                            dcc.Dropdown(
                                id='rup-censor-mode',
                                options=[
                                    {'label': 'None', 'value': 'none'},
                                    {'label': 'Right Censor (Last Index)', 'value': 'right'},
                                    {'label': 'Left Censor (First Index)', 'value': 'left'},
                                    {'label': 'Both', 'value': 'both'}
                                ],
                                value='none',
                                clearable=False
                            )
                        ], width=12)
                    ], className="mb-3"),
                    
                    dbc.Row([
                        dbc.Col([
                            dbc.Label("Upper Intensity Threshold"),
                            dcc.Input(
                                id='rup-upper-thresh', 
                                type='number', 
                                value=2000, 
                                step=0.01,
                                className="form-control"
                            )
                        ], width=12)
                    ], className="mb-3"),

                    dbc.Row([
                        dbc.Col([
                            dbc.Label("Lower Intensity Threshold"),
                            dcc.Input(
                                id='rup-lower-thresh', 
                                type='number', 
                                value=1000, 
                                step=0.01,
                                className="form-control"
                            )
                        ], width=12)
                    ], className="mb-3"),

                    # Helpful text box to explain the feature
                    html.Div([
                        html.Small(
                            "If Rupture detection yields no breakpoints, the algorithm checks the overall trace mean against these thresholds. If triggered, it sets a fallback breakpoint at the start (Left) or end (Right) of the trace.",
                            className="text-muted"
                        )
                    ], className="mt-4 p-3 bg-light border rounded")

                ], width=5, style={"borderLeft": "2px solid #eaeaea", "paddingLeft": "30px"}) 

            ]) 
        ], style={"padding": "30px"}) 
    ])