from dash import dcc, html
import dash_daq as daq
from dash_extensions.enrich import dash_table
import dash_bootstrap_components as dbc


def hmm_tab():
    return dcc.Tab(
        label='HMM',
        value='HMM',
        children=[
            html.Div([
                dbc.Row([

                    # ══════════════════════════════════════════════
                    # LEFT COLUMN — Select channel and Photobleach Filter
                    # ══════════════════════════════════════════════
                    dbc.Col([
                        # ── Channel selector ──────────────────────
                        html.H5("Target Channel", className="mb-3 font-weight-bold"),
                        dbc.Row([
                            dcc.Dropdown(
                                id='hmm-channel',
                                options=[
                                    {'label': 'fret_g  (GR FRET)', 'value': 'fret_g'},
                                    {'label': 'fret_b  (BG FRET)', 'value': 'fret_b'},
                                    {'label': 'GG',                'value': 'gg'},
                                    {'label': 'BB',                'value': 'bb'},
                                    {'label': 'RR',                'value': 'rr'},
                                ],
                                value='fret_g',
                                clearable=False,
                                searchable=False
                            )
                        ], className="mb-3"),

                        html.Hr(className="my-3"),

                        # ── Selected filter toggle ─────────────────
                        dbc.Row([
                            dbc.Col(
                                html.H5("Selected Filter", className="mb-0 font-weight-bold"),
                                width="auto"
                            ),
                            dbc.Col([
                                html.Div([
                                    daq.ToggleSwitch(
                                        id='hmm-selected-filter',
                                        value=True,
                                        color='#4C72B0',
                                        style={'display': 'inline-block'},
                                    ),
                                    html.Span(
                                        id='hmm-selected-filter-label',
                                        children='ON',
                                        style={
                                            'marginLeft': '8px',
                                            'fontWeight': 'bold',
                                            'fontSize': '13px',
                                            'color': '#4C72B0',
                                            'verticalAlign': 'middle',
                                        }
                                    ),
                                ], style={'display': 'flex', 'alignItems': 'center', 'marginTop': '4px'}),
                            ], width="auto"),
                        ], className="mb-2 align-items-center"),
                        html.Small(
                            "ON: only selected (good) traces  |  OFF: all non-bad traces",
                            className="text-muted d-block mb-3",
                            style={'fontSize': '11px'},
                        ),

                        html.Hr(className="my-3"),

                        # ── Enable toggle ─────────────────────────
                        dbc.Row([
                            dbc.Col(
                                html.H5("Photobleach Filter", className="mb-0 font-weight-bold"),
                                width="auto"
                            ),
                            dbc.Col([
                                html.Div([
                                    daq.ToggleSwitch(
                                        id='hmm-pb-enable',
                                        value=False,
                                        color='#55A868',
                                        style={'display': 'inline-block'},
                                    ),
                                    html.Span(
                                        id='hmm-pb-toggle-label',
                                        children='OFF',
                                        style={
                                            'marginLeft': '8px',
                                            'fontWeight': 'bold',
                                            'fontSize': '13px',
                                            'color': '#868e96',
                                            'verticalAlign': 'middle',
                                        }
                                    ),
                                ], style={'display': 'flex', 'alignItems': 'center', 'marginTop': '4px'}),
                            ], width="auto"),
                        ], className="mb-3 align-items-center"),

                        # ── fret_g panel ──────────────────────────
                        html.Div(id='hmm-pb-fretg-panel', style={'display': 'none'}, children=[
                            dbc.Row([
                                # First Column
                                dbc.Col([
                                    # Wrapped in a Div to match the height and alignment of the second column
                                    html.Div([
                                        dbc.Label("Min total_g intensity", style={'marginBottom': '0'})
                                    ], style={'display': 'flex', 'alignItems': 'center', 'height': '32px', 'marginBottom': '8px'}),
                                    
                                    dbc.Input(id='hmm-pb-ming-totg', type='number',
                                              value=50, min=0, step=1, placeholder='e.g. 50'),
                                ], width=4),

                                # Second Column
                                dbc.Col([
                                    # This container now matches the one in Column 1 perfectly
                                    html.Div([
                                        dbc.Label("Use RR filter", style={'marginRight': '10px', 'marginBottom': '0'}),
                                        daq.ToggleSwitch(
                                            id='hmm-pb-use-rr', 
                                            value=False,
                                            color='#4C72B0',
                                            style={'display': 'inline-block'} 
                                        ),
                                    ], style={'display': 'flex', 'alignItems': 'center', 'height': '32px', 'marginBottom': '8px'}),
                                    
                                    dbc.Input(id='hmm-pb-ming-rr', type='number',
                                              value=50, min=0, step=1,
                                              placeholder='Min RR', disabled=True),
                                ], width=4),
                                
                            ], className="mb-2", align="start"),
                        ]),

                        # ── fret_b panel ──────────────────────────
                        html.Div(id='hmm-pb-fretb-panel', style={'display': 'none'}, children=[
                            dbc.Row([
                                # First Column
                                dbc.Col([
                                    # Wrapped in a Div to match the height and alignment of the second column
                                    html.Div([
                                        dbc.Label("Min total_bg intensity", style={'marginBottom': '0'})
                                    ], style={'display': 'flex', 'alignItems': 'center', 'height': '32px', 'marginBottom': '8px'}),
                                    
                                    dbc.Input(id='hmm-pb-minb-totb', type='number',
                                              value=50, min=0, step=1, placeholder='e.g. 50'),
                                ], width=4),

                                # Second Column
                                dbc.Col([
                                    # This container now matches the one in Column 1 perfectly
                                    html.Div([
                                        dbc.Label("Use GG filter", style={'marginRight': '10px', 'marginBottom': '0'}),
                                        daq.ToggleSwitch(
                                            id='hmm-pb-use-rr-b', 
                                            value=False,
                                            color='#4C72B0',
                                            style={'display': 'inline-block'} 
                                        ),
                                    ], style={'display': 'flex', 'alignItems': 'center', 'height': '32px', 'marginBottom': '8px'}),
                                    
                                    dbc.Input(id='hmm-pb-minb-rr', type='number',
                                              value=50, min=0, step=1,
                                              placeholder='Min GG', disabled=True),
                                ], width=4),
                                
                            ], className="mb-2", align="start"),
                        ]),

                        # ── GG / BB / RR single panel ─────────────
                        html.Div(id='hmm-pb-single-panel', style={'display': 'none'}, children=[
                            dbc.Row([
                                dbc.Col([
                                    dbc.Label("Min intensity"),
                                    dbc.Input(id='hmm-pb-min-single', type='number',
                                              value=50, min=0, step=1, placeholder='e.g. 50'),
                                ], width=4),
                            ], className="mb-2"),
                        ]),

                        html.Hr(className="my-3"),

                        # ── Apply / Clear ─────────────────────────
                        dbc.Row([
                            dbc.Col([
                                html.Button("▶ Apply Filter", id='hmm-pb-apply', n_clicks=0,
                                            className="btn btn-primary w-100",
                                            style={"fontWeight": "bold"}),
                            ], width=6),
                            dbc.Col([
                                html.Button("✕ Clear Filter", id='hmm-pb-clear', n_clicks=0,
                                            className="btn btn-danger w-100",
                                            style={"fontWeight": "bold"}),
                            ], width=6),
                        ], className="mb-2"),

                        dbc.Row([
                            dbc.Col([
                                html.Div(id='hmm-pb-status',
                                         className="mt-2 text-muted",
                                         style={'fontSize': '12px', 'fontStyle': 'italic'}),
                            ], width=12),
                        ]),

                    ], width=3, style={"paddingRight": "24px"}),

                    # ══════════════════════════════════════════════
                    # MIDDLE COLUMN — HMM Prediction
                    # ══════════════════════════════════════════════
                    dbc.Col([
                        html.H5("HMM Prediction", className="mb-3 font-weight-bold"),
                        # ── Train / Fix / Plot / State-reverse toggles ───────
                        dbc.Row([
                            # TRAIN TOGGLE  (id kept as hmm_fit for back-compat)
                            dbc.Col([
                                dbc.Label("Train", style={'marginRight': '8px', 'marginBottom': '0'}),
                                daq.ToggleSwitch(
                                    id='hmm_fit',
                                    value=True,
                                    color='#55A868',
                                    style={'display': 'inline-block'},
                                    size=36
                                ),
                            ], style={'display': 'flex', 'alignItems': 'center'}, width="auto"),

                            # STATE REVERSE TOGGLE
                            dbc.Col([
                                dbc.Label("State reverse", style={'marginRight': '8px', 'marginBottom': '0'}),
                                daq.ToggleSwitch(
                                    id='hmm_reverse',
                                    value=False,
                                    color='#C44E52',
                                    style={'display': 'inline-block'},
                                    size=36
                                ),
                            ], style={'display': 'flex', 'alignItems': 'center'}, width="auto"),

                            # STICKY WEIGHT INPUT
                            dbc.Col([
                                dbc.Label("Sticky W", style={'marginRight': '8px', 'marginBottom': '0'}),
                                dbc.Input(
                                    id='hmm_sticky_w', type='number', value=1, min=0, step=1,
                                    style={'width': '65px', 'height': '32px', 'padding': '4px 8px'}
                                ),
                            ], style={'display': 'flex', 'alignItems': 'center'}, width="auto"),

                            # N_ITER INPUT
                            dbc.Col([
                                dbc.Label("n_iter", style={'marginRight': '8px', 'marginBottom': '0'}),
                                dbc.Input(
                                    id='hmm_niter', type='number', value=30, min=1, step=1,
                                    style={'width': '65px', 'height': '32px', 'padding': '4px 8px'}
                                ),
                            ], style={'display': 'flex', 'alignItems': 'center'}, width="auto"),

                            # EPOCH INPUT
                            dbc.Col([
                                dbc.Label("Epoch", style={'marginRight': '8px', 'marginBottom': '0'}),
                                dbc.Input(
                                    id='hmm_epoch', type='number', value=20, min=1, step=1,
                                    style={'width': '65px', 'height': '32px', 'padding': '4px 8px'}
                                ),
                            ], style={'display': 'flex', 'alignItems': 'center'}, width="auto"),

                        ], className="mb-3 g-3 justify-content-start"),
                        
                        # ── Covariance type ───────────────────────
                        dbc.Row([
                            html.H5("Covariance type", className="mb-3 font-weight-bold"),
                            dbc.Col([
                                dcc.RadioItems(
                                    id='hmm_cov_type',
                                    options=['spherical', 'diag', 'full', 'tied'],
                                    value='spherical',
                                    inline=True,
                                    inputStyle={"marginRight": "4px", "marginLeft": "10px"},
                                    style={'fontSize': '20px'},
                                ),
                            ], width=15),
                        ], className="mb-3"),

                        # ── Initial means table ───────────────────────
                        # Enter expected FRET/intensity values; leave unused
                        # cells at −1.  Number of valid entries = k (states).
                        # Values seed EM starting means; EM refines from there.
                        dbc.Row([
                            dbc.Col([
                                html.H5("Initial means  (−1 to skip)", className="mb-3 font-weight-bold"),
                                dash_table.DataTable(
                                    id='hmm_means',
                                    columns=[{'id': str(p), 'name': str(p)} for p in range(10)],
                                    data=[
                                        {str(p): -1 for p in range(10)},
                                    ],
                                    style_table={'width': '100%', 'overflowX': 'auto'},
                                    style_cell={
                                        'minWidth': '36px', 'width': '36px', 'maxWidth': '36px',
                                        'textAlign': 'center', 'fontSize': '20px', 'padding': '4px',
                                    },
                                    style_header={
                                        'backgroundColor': '#e9ecef',
                                        'fontWeight': 'bold', 'fontSize': '20px',
                                    },
                                    editable=True,
                                    persistence=True,
                                    persisted_props=['data'],
                                ),
                            ], width=12),
                        ], className="mb-3"),

                        html.Hr(className="my-3"),

                        # ── Start button ──────────────────────────
                        dbc.Row([
                            dbc.Col([
                                html.Button("▶ Start HMM", id='hmm_start', n_clicks=0,
                                            className="btn btn-success w-100",
                                            style={"fontWeight": "bold", "fontSize": "18px"}),
                            ], width=6),
                            dbc.Col([
                                html.Button("✕ HMM Clear", id='hmm_clear', n_clicks=0,
                                            className="btn btn-danger w-100",
                                            style={"fontWeight": "bold", "fontSize": "18px"}),
                            ], width= 6 ),
                        ], className="mb-2"),

                        dbc.Row([
                            dbc.Col([
                                html.Button(
                                    "💾  Save hmm.npz",
                                    id='hmm-save-npz',
                                    n_clicks=0,
                                    className="btn btn-primary w-100",
                                    style={"fontWeight": "bold"},
                                ),
                            ], width=6),
                            dbc.Col([
                                html.Button(
                                    "📤 Load hmm.npz",
                                    id='hmm-load-npz',
                                    n_clicks=0,
                                    className="btn btn-warning w-100",
                                    style={"fontWeight": "bold"},
                                ),
                            ], width=6)
                        ], className="mb-3"),
                        dbc.Row([
                            dbc.Col([
                                html.Div(id='hmm-fit-status',
                                         className="mt-2 text-muted",
                                         style={'fontSize': '12px', 'fontStyle': 'italic'}),
                            ], width=12),
                        ]),

                        html.Hr(className="my-3"),

                        # ── Fitted states summary + Merge Controls ────────────
                        # Row 1: title + merge controls inline
                        html.Div([
                            html.H5("Current Fitted States",
                                    className="mb-0 font-weight-bold"),
                            html.Div([
                                dbc.Label(
                                    "Merge threshold",
                                    style={'marginRight': '8px', 'marginBottom': '0'},
                                ),
                                dbc.Input(
                                    id='hmm-merge-threshold',
                                    type='number',
                                    value=0.05, min=0, step='any',
                                    style={'width': '80px', 'height': '32px',
                                           'padding': '4px 8px',
                                           'marginRight': '8px'},
                                ),
                                html.Button(
                                    "🔀 Merge States",
                                    id='hmm-merge-btn',
                                    n_clicks=0,
                                    className="btn btn-secondary",
                                    style={"fontWeight": "bold",
                                           "height": "32px",
                                           "padding": "0 12px"},
                                ),
                            ], style={'display': 'flex', 'alignItems': 'center',
                                      'marginLeft': 'auto'}),
                        ], style={'display': 'flex', 'alignItems': 'center',
                                  'marginBottom': '8px'}),
                        # Row 2: table full width
                        dbc.Row([
                            dbc.Col([
                                dash_table.DataTable(
                                    id='hmm-fitted-states-table',
                                    columns=[],
                                    data=[],
                                    editable=False,
                                    style_table={'width': '100%', 'overflowX': 'auto'},
                                    style_cell={
                                        'minWidth':   '36px',
                                        'width':      '36px',
                                        'maxWidth':   '36px',
                                        'textAlign':  'center',
                                        'fontSize':   '13px',
                                        'padding':    '4px',
                                        'fontFamily': 'Arial, sans-serif',
                                    },
                                    style_header={
                                        'fontWeight': 'bold',
                                        'fontSize':   '13px',
                                        'color':      'white',
                                    },
                                    # Header cell colour matches the STATE_COLORS used in the bar
                                    style_header_conditional=[
                                        {
                                            'if': {'column_id': f'S{i}'},
                                            'backgroundColor': color,
                                        }
                                        for i, color in enumerate([
                                            '#4C72B0', '#DD8452', '#55A868', '#C44E52',
                                            '#8172B2', '#937860', '#DA8BC3', '#8C8C8C',
                                            '#CCB974', '#64B5CD',
                                        ])
                                    ],
                                ),
                            ], width=12),
                        ], className="mb-2"),
                        dbc.Row([
                            dbc.Col([
                                html.Div(
                                    id='hmm-merge-status',
                                    className="text-muted",
                                    style={'fontSize': '12px', 'fontStyle': 'italic'},
                                ),
                            ], width=12),
                        ]),

                    ], width=6, style={
                        "borderLeft":  "2px solid #eaeaea",
                        "borderRight": "2px solid #eaeaea",
                        "paddingLeft":  "24px",
                        "paddingRight": "24px",
                    }),


                    # ══════════════════════════════════════════════
                    # RIGHT COLUMN — Amend & Dwell Save
                    # ══════════════════════════════════════════════
                    dbc.Col([ # <-- ADDED MISSING OPENING BRACKET HERE
                        html.H5("HMM Tool", className="mb-3 font-weight-bold"),
                        
                        # ── Amend placeholder ─────────────────────
                        dbc.Row([
                            dbc.Col([
                                dcc.RadioItems(
                                    id='hmm_tool',
                                    options=['Boundary move', 'Add', 'Delete', 'Change state'],
                                    value='Boundary move', # Changed from 'spherical' to match options
                                    inline=True,
                                    inputStyle={"marginRight": "4px", "marginLeft": "10px"},
                                    style={'fontSize': '20px'},
                                ),
                            ], width=12)
                        ], className="mb-4"),

                        html.Hr(className="my-3"),

                        dbc.Row([
                            dbc.Col([
                                html.H5("Dwell Manager", className="mb-0 font-weight-bold"),
                            ], width=7, style={'display': 'flex', 'alignItems': 'center'}),
                            dbc.Col([
                                html.Div([
                                    dbc.Label(
                                        "Max Censor",
                                        style={'marginRight': '8px', 'marginBottom': '0'},
                                    ),
                                    daq.ToggleSwitch(
                                        id='dwell-censor-last',
                                        value=False,
                                        color='#4C72B0',
                                        style={'display': 'inline-block'},
                                    ),
                                    html.Span(
                                        id='dwell-censor-last-label',
                                        children='OFF',
                                        style={
                                            'marginLeft':    '8px',
                                            'fontWeight':    'bold',
                                            'fontSize':      '13px',
                                            'color':         '#868e96',
                                            'verticalAlign': 'middle',
                                        }
                                    ),
                                ], style={'display': 'flex', 'alignItems': 'center'}),
                            ], width=5, style={'paddingLeft': '12px'}),
                        ], className="mb-2 align-items-center"),

                        dbc.Row([

                            # ── Left boundary ─────────────────────
                            dbc.Col([
                                html.Div("Left boundary",
                                         style={'fontSize': '18px', 'fontWeight': 'bold',
                                                'marginBottom': '8px'}),
                                dcc.RadioItems(
                                    id='dwell-left-mode',
                                    options=[
                                        {'label': ' Cut states',    'value': 'cut'},
                                        {'label': ' Dead time (s)', 'value': 'dead'},
                                    ],
                                    value='cut',
                                    inline=True,
                                    inputStyle={'marginRight': '4px', 'marginLeft': '10px'},
                                    style={'fontSize': '20px', 'marginBottom': '6px'},
                                ),
                                html.Div([
                                    dbc.Input(
                                        id='dwell-left-cut', type='number', value=0, min=0, step=1,
                                        debounce=True,
                                        disabled=False,
                                        style={'width': '65px', 'height': '32px', 'padding': '4px 8px'},
                                    ),
                                    dbc.Input(
                                        id='dwell-left-deadtime', type='number', value=0, min=0, step=1,
                                        debounce=True,
                                        disabled=True,
                                        style={'width': '65px', 'height': '32px', 'padding': '4px 8px',
                                               'marginLeft': '55px'},
                                    ),
                                ], style={'display': 'flex', 'alignItems': 'center',
                                          'marginLeft': '14px'}),
                            ], width=7),

                            # ── Right boundary ────────────────────
                            dbc.Col([
                                html.Div("Right boundary",
                                         style={'fontSize': '18px', 'fontWeight': 'bold',
                                                'marginBottom': '8px'}),
                                html.Span('Cut states',
                                          style={'fontSize': '20px', 'display': 'block',
                                                 'marginBottom': '6px', 'marginLeft': '10px'}),
                                html.Div([
                                    dbc.Input(
                                        id='dwell-right-cut', type='number', value=0, min=0, step=1,
                                        debounce=True,
                                        style={'width': '65px', 'height': '32px', 'padding': '4px 8px'},
                                    ),
                                ], style={'marginLeft': '14px'}),
                            ], width=5, style={'borderLeft': '1px dashed #aaa',
                                               'paddingLeft': '12px'}),

                        ], className='mb-4'),

                        # ── Save dwell times ──────────────────────
                        dbc.Row([
                            dbc.Col([
                                html.Button(
                                    "💾  Save Dwell.npz",
                                    id='hmm-save-dwell',
                                    n_clicks=0,
                                    className="btn btn-primary w-100",
                                    style={"fontWeight": "bold"},
                                ),
                            ], width=12),
                        ], className="mb-2"),

                        # ── Save status ───────────────────────────
                        dbc.Row([
                            dbc.Col([
                                html.Div(
                                    id='hmm-save-status',
                                    className="mt-2 text-muted",
                                    style={'fontSize': '12px', 'fontStyle': 'italic'},
                                ),
                            ], width=12),
                        ]),

                        html.Div([
                            html.Small(
                                "Dwell times are computed from the HMM state-bar segments. "
                                "hmm.npz saves the full predicted state sequence for all traces.",
                                className="text-muted",
                            )
                        ], className="mt-4 p-3 bg-light border rounded"),
                    ], # <-- CLOSED BRACKET FOR RIGHT COLUMN HERE
                    width=3, 
                    style={"paddingLeft": "24px"})
                ]) # Closes dbc.Row
            ]) # Closes html.Div
        ] # Closes dcc.Tab children list
    ) # Closes dcc.Tab