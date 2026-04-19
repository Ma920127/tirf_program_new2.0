from dash import dcc, html
import dash_bootstrap_components as dbc

def rupture_tab():
    return dcc.Tab(label='Rupture', value='Rupture', children=[
        html.Div([
    # ROW 1: Target Channel and Penalty side-by-side
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
        ], width=3), # width=3 makes the box short!
        
        dbc.Col([
            dbc.Label("Penalty (λ)"),
            dcc.Input(
                id='rup-penalty', 
                type='number', 
                value=50000, 
                className="form-control" # Uses standard bootstrap styling
            )
        ], width=3),
    ], className="mb-3"), # mb-3 adds a little margin at the bottom of the row

    # ROW 2: Direction and Min Gap side-by-side
    dbc.Row([
        dbc.Col([
            dbc.Label("Direction Filter", className="d-block"), # d-block forces label to its own line
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
        ], width=4),
        
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
        ], width=2), # Super short box for a simple number
    ], className="mb-4"),
    dbc.Row([
        dbc.Col([
            html.Button(
                "DETECT CURRENT", 
                id="btn-run-rupture", 
                className="btn btn-primary w-100", 
                style={"color": "white", "fontWeight": "bold"}
            ),
        ], width=2), 

        dbc.Col([
            html.Button(
                "FIT ALL TRACES", 
                id="btn-rup-fit-all", 
                className="btn btn-success w-100", 
                style={"color": "white", "fontWeight": "bold"}
            ),
        ], width=2),
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
        ], width=2),
        
        dbc.Col([
            html.Button(
                "CLEAR ALL", 
                id="btn-rup-clear-all", 
                className="btn btn-danger w-100", 
                style={"color": "white", "fontWeight": "bold"}
            )
        ], width=2),
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
        ], width=4), 
    ], className="mb-3"),
    dbc.Row([
        dbc.Col([
            html.Div(id='rup-status-output', className="mt-2 text-danger font-weight-bold")
        ], width=6)
    ])
], style={"padding": "20px"}) # Adds breathing room around the edges
    ])