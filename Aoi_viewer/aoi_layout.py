from dash import dcc, html
import dash_bootstrap_components as dbc
from aoi_utils import load_config
from layout.blob_tab import get_blob_tab
from layout.fret_tab import get_fret_tab


def make_layout(fig):
    config = load_config(1, "1024")
    layout = html.Div([
        # Hidden store to hold all extra state; initial state can be adjusted as needed.
        dcc.Store(
            id="state-store",
            storage_type="memory",
            data={
                "coord_list": [],
                "org_size": 1,
            }
        ),
        dbc.Row([
            dbc.Col([
            # ONE master wrapper for Tabs -> Graph -> Slider
            html.Div([
                
                # 1. The Tabs
                dcc.Tabs(
                    id="graph-size-tabs",
                    value="1024",
                    children=[
                        dcc.Tab(label="1024 x 1024", value="1024"),
                        dcc.Tab(label="512 x 512", value="512"),
                    ],
                    style={'width': '800px', 'padding': 5}
                ),
                
                # 2. The Graph
                dcc.Graph(
                    id="graph",
                    figure=fig,
                    config={'scrollZoom': True, 'modebar_remove': ['box select', 'lasso select']},
                    style={'width': '1000px', 'height': '1000px'} # Locked size
                ),
                
                # 3. The Slider and Input (Moved inside the same vertical wrapper)
                html.Div([
                    html.Div(
                        dcc.Slider(
                            0, 0, 1,
                            value=0,
                            updatemode='drag',
                            tooltip={"placement": "top", "always_visible": True},
                            marks=None,
                            id='frame_slider'
                        ),
                        style={'flex': '1', 'paddingBottom': '15px'} # Lets the slider take up the remaining width nicely
                    ),
                    dcc.Input(
                        value=0, id="anchor", type="text",
                        style={'textAlign': 'center', 'marginLeft': '10px', 'paddingBottom': '15px'},
                        size='3', debounce=True
                    )
                ], style={'width': '800px', 'marginTop': '15px', 'display': 'flex', 'flexDirection': 'row', 'alignItems': 'center'})
                
            ], style={'display': 'flex', 'flexDirection': 'column', 'alignItems': 'center'}),
            ]),
            dbc.Col([
                dcc.Tabs(
                    id='tabs-example-1',
                    value='tab-1',
                    children=[
                        dcc.Tab(label='Blob', children=get_blob_tab(config)),
                        dcc.Tab(label='FRET', children=get_fret_tab(config))
                    ],
                    style={'width': '600px', 'padding': 5}
                ),
                html.Div([
                    dbc.RadioItems(
                        id="configs",
                        className="btn-group",
                        inputClassName="btn-check",
                        labelClassName="btn btn-outline-primary",
                        labelCheckedClassName="active",
                        options=[
                            {"label": "Config 1", "value": 1},
                            {"label": "Config 2", "value": 2},
                            {"label": "Config 3", "value": 3},
                            {"label": "Config 4", "value": 4},
                        ],
                        value=1,
                        style={'width': 500},
                        labelStyle={'width': '100%'}
                    ),
                    html.Button('Save Config', id='savec', className="btn btn-outline-primary")
                ], style={'padding': 20, 'display': 'flex', 'flexDirection': 'row'})
            ]),
        ], align="center")
    ])

    return layout
