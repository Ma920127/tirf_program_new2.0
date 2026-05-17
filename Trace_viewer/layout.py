from dash_extensions.enrich import DashProxy, dcc, html, BlockingCallbackTransform
from dash_extensions import EventListener
import dash_bootstrap_components as dbc

# Import tab layouts
from layout_tabs.tools import tools_tab
from layout_tabs.aois import aois_tab
from layout_tabs.hmm import hmm_tab
from layout_tabs.gmm import gmm_tab
from layout_tabs.rupture import rupture_tab

def make_app(fig, fig_blob, fig2):
    app = DashProxy(
        __name__,
        external_stylesheets=[dbc.themes.LUMEN],
        prevent_initial_callbacks=True,
        transforms=[BlockingCallbackTransform(timeout=10)]
    )

    events = [{'event': 'keydown', 'props': ['key']}]

    app.layout = html.Div([
        EventListener(id='key_events', events=events),

        # ── Graph + HMM sidebar stacked together ──
        html.Div([
            dcc.Graph(id="graph", figure=fig, config={'doubleClick': False}),

            # HMM State Bar — only visible when HMM tab is active
            html.Div(
                id='hmm-state-bar-container',
                children=[
                    # Title row
                    html.Div(
                        'HMM States',
                        style={
                            'fontFamily': 'Arial, sans-serif',
                            'fontSize': '13px',
                            'fontWeight': 'bold',
                            'color': '#333',
                            'marginLeft': '80px',
                            'marginBottom': '4px',
                            'letterSpacing': '0.5px',
                        }
                    ),
                    # The actual coloured segment bar (Plotly graph for click detection)
                    # Width = 90% of figure plot-area = xaxis1 domain [0,0.9] × (W − margin.l − margin.r)
                    # margin.l=80, margin.r=20  →  bar starts at 80px, width = calc(90%·(W−100)) ≈ calc(90% − 90px)
                    dcc.Graph(
                        id='hmm-state-bar',
                        figure={
                            'data': [],
                            'layout': {
                                'height': 50,
                                'margin': {'l': 0, 'r': 0, 't': 0, 'b': 0},
                                'xaxis': {'visible': False, 'fixedrange': True},
                                'yaxis': {'visible': False, 'fixedrange': True},
                                'paper_bgcolor': '#f5f5f5',
                                'plot_bgcolor': '#f5f5f5',
                                'dragmode': False,
                                'annotations': [{
                                    'x': 0.5, 'y': 0.5, 'xref': 'paper', 'yref': 'paper',
                                    'text': 'No HMM prediction loaded — fit HMM first',
                                    'showarrow': False,
                                    'font': {'size': 12, 'color': '#aaa', 'family': 'Arial'},
                                }],
                            }
                        },
                        config={'displayModeBar': False, 'scrollZoom': False},
                        style={
                            'width':      'calc(90% - 90px)',
                            'height':     '50px',
                            'marginLeft': '80px',
                            'border':     '1.5px solid #888',
                            'borderRadius': '3px',
                            'overflow':   'hidden',
                        }
                    ),
                ],
                style={'display': 'none',           # hidden on all other tabs
                       'paddingBottom': '10px'}
            ),
        ]),

        dcc.Store(id='hmm-segments-store',     data=[]),
        dcc.Store(id='hmm-selected-boundary',  data=-1),     # index of highlighted boundary (-1 = none)
        dcc.Store(id='hmm-boundary-seg',       data=-1),     # seg index clicked on bar (-1 = none)
        dcc.Store(id='hmm-vacuum-store',       data=None),   # pending Add-tool vacuum {f_start,f_end,t_start,t_end,ch}

        # ── Tabs ──
        html.Div(
            dcc.Tabs(
                id='tabs',
                value='Tools',
                vertical=True,
                children=[
                    tools_tab(),
                    aois_tab(fig_blob),
                    hmm_tab(),
                    gmm_tab(fig2),
                    rupture_tab()
                ],
                style={'width': "5%", "height": '600px'},
                content_style={'width': "95%", "height": '100%'}
            ),
            style={'width': "100%", "height": "100%"}
        ),

        html.Div(children=None, id='trash', hidden=True),
        html.Br(), html.Br(), html.Br(), html.Br(), html.Br(), html.Br()
    ])

    return app
