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
        transforms=[BlockingCallbackTransform(timeout = 10)]
    )

    events = [{'event': 'keydown', 'props': ['key']}]

    app.layout = html.Div([
        EventListener(id='key_events', events=events),     
        dcc.Graph(id="graph", figure=fig, config={'doubleClick': False}),
        
        # Wrap the Tabs in a Div to apply parent styles.
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
                # "style" affects the tab label area.
                style={'width': "5%", "height": '600px'},
                # "content_style" affects the content area.
                content_style={'width': "95%", "height": '100%'}
            ),
            # Apply parent styles by wrapping the Tabs in a Div.
            style={'width': "100%", "height": "100%"}
        ),
        
        html.Div(children=None, id='trash', hidden=True),
        html.Br(), html.Br(), html.Br(), html.Br(), html.Br(), html.Br()
    ])


    return app
