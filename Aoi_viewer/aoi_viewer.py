from dash_extensions.enrich import DashProxy, CycleBreakerTransform, BlockingCallbackTransform, FileSystemCache
import dash_bootstrap_components as dbc
from global_state import global_state as gs
from aoi_layout import make_layout
from aoi_callbacks.callback_update_fig import register_update_fig
from aoi_callbacks.callback_load_config import register_load_config
from aoi_callbacks.callback_progress import register_update_progress
from aoi_callbacks.callback_log import register_update_log
from aoi_callbacks.callback_cal_FRET import register_cal_FRET
from aoi_callbacks.callback_auto import register_auto
import logging


# Create the shared cache object and initialize progress values
fsc = FileSystemCache("cache_dir")
fsc.set("load_progress", 0)
fsc.set("progress", 0)
fsc.set("cal_progress", 0)
fsc.set("fret_progress", 0)
fsc.set("stage", 'Idle')
fsc.set('mode', 'manual')

app = DashProxy(
    __name__,
    external_stylesheets=[dbc.themes.ZEPHYR, dbc.icons.BOOTSTRAP],
    prevent_initial_callbacks=True,
    transforms=[CycleBreakerTransform(), BlockingCallbackTransform(timeout = 10)]
)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.StreamHandler(),  # logs to the console
        logging.FileHandler("app.log")  # logs to a file named app.log
    ]
)

app.layout = make_layout(gs.fig)

register_update_fig(app, fsc)
register_load_config(app, fsc)
register_update_progress(app, fsc)
register_update_log(app, fsc)
register_cal_FRET(app, fsc)
register_auto(app, fsc)

server = app.server

if __name__ == '__main__':
    app.run(debug=True)
