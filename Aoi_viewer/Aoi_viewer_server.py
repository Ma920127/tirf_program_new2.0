import sys

# Force UTF-8 console output.  Windows consoles in non-Latin locales
# (e.g. cp950 on Traditional-Chinese Windows) cannot encode the status
# glyphs used across the app (✓, ✗, →, emoji); without this the server
# crashes with UnicodeEncodeError before it ever starts serving.
try:
    sys.stdout.reconfigure(encoding='utf-8')
    sys.stderr.reconfigure(encoding='utf-8')
except Exception:
    pass

from aoi_viewer import server, app
from waitress import serve

if __name__ == '__main__':
    print("🚀 Starting AOI Viewer Server on http://0.0.0.0:8042 ...")
    serve(server, host="0.0.0.0", port=8042, threads=12)
