from waitress import serve 
from app import server 



if __name__ == '__main__':
    print("🚀 Starting Trace Viewer Server on http://0.0.0.0:8041 ...")
    serve(server, host="0.0.0.0", port=8041, threads=12) 