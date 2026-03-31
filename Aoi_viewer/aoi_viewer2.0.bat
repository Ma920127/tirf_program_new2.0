@echo off
REM Open Chrome with a new tab for the Aoi viewer.
start chrome.exe --new-tab http://127.0.0.1:8042/

REM Change to the directory where this script is located.
cd /d %~dp0

REM Move one directory up (assuming your .venv folder is located there).
cd ..

REM Activate the virtual environment.
call .venv\Scripts\activate.bat

REM Change to the Aoi_viewer directory.
cd Aoi_viewer

@echo on
REM Run the Aoi_viewer server.
python Aoi_viewer_server.py

PAUSE