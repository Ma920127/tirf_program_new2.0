@echo off
REM Open Chrome to the specified URL.
start chrome.exe --new-tab http://127.0.0.1:8041/

REM Change the current directory to the directory where this script is located.
cd /d %~dp0

REM Activate the virtual environment from the main project root.
call ..\..\..\..\.venv\Scripts\activate.bat

REM Change to the Trace_viewer directory.
cd /d %~dp0

@echo on
python Trace_viewer_server.py

PAUSE