@echo off
title AgriInsight Platform
set PYTHONUTF8=1
echo Starting AgriInsight Streamlit Application...
py -m streamlit run app.py
if %ERRORLEVEL% NEQ 0 (
    "%LOCALAPPDATA%\Python\bin\python.exe" -m streamlit run app.py
)
pause
