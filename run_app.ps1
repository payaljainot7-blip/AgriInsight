# Launch AgriInsight Streamlit Application
$env:PYTHONUTF8 = "1"
Write-Host "Starting AgriInsight Platform..." -ForegroundColor Green

if (Get-Command py -ErrorAction SilentlyContinue) {
    py -m streamlit run app.py
} elseif (Test-Path "$env:LOCALAPPDATA\Python\bin\python.exe") {
    & "$env:LOCALAPPDATA\Python\bin\python.exe" -m streamlit run app.py
} else {
    Write-Error "Python executable not located. Please verify your Python installation."
}
