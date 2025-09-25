# Change to the src directory and run main.py with the virtual environment Python
Set-Location -Path "$PSScriptRoot/src"
& "$PSScriptRoot/venv/Scripts/python.exe" main.py
Write-Host "Press any key to continue..."
$null = $Host.UI.RawUI.ReadKey("NoEcho,IncludeKeyDown")
