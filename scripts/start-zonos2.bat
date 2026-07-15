@echo off
rem start-zonos2 — double-click to download the models (if missing) and launch the server.
powershell -NoProfile -ExecutionPolicy Bypass -File "%~dp0start-zonos2.ps1" %*
if errorlevel 1 pause
