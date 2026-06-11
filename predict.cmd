@echo off
powershell -NoProfile -ExecutionPolicy Bypass -File "%~dp0scripts\ag_predict.ps1" %*

