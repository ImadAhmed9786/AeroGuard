@echo off
title AeroGuard - Emergency Flight Data Backup
cd /d "%~dp0"

echo ==========================================
echo       AEROGUARD DEMONSTRATION
echo ==========================================
echo.
echo Starting candidate monitoring system...
echo.

python aeroguard_single_candidate.py

echo.
echo AeroGuard has stopped.
pause
