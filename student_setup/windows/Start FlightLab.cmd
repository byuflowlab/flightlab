@echo off
setlocal
title FlightLab Workbench

rem Double-click launcher for the ME 415 FlightLab workbench.
rem FlightLab is pinned so every student uses the same course version.
rem
rem The first launch downloads uv, Python 3.12, and the course build into a
rem FlightLab-only folder under %LOCALAPPDATA%\FlightLab. Later launches start
rem that installed copy directly, so nothing is downloaded or re-resolved again
rem until the instructor promotes a new course build.

set "FLIGHTLAB_DEFAULT_COMMIT=0ee06b60ba2d657cb7dbe324faef81d2c8be8e5a"
set "FLIGHTLAB_RELEASE_URL=https://raw.githubusercontent.com/byuflowlab/flightlab/main/student_setup/release.txt"
set "FLIGHTLAB_DATA_ROOT=%LOCALAPPDATA%\FlightLab"
set "FLIGHTLAB_UV_DIR=%FLIGHTLAB_DATA_ROOT%\uv"
set "FLIGHTLAB_UV=%FLIGHTLAB_UV_DIR%\uv.exe"
set "FLIGHTLAB_RELEASE_FILE=%FLIGHTLAB_DATA_ROOT%\release.txt"
set "FLIGHTLAB_RELEASE_TEMP=%FLIGHTLAB_DATA_ROOT%\release-download.txt"
set "FLIGHTLAB_ENV_DIR=%FLIGHTLAB_DATA_ROOT%\env"
set "FLIGHTLAB_ENV_TEMP=%FLIGHTLAB_DATA_ROOT%\env-install"
set "FLIGHTLAB_INSTALLED_FILE=%FLIGHTLAB_DATA_ROOT%\installed.txt"
set "FLIGHTLAB_PYTHON=%FLIGHTLAB_ENV_DIR%\Scripts\python.exe"
set "FLIGHTLAB_INSTALLED="

echo FlightLab Workbench
echo ==============================================
echo.

if not exist "%FLIGHTLAB_DATA_ROOT%" mkdir "%FLIGHTLAB_DATA_ROOT%"

if not exist "%FLIGHTLAB_UV%" (
    echo First-time setup: downloading the FlightLab launcher...
    echo This does not need administrator access.
    echo.
    if not exist "%FLIGHTLAB_UV_DIR%" mkdir "%FLIGHTLAB_UV_DIR%"
    powershell.exe -NoProfile -ExecutionPolicy Bypass -Command "$env:UV_UNMANAGED_INSTALL='%FLIGHTLAB_UV_DIR%'; irm 'https://astral.sh/uv/install.ps1' | iex"
    if errorlevel 1 goto setup_error
    if not exist "%FLIGHTLAB_UV%" goto setup_error
)

echo Checking for a course update...
powershell.exe -NoProfile -ExecutionPolicy Bypass -Command "$v=(Invoke-RestMethod -TimeoutSec 15 '%FLIGHTLAB_RELEASE_URL%').Trim(); if ($v -notmatch '^[0-9a-f]{40}$') { exit 1 }; Set-Content -NoNewline -Path '%FLIGHTLAB_RELEASE_TEMP%' -Value $v"
if errorlevel 1 goto update_unavailable
move /Y "%FLIGHTLAB_RELEASE_TEMP%" "%FLIGHTLAB_RELEASE_FILE%" >nul
goto update_ready

:update_unavailable
if exist "%FLIGHTLAB_RELEASE_FILE%" (
    echo The update check was unavailable; using the most recent downloaded version.
    goto update_ready
)
echo The update check was unavailable; using the version included with this launcher.
>"%FLIGHTLAB_RELEASE_FILE%" echo %FLIGHTLAB_DEFAULT_COMMIT%

:update_ready
set /p FLIGHTLAB_COMMIT=<"%FLIGHTLAB_RELEASE_FILE%"
if exist "%FLIGHTLAB_INSTALLED_FILE%" set /p FLIGHTLAB_INSTALLED=<"%FLIGHTLAB_INSTALLED_FILE%"
if not exist "%FLIGHTLAB_PYTHON%" goto install
if not "%FLIGHTLAB_INSTALLED%"=="%FLIGHTLAB_COMMIT%" goto install
goto run

:install
set "FLIGHTLAB_REQUIREMENT=flightlab[workbench] @ https://github.com/byuflowlab/flightlab/archive/%FLIGHTLAB_COMMIT%.zip"
echo Installing course build %FLIGHTLAB_COMMIT:~0,8%...
echo The first install downloads Python and FlightLab's libraries (150-200 MB)
echo and can take several minutes. Later updates reuse what is already downloaded.
echo.
if exist "%FLIGHTLAB_ENV_TEMP%" rmdir /s /q "%FLIGHTLAB_ENV_TEMP%"
"%FLIGHTLAB_UV%" python install 3.12
if errorlevel 1 goto install_failed
"%FLIGHTLAB_UV%" venv --quiet --python 3.12 "%FLIGHTLAB_ENV_TEMP%"
if errorlevel 1 goto install_failed
"%FLIGHTLAB_UV%" pip install --python "%FLIGHTLAB_ENV_TEMP%\Scripts\python.exe" "%FLIGHTLAB_REQUIREMENT%"
if errorlevel 1 goto install_failed
if exist "%FLIGHTLAB_ENV_DIR%" rmdir /s /q "%FLIGHTLAB_ENV_DIR%"
move "%FLIGHTLAB_ENV_TEMP%" "%FLIGHTLAB_ENV_DIR%" >nul
if errorlevel 1 goto install_failed
>"%FLIGHTLAB_INSTALLED_FILE%" echo %FLIGHTLAB_COMMIT%
echo.
goto run

:install_failed
if exist "%FLIGHTLAB_ENV_TEMP%" rmdir /s /q "%FLIGHTLAB_ENV_TEMP%"
echo.
if not exist "%FLIGHTLAB_PYTHON%" goto install_error
if "%FLIGHTLAB_INSTALLED%"=="" goto install_error
echo The update could not be installed; starting the previously installed build instead.
set "FLIGHTLAB_COMMIT=%FLIGHTLAB_INSTALLED%"
goto run

:run
echo Course build: %FLIGHTLAB_COMMIT:~0,8%
echo Starting FlightLab. Your web browser will open when it is ready.
echo.
echo Keep this window open while using FlightLab.
echo Close this window, or press Control-C, when you are finished.
echo.

if "%FLIGHTLAB_TEST_ONLY%"=="1" (
    "%FLIGHTLAB_PYTHON%" -m flightlab
) else (
    "%FLIGHTLAB_PYTHON%" -m flightlab workbench
)
if errorlevel 1 goto run_error
exit /b 0

:setup_error
echo.
echo Setup could not be downloaded. Check the internet connection and try again.
echo If the problem continues, take a screenshot of this window and send it to your TA.
if "%FLIGHTLAB_TEST_ONLY%"=="1" exit /b 1
pause
exit /b 1

:install_error
echo FlightLab could not be installed. Check the internet connection and try again.
echo If the problem continues, take a screenshot of this window and send it to your TA.
if "%FLIGHTLAB_TEST_ONLY%"=="1" exit /b 1
pause
exit /b 1

:run_error
echo.
echo FlightLab stopped because of an error.
echo Take a screenshot of this window and send it to your TA.
if "%FLIGHTLAB_TEST_ONLY%"=="1" exit /b 1
pause
exit /b 1
