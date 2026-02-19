@echo off
echo ========================================
echo   AST-NEXT Web Application
echo   Starting Server...
echo ========================================
echo.

REM === Configuration ===
set ENV_ARCHIVE=\\giswhse.env.gov.bc.ca\whse_np\corp\script_whse\python\Utility_Misc\In_Progress\ast_next_test\env\ast_py310.tar.gz
set ENV_VERSION=\\giswhse.env.gov.bc.ca\whse_np\corp\script_whse\python\Utility_Misc\In_Progress\ast_next_test\env\.env_version
set ENV_LOCAL=C:\ast_env\ast_py310
set APP_DIR=\\spatialfiles.bcgov\Work\srm\gss\sandbox\mlabiadh\git\ast-rework\ast_web_app

REM === Cache the Python env locally for performance ===
set NEEDS_SETUP=0

REM Check if local cache exists and is up to date
if not exist "%ENV_LOCAL%\python.exe" (
    set NEEDS_SETUP=1
) else (
    if not exist "%ENV_LOCAL%\.env_version" (
        set NEEDS_SETUP=1
    ) else (
        fc /B "%ENV_VERSION%" "%ENV_LOCAL%\.env_version" >nul 2>&1
        if errorlevel 1 set NEEDS_SETUP=1
    )
)

if "%NEEDS_SETUP%"=="1" (
    echo ========================================
    echo   First-time setup detected.
    echo   Please wait while the environment is
    echo   being prepared. This may take a few
    echo   minutes but only happens once.
    echo.
    echo   DO NOT close this window.
    echo ========================================
    echo.
    if exist "%ENV_LOCAL%" rmdir /S /Q "%ENV_LOCAL%"
    if not exist "C:\ast_env" mkdir "C:\ast_env"
    echo [1/2] Downloading environment...
    copy /Y "%ENV_ARCHIVE%" "C:\ast_env\ast_py310.tar.gz" >nul
    if errorlevel 1 (
        echo ERROR: Failed to copy environment archive.
        pause
        exit /b 1
    )
    echo [2/2] Extracting environment...
    mkdir "%ENV_LOCAL%"
    tar -xzf "C:\ast_env\ast_py310.tar.gz" -C "%ENV_LOCAL%"
    if errorlevel 1 (
        echo ERROR: Failed to extract environment.
        pause
        exit /b 1
    )
    copy /Y "%ENV_VERSION%" "%ENV_LOCAL%\.env_version" >nul
    del "C:\ast_env\ast_py310.tar.gz"
    echo.
    echo Setup complete!
    echo.
)

set PYTHON_EXE=%ENV_LOCAL%\python.exe

REM Change to app directory (pushd maps UNC paths to a temp drive letter)
pushd "%APP_DIR%"

REM Start the application in background
echo Starting Flask/Dash server...
start /B "" "%PYTHON_EXE%" app.py

REM Wait for server
echo Waiting for server to initialize...
:wait_loop
powershell -Command ^
  "try { (New-Object Net.Sockets.TcpClient('localhost',8050)).Close(); exit 0 } catch { exit 1 }"
if errorlevel 1 (
    timeout /t 1 /nobreak >nul
    goto wait_loop
)

echo Server is ready!
start http://localhost:8050/
echo.
echo Press Ctrl+C or close this window to stop.
pause >nul
