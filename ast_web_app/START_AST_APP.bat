@echo off
REM ============================================================================
REM AST Lite Web App - Startup Script (Configured for MLABIADH)
REM ============================================================================

echo ========================================
echo   AST Lite Web Application
echo   Starting Server...
echo ========================================
echo.

REM Initialize Anaconda from your installation
echo Initializing Anaconda...
call "C:\Users\MLABIADH\AppData\Local\anaconda3\Scripts\activate.bat"

if errorlevel 1 (
    echo ERROR: Failed to initialize Anaconda
    echo Please check if Anaconda is installed at:
    echo C:\Users\MLABIADH\AppData\Local\anaconda3
    pause
    exit /b 1
)

REM Change to the application directory
cd /d W:\srm\gss\sandbox\mlabiadh\git\ast-rework\ast_web_app

REM Activate conda environment
echo Activating conda environment: ast_py310
call conda activate ast_py310

if errorlevel 1 (
    echo ERROR: Failed to activate conda environment 'ast_py310'
    echo.
    echo Available environments:
    call conda env list
    echo.
    echo Please make sure the 'ast_py310' environment exists.
    pause
    exit /b 1
)

echo.
echo Starting Flask/Dash server...
echo The application will open in your browser shortly...
echo.
echo ========================================
echo  Server is starting...
echo  URL: http://localhost:8050/
echo ========================================
echo.
echo Press Ctrl+C in this window to stop the server
echo.

REM Start the application in the background
start /B python app.py

REM Wait for Flask/Dash to be available on port 8050
echo Waiting for server to initialize...

:wait_loop
powershell -Command ^
  "try { (New-Object Net.Sockets.TcpClient('localhost',8050)).Close(); exit 0 } catch { exit 1 }"

if errorlevel 1 (
    timeout /t 1 /nobreak >nul
    goto wait_loop
)

echo Server is ready!
echo Opening browser...
start http://localhost:8050/

echo.
echo ========================================
echo  Server is running!
echo  Browser should open automatically.
echo  
echo  To stop the server:
echo  - Close this window, OR
echo  - Press Ctrl+C
echo ========================================
echo.

REM Keep the window open
pause >nul
