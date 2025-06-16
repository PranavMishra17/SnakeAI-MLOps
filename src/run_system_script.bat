@echo off
:: SnakeAI-MLOps System Runner
:: Automated build and execution script for Windows

echo.
echo ============================================
echo 🐍 SnakeAI-MLOps System Runner
echo ============================================
echo.

:: Check if we're in the right directory
if not exist "src\main.cpp" (
    echo ❌ Error: Please run this script from the SnakeAI-MLOps root directory
    echo    Current directory: %CD%
    echo    Expected files: src\main.cpp, CMakeLists.txt
    pause
    exit /b 1
)

:: Check for required tools
echo 🔍 Checking prerequisites...

where cmake >nul 2>&1
if %ERRORLEVEL% neq 0 (
    echo ❌ CMake not found. Please install CMake 3.20+ and add to PATH
    pause
    exit /b 1
)

where git >nul 2>&1
if %ERRORLEVEL% neq 0 (
    echo ❌ Git not found. Please install Git and add to PATH
    pause
    exit /b 1
)

:: Check vcpkg
if not defined VCPKG_ROOT (
    echo ❌ VCPKG_ROOT environment variable not set
    echo    Please install vcpkg and set VCPKG_ROOT to vcpkg directory
    echo    Example: setx VCPKG_ROOT "C:\tools\vcpkg"
    pause
    exit /b 1
)

if not exist "%VCPKG_ROOT%\vcpkg.exe" (
    echo ❌ vcpkg.exe not found at %VCPKG_ROOT%
    echo    Please ensure vcpkg is properly installed and bootstrapped
    pause
    exit /b 1
)

echo ✅ Prerequisites check passed

:: Check and install dependencies
echo.
echo 📦 Checking dependencies...

"%VCPKG_ROOT%\vcpkg.exe" list | findstr "sfml:x64-windows" >nul
if %ERRORLEVEL% neq 0 (
    echo 📥 Installing SFML...
    "%VCPKG_ROOT%\vcpkg.exe" install sfml:x64-windows
    if %ERRORLEVEL% neq 0 (
        echo ❌ Failed to install SFML
        pause
        exit /b 1
    )
)

"%VCPKG_ROOT%\vcpkg.exe" list | findstr "nlohmann-json:x64-windows" >nul
if %ERRORLEVEL% neq 0 (
    echo 📥 Installing nlohmann-json...
    "%VCPKG_ROOT%\vcpkg.exe" install nlohmann-json:x64-windows
    if %ERRORLEVEL% neq 0 (
        echo ❌ Failed to install nlohmann-json
        pause
        exit /b 1
    )
)

"%VCPKG_ROOT%\vcpkg.exe" list | findstr "spdlog:x64-windows" >nul
if %ERRORLEVEL% neq 0 (
    echo 📥 Installing spdlog...
    "%VCPKG_ROOT%\vcpkg.exe" install spdlog:x64-windows
    if %ERRORLEVEL% neq 0 (
        echo ❌ Failed to install spdlog
        pause
        exit /b 1
    )
)

echo ✅ Dependencies verified

:: Create required directories
echo.
echo 📁 Creating directories...
if not exist "assets\fonts" mkdir "assets\fonts"
if not exist "models" mkdir "models"
if not exist "models\qlearning" mkdir "models\qlearning"
if not exist "data" mkdir "data"
if not exist "logs" mkdir "logs"

:: Check for required font
if not exist "assets\fonts\arial.ttf" (
    echo ⚠️  Warning: arial.ttf not found in assets\fonts\
    echo    The game will start but may have display issues
    echo    Please copy arial.ttf to assets\fonts\ directory
    echo.
)

:: Configure CMake
echo.
echo 🔧 Configuring build...
cmake --preset windows-default
if %ERRORLEVEL% neq 0 (
    echo ❌ CMake configuration failed
    echo    Check CMakePresets.json and vcpkg installation
    pause
    exit /b 1
)

:: Build project
echo.
echo 🔨 Building project...
cmake --build out\build\windows-default --config Release
if %ERRORLEVEL% neq 0 (
    echo ❌ Build failed
    echo    Check compiler installation and dependencies
    pause
    exit /b 1
)

:: Check if executable exists
if not exist "out\build\windows-default\Release\SnakeAI-MLOps.exe" (
    echo ❌ Executable not found after build
    echo    Expected: out\build\windows-default\Release\SnakeAI-MLOps.exe
    pause
    exit /b 1
)

echo ✅ Build completed successfully

:: Run the game
echo.
echo 🚀 Starting SnakeAI-MLOps...
echo ============================================
echo.
echo 🎮 Game Controls:
echo    ↑↓←→ / WASD: Move snake (Single Player)
echo    Mouse Click: Place apple (Agent vs Player)
echo    ESC: Pause menu
echo    +/-: Adjust speed
echo    F1: Leaderboard
echo    F2: Change AI agent
echo.
echo 🤖 Available Features:
echo    - Multiple game modes
echo    - AI agent selection
echo    - Real-time performance monitoring
echo    - Persistent leaderboard
echo    - Enhanced state representation
echo.
echo Press any key to launch the game...
pause >nul

echo Starting game...
cd out\build\windows-default\Release
SnakeAI-MLOps.exe
cd ..\..\..\..

if %ERRORLEVEL% neq 0 (
    echo.
    echo ⚠️  Game exited with error code %ERRORLEVEL%
    echo    Check logs\debug.log for details
) else (
    echo.
    echo ✅ Game exited normally
)

echo.
echo 📊 Check these locations for data:
echo    📁 models\     - Saved AI models
echo    📁 data\       - Training data and metrics
echo    📁 logs\       - Application logs
echo    📄 leaderboard.json - High scores
echo.
pause