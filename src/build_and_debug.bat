@echo off
echo ================================================================
echo          SnakeAI-MLOps Build and Debug Script
echo ================================================================
echo.

REM Check if CMake is available
cmake --version >nul 2>&1
if errorlevel 1 (
    echo ❌ CMake not found in PATH
    echo Please install CMake and add it to your PATH
    pause
    exit /b 1
)

REM Check if vcpkg is set up
if not defined VCPKG_ROOT (
    echo ❌ VCPKG_ROOT environment variable not set
    echo Please run: setx VCPKG_ROOT "path\to\your\vcpkg"
    pause
    exit /b 1
)

echo ✅ Build environment looks good
echo.

REM Create build directory
if not exist "out\build\windows-default" (
    echo 📁 Creating build directory...
    mkdir "out\build\windows-default"
)

REM Configure with CMake
echo 🔧 Configuring project with CMake...
cmake --preset windows-default -DCMAKE_BUILD_TYPE=Debug
if errorlevel 1 (
    echo ❌ CMake configuration failed
    echo Check CMakeLists.txt and LibTorch installation
    pause
    exit /b 1
)

echo ✅ Configuration successful
echo.

REM Build the project
echo 🔨 Building project...
cmake --build out\build\windows-default --config Debug
if errorlevel 1 (
    echo ❌ Build failed
    echo Check compiler errors above
    pause
    exit /b 1
)

echo ✅ Build successful
echo.

REM Check if debug executable exists
if exist "out\build\windows-default\Debug\ModelVerification.exe" (
    echo 🧪 Running Model Debugger...
    echo.
    out\build\windows-default\Debug\ModelVerification.exe
) else (
    echo ⚠️  Model debugger not built, trying to build it separately...
    
    REM Try to build the debug utility manually
    echo 🔧 Building ModelDebugger...
    
    REM Create a temporary CMakeLists for the debugger
    echo cmake_minimum_required(VERSION 3.18) > debug_build.cmake
    echo project(ModelDebugger) >> debug_build.cmake
    echo set(CMAKE_CXX_STANDARD 17) >> debug_build.cmake
    echo find_package(Torch REQUIRED) >> debug_build.cmake
    echo find_package(nlohmann_json REQUIRED) >> debug_build.cmake
    echo find_package(spdlog REQUIRED) >> debug_build.cmake
    echo add_executable(ModelDebugger src/ModelDebugger.cpp src/MLAgents.cpp src/TorchInference.cpp) >> debug_build.cmake
    echo target_include_directories(ModelDebugger PRIVATE src/ ${TORCH_INCLUDE_DIRS}) >> debug_build.cmake
    echo target_link_libraries(ModelDebugger PRIVATE nlohmann_json::nlohmann_json spdlog::spdlog ${TORCH_LIBRARIES}) >> debug_build.cmake
    
    mkdir debug_build 2>nul
    cd debug_build
    cmake -DCMAKE_TOOLCHAIN_FILE=%VCPKG_ROOT%\scripts\buildsystems\vcpkg.cmake ..
    cmake --build . --config Debug
    
    if exist "Debug\ModelDebugger.exe" (
        echo ✅ Debug utility built successfully
        Debug\ModelDebugger.exe
    ) else (
        echo ❌ Failed to build debug utility
    )
    
    cd ..
    rmdir /s /q debug_build 2>nul
    del debug_build.cmake 2>nul
)

echo.
echo 🎮 Starting main game...
if exist "out\build\windows-default\Debug\SnakeAI-MLOps.exe" (
    out\build\windows-default\Debug\SnakeAI-MLOps.exe
) else if exist "out\build\windows-default\Release\SnakeAI-MLOps.exe" (
    out\build\windows-default\Release\SnakeAI-MLOps.exe
) else (
    echo ❌ Game executable not found
    echo Build may have failed or executable is in a different location
)

echo.
echo 📋 Build and debug session complete!
echo Check the following files for detailed information:
echo   - logs\debug.log (game runtime logs)
echo   - logs\model_debug.log (model debugging logs)
echo.
pause