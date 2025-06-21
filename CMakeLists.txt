cmake_minimum_required(VERSION 3.20)
project(SnakeAI-MLOps)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Ensure x64 build
set(CMAKE_GENERATOR_PLATFORM x64)

# Required dependencies
find_package(SFML COMPONENTS System Window Graphics CONFIG REQUIRED)
find_package(nlohmann_json CONFIG REQUIRED)
find_package(spdlog CONFIG REQUIRED)

# Optional LibTorch configuration (for future neural network implementations)
set(CMAKE_PREFIX_PATH "C:/libtorch")
set(Torch_DIR "C:/libtorch/share/cmake/Torch")

# Try to find optional dependencies
find_package(Torch QUIET)
find_package(CUDAToolkit QUIET)

# Set compiler flags
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS}")
if(Torch_FOUND)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${TORCH_CXX_FLAGS}")
    message(STATUS "LibTorch found - neural network support enabled")
else()
    message(STATUS "LibTorch not found - continuing without neural network support")
endif()

add_executable(${PROJECT_NAME}
    src/main.cpp
    src/Game.cpp
    src/Snake.cpp
    src/Apple.cpp
    src/Grid.cpp
    src/Menu.cpp
    src/PauseMenu.cpp
    src/AgentSelection.cpp
    src/Leaderboard.cpp
    src/QLearningAgent.cpp
    src/MLAgents.cpp
    src/InputManager.cpp
    src/UnifiedDataCollector.cpp
    src/TorchInference.cpp
    src/StatsGallery.cpp
    src/Settings.cpp
    src/HowToPlay.cpp
)

# Link required libraries
target_link_libraries(${PROJECT_NAME} 
    SFML::System 
    SFML::Window 
    SFML::Graphics
    nlohmann_json::nlohmann_json
    spdlog::spdlog
)

# Link optional libraries if available
if(Torch_FOUND)
    target_link_libraries(${PROJECT_NAME} ${TORCH_LIBRARIES})
    target_compile_definitions(${PROJECT_NAME} PRIVATE TORCH_AVAILABLE)
endif()

if(CUDAToolkit_FOUND)
    target_link_libraries(${PROJECT_NAME} CUDA::cudart)
    target_compile_definitions(${PROJECT_NAME} PRIVATE CUDA_AVAILABLE)
endif()

# Suppress warnings for MSVC
if(MSVC)
    target_compile_options(${PROJECT_NAME} PRIVATE 
        /wd4267  # size_t to int conversion
        /wd4244  # possible loss of data
        /wd4996  # deprecated functions
    )
    
    # Copy LibTorch DLLs if available
    if(Torch_FOUND)
        file(GLOB TORCH_DLLS "${TORCH_INSTALL_PREFIX}/lib/*.dll")
        if(TORCH_DLLS)
            add_custom_command(TARGET ${PROJECT_NAME} POST_BUILD
                COMMAND ${CMAKE_COMMAND} -E copy_if_different
                ${TORCH_DLLS} $<TARGET_FILE_DIR:${PROJECT_NAME}>)
        endif()
    endif()
endif()

# Copy assets
add_custom_command(TARGET ${PROJECT_NAME} POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E copy_directory
    ${CMAKE_SOURCE_DIR}/assets $<TARGET_FILE_DIR:${PROJECT_NAME}>/assets)

# Copy models directory (optional)
add_custom_command(TARGET ${PROJECT_NAME} POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E copy_directory
    ${CMAKE_SOURCE_DIR}/models $<TARGET_FILE_DIR:${PROJECT_NAME}>/models)