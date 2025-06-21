cmake_minimum_required(VERSION 3.20)
project(SnakeAI-MLOps)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Ensure x64 build
set(CMAKE_GENERATOR_PLATFORM x64)

# LibTorch configuration
set(CMAKE_PREFIX_PATH "C:/libtorch")
set(Torch_DIR "C:/libtorch/share/cmake/Torch")

find_package(Torch REQUIRED)
find_package(SFML COMPONENTS System Window Graphics CONFIG REQUIRED)
find_package(nlohmann_json CONFIG REQUIRED)
find_package(spdlog CONFIG REQUIRED)
find_package(CUDAToolkit REQUIRED)

# Set compiler flags
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${TORCH_CXX_FLAGS}")

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

target_link_libraries(${PROJECT_NAME} 
    SFML::System 
    SFML::Window 
    SFML::Graphics
    nlohmann_json::nlohmann_json
    spdlog::spdlog
    ${TORCH_LIBRARIES}
    CUDA::cudart
)

# Suppress LibTorch warnings
if(MSVC)
    target_compile_options(${PROJECT_NAME} PRIVATE 
        /wd4267  # size_t to int conversion
        /wd4244  # possible loss of data
        /wd4996  # deprecated functions
    )
    
    # Copy LibTorch DLLs
    file(GLOB TORCH_DLLS "${TORCH_INSTALL_PREFIX}/lib/*.dll")
    add_custom_command(TARGET ${PROJECT_NAME} POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E copy_if_different
        ${TORCH_DLLS} $<TARGET_FILE_DIR:${PROJECT_NAME}>)
endif()

# Copy assets
add_custom_command(TARGET ${PROJECT_NAME} POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E copy_directory
    ${CMAKE_SOURCE_DIR}/assets $<TARGET_FILE_DIR:${PROJECT_NAME}>/assets)

# Copy models directory (optional)
add_custom_command(TARGET ${PROJECT_NAME} POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E copy_directory
    ${CMAKE_SOURCE_DIR}/models $<TARGET_FILE_DIR:${PROJECT_NAME}>/models)