cmake_minimum_required(VERSION 3.20)
project(SnakeAI-MLOps)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(SFML COMPONENTS System Window Graphics CONFIG REQUIRED)
find_package(nlohmann_json CONFIG REQUIRED)
find_package(spdlog CONFIG REQUIRED)

add_executable(${PROJECT_NAME}
    src/main.cpp
    src/Game.cpp
    src/Snake.cpp
    src/Apple.cpp
    src/Grid.cpp
    src/Menu.cpp
    src/QLearningAgent.cpp
    src/DataCollector.cpp
    src/InputManager.cpp
)

target_link_libraries(${PROJECT_NAME} 
    SFML::System 
    SFML::Window 
    SFML::Graphics
    nlohmann_json::nlohmann_json
    spdlog::spdlog
)

add_custom_command(TARGET ${PROJECT_NAME} POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E copy_directory
    ${CMAKE_SOURCE_DIR}/assets $<TARGET_FILE_DIR:${PROJECT_NAME}>/assets)