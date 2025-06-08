 
cmake_minimum_required(VERSION 3.20)
project(SnakeAI-MLOps)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(sfml REQUIRED COMPONENTS system window graphics)
find_package(nlohmann_json REQUIRED)
find_package(spdlog REQUIRED)

add_executable(${PROJECT_NAME}
    src/main.cpp
)

target_link_libraries(${PROJECT_NAME} 
    sfml-system 
    sfml-window 
    sfml-graphics
    nlohmann_json::nlohmann_json
    spdlog::spdlog
)