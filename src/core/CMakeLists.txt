cmake_minimum_required(VERSION 3.11)

add_library(core)

target_sources(core PRIVATE
    random.cpp
)
target_include_directories(core PRIVATE ${PROJECT_SOURCE_DIR}/src)
