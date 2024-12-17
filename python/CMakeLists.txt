cmake_minimum_required(VERSION 3.21)

message(STATUS "Building library for python...")

nanobind_add_module(scaluq_core STABLE_ABI binding.cpp)
target_link_libraries(scaluq_core PRIVATE
    scaluq
    Kokkos::kokkos
    Eigen3::Eigen
    nlohmann_json::nlohmann_json
)
target_include_directories(scaluq_core PRIVATE ${PROJECT_SOURCE_DIR}/include)
target_include_directories(scaluq_core PRIVATE ${eigen_SOURCE_DIR})
install(TARGETS scaluq_core LIBRARY DESTINATION scaluq)

nanobind_add_stub(
    scaluq_stub
    INSTALL_TIME
    MODULE scaluq.scaluq_core
    OUTPUT scaluq/__init__.pyi
    PYTHON_PATH $<TARGET_FILE_DIR:scaluq>
    MARKER_FILE scaluq/py.typed
    VERBOSE
)
foreach(FLOAT IN ITEMS f64 f32)
    nanobind_add_stub(
        scaluq_stub
        INSTALL_TIME
        MODULE scaluq.scaluq_core.${FLOAT}
        OUTPUT scaluq/${FLOAT}/__init__.pyi
        PYTHON_PATH $<TARGET_FILE_DIR:scaluq>
        MARKER_FILE scaluq/py.typed
        VERBOSE
    )
    nanobind_add_stub(
        scaluq_stub
        INSTALL_TIME
        MODULE scaluq.scaluq_core.${FLOAT}.gate
        OUTPUT scaluq/${FLOAT}/gate/__init__.pyi
        PYTHON_PATH $<TARGET_FILE_DIR:scaluq>
        MARKER_FILE scaluq/py.typed
        VERBOSE
    )
endforeach()
