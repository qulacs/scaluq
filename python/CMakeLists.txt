cmake_minimum_required(VERSION 3.21)


message(STATUS "Building library for python...")

nanobind_add_module(scaluq_core STABLE_ABI binding.cpp)
target_link_libraries(scaluq_core PRIVATE scaluq
    Kokkos::kokkos
    Kokkos::kokkoskernels
    Eigen3::Eigen
)
target_include_directories(scaluq_core PRIVATE ${PROJECT_SOURCE_DIR}/scaluq)
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
nanobind_add_stub(
    scaluq_stub
    INSTALL_TIME
    MODULE scaluq.scaluq_core.gate
    OUTPUT scaluq/gate.pyi
    PYTHON_PATH $<TARGET_FILE_DIR:scaluq>
    VERBOSE
)
