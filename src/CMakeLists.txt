cmake_minimum_required(VERSION 3.21)

target_sources(scaluq PRIVATE
    state/state_vector.cpp
    util/utility.cpp
    kokkos.cpp
    types.cpp
)
target_link_libraries(scaluq PUBLIC 
    Kokkos::kokkos
)
target_include_directories(scaluq PRIVATE $<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}/include> $<INSTALL_INTERFACE:include>)
target_include_directories(scaluq PRIVATE ${eigen_SOURCE_DIR})

install(DIRECTORY ${PROJECT_SOURCE_DIR}/include/scaluq DESTINATION include/scaluq)
install(TARGETS scaluq kokkos kokkoscore kokkoscontainers kokkosalgorithms kokkossimd LIBDL EXPORT scaluqTargets LIBRARY DESTINATION lib PUBLIC_HEADER DESTINATION include)
install(EXPORT scaluqTargets FILE scaluq-config.cmake DESTINATION share/cmake/scaluq/ NAMESPACE scaluq::)
