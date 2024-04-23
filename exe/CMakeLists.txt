cmake_minimum_required(VERSION 3.21)

function(exe name)
    add_executable(${name} ${name}.cpp)
    target_link_libraries(${name} PRIVATE
        scaluq
        Eigen3::Eigen
        Kokkos::kokkos
        Kokkos::kokkoskernels
        GTest::gtest_main
    )
    target_include_directories(${name} PRIVATE ${PROJECT_SOURCE_DIR}/scaluq)
endfunction()

# exe({file}) を追加すると、{file}.cpp をコンパイルして実行ファイルを build/exe/{file} として生成する
exe(main)
