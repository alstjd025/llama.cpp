# llama.cpp/tools/arrayfire/CMakeLists.txt

# ArrayFire wrapper 라이브러리 정의
add_library(arrayfire_wrapper STATIC
    Arrayfire_wrapper.cpp
)

# Include 경로 설정 (Arrayfire 및 현재 디렉토리)
target_include_directories(arrayfire_wrapper PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}                      # for Arrayfire_wrapper.h
    /host_home/mzcache_libs/arrayfire/include     # for arrayfire.h
)

# 외부 ArrayFire 라이브러리 (IMPORTED)
add_library(afopencl SHARED IMPORTED GLOBAL)
set_target_properties(afopencl PROPERTIES
    IMPORTED_LOCATION /host_home/mzcache_libs/arrayfire/libafopencl.so
    INTERFACE_INCLUDE_DIRECTORIES /host_home/mzcache_libs/arrayfire/include
)

# arrayfire_wrapper가 afopencl을 링크하도록 설정
target_link_libraries(arrayfire_wrapper PUBLIC afopencl)
