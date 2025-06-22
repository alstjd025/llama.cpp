##################################################################################
# Minsung Kim @ NXC, SNU.
# This file is part of the mzCache project.
# ##################################################################################
# CMakeLists for ArrayFire wrapper.
##################################################################################
# llama.cpp/tools/arrayfire/CMakeLists.txt

# ArrayFire wrapper
add_library(arrayfire_wrapper STATIC
    Arrayfire_wrapper.cpp
)

# Include directories
target_include_directories(arrayfire_wrapper PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}                      # for Arrayfire_wrapper.h
    /host_home/mzcache_libs/arrayfire/include     # for arrayfire.h
)

# External ArrayFire libs (IMPORTED)
add_library(afopencl SHARED IMPORTED GLOBAL)
set_target_properties(afopencl PROPERTIES
    IMPORTED_LOCATION /host_home/mzcache_libs/arrayfire/libafopencl.so
    INTERFACE_INCLUDE_DIRECTORIES /host_home/mzcache_libs/arrayfire/include
)

# Link the external ArrayFire library to the wrapper
target_link_libraries(arrayfire_wrapper PUBLIC afopencl)
