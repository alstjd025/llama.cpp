##################################################################################
# Minsung Kim @ NXC, SNU.
# This file is part of the mzCache project.
# ##################################################################################
# CMakeLists for ArrayFire wrapper in llama.cpp.
##################################################################################

# Prebuilt library directory
# Minsung note: IMPORTANT! make sure that the directory is right and containing proper files.
set(mzCache_prebuilt_libs /host_dir/mzCache_prebuilt)

# ArrayFire wrapper
add_library(arrayfire_wrapper STATIC
    Arrayfire_wrapper.cpp
)

# Include directories
target_include_directories(arrayfire_wrapper PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}                      # for Arrayfire_wrapper.h
    ${mzCache_prebuilt_libs}/arrayfire/include     # for arrayfire.h
)

# External ArrayFire and dependant libs 
add_library(afopencl SHARED IMPORTED GLOBAL)
set_target_properties(afopencl PROPERTIES
    IMPORTED_LOCATION ${mzCache_prebuilt_libs}/arrayfire/libafopencl.so
    INTERFACE_INCLUDE_DIRECTORIES ${mzCache_prebuilt_libs}/arrayfire/include
)

# Link the external ArrayFire library to the wrapper
target_link_libraries(arrayfire_wrapper PUBLIC afopencl)
