/*
  Minsung Kim @ NXC, SNU.
  This file is part of the mzCache project.
*/

#include <Arrayfire_wrapper.h>

ArrayfireEnvironment::ArrayfireEnvironment() {
  af::setDevice(0);
  af::info();
}

ArrayfireEnvironment::~ArrayfireEnvironment() {
  af::deviceGC();
}

bool ArrayfireEnvironment::Init() {
    af::setDevice(0);
    af::info();
    
    #ifdef ANDROID_OPENCL
      af::setBackend(AF_BACKEND_OPENCL);
      std::cout << "Using OpenCL backend for ArrayFire on Android." << "\n";
    #else // set CUDA backend for non-Android platforms.
      std::cout << "Using CUDA backend for ArrayFire." << "\n";
      af::setBackend(AF_BACKEND_CUDA);
    #endif
    
    return true;
  // af::setMemoryStep(1024 * 1024); // Set memory step to 1MB
  }