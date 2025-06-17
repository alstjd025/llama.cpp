/*
  Minsung Kim @ NXC, SNU.
  This file is part of the mzCache project.
  ##################################################################################
  Header for ArrayFire wrapper functions which are C++ APIs for ArrayFire usage in
  llama.cpp.
  ##################################################################################
*/

// Includes
#include <arrayfire.h>
#include <iostream>


// Predifined macros
#ifndef ANDROID_OPENCL
  #define ANDROID_OPENCL
#endif


class ArrayfireEnvironment {
public:
  ArrayfireEnvironment(); 
  ~ArrayfireEnvironment(); 

  bool Init();
};