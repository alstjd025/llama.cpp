# set(TARGET llama-cli)
# add_executable(${TARGET} main.cpp)
# install(TARGETS ${TARGET} RUNTIME)
# target_link_libraries(${TARGET} PRIVATE common llama ${CMAKE_THREAD_LIBS_INIT})
# target_compile_features(${TARGET} PRIVATE cxx_std_17)



# llama.cpp/examples/main/CMakeLists.txt
add_executable(llama-cli
    main.cpp
)

# include 디렉토리 설정 (main.cpp 내에서 arrayfire wrapper 헤더를 사용할 수 있게)
target_include_directories(llama-cli PRIVATE
    ${CMAKE_SOURCE_DIR}/tools/arrayfire
)

# 필요한 라이브러리 링크
target_link_libraries(llama-cli
    PRIVATE
    common
    llama
    arrayfire_wrapper
    ${CMAKE_THREAD_LIBS_INIT}
)

target_compile_features(llama-cli PRIVATE cxx_std_17)
install(TARGETS llama-cli RUNTIME)



