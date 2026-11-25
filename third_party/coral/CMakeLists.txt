if(TRITON_BUILD_PYTHON_MODULE)
  include_directories(${CMAKE_CURRENT_BINARY_DIR}/include)
  include_directories(${CMAKE_CURRENT_SOURCE_DIR}/include)
  add_subdirectory(include/TritonCoralToLLVM)
  add_subdirectory(lib/TritonCoralToLLVM)

  add_triton_plugin(TritonCoral
    ${CMAKE_CURRENT_SOURCE_DIR}/triton_coral.cc

    DEPENDS
    TritonCoralConversionPassIncGen

    LINK_LIBS
    TritonCoralToLLVM
  )
  target_link_libraries(TritonCoral PRIVATE Python3::Module nanobind-static)
endif()
