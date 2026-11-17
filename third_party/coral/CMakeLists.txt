if(TRITON_BUILD_PYTHON_MODULE)
  add_triton_plugin(TritonCoral ${CMAKE_CURRENT_SOURCE_DIR}/triton_coral.cc)
  target_link_libraries(TritonCoral PRIVATE Python3::Module nanobind-static)
endif()
