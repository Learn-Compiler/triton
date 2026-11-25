add_triton_library(TritonCoralToLLVM
  TritonCoralToLLVM.cpp

  DEPENDS
  TritonCoralConversionPassIncGen

  LINK_LIBS
  TritonIR
  MLIRFuncToLLVM
  MLIRLLVMCommonConversion
  MLIRLLVMDialect
  MLIRTransforms
)
