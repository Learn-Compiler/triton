add_triton_library(TritonCoralToLLVM
  TritonCoralToLLVM.cpp

  DEPENDS
  TritonCoralConversionPassIncGen

  LINK_LIBS
  TritonIR
  MLIRArithToLLVM
  MLIRFuncToLLVM
  MLIRLLVMCommonConversion
  MLIRLLVMDialect
  MLIRTransforms
)
