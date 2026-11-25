include "mlir/Pass/PassBase.td"

def ConvertTritonCoralToLLVM
    : Pass<"convert-triton-coral-to-llvm", "mlir::ModuleOp"> {
      let summary = "Convert scalar Coral TTIR to the LLVM dialect";
      let dependentDialects = ["mlir::LLVM::LLVMDialect"];
}
