#ifndef TRITON_CORAL_TO_LLVM_H
#define TRITON_CORAL_TO_LLVM_H

#include "mlir/Pass/Pass.h"

namespace mlir::triton::coral {

#define GEN_PASS_DECL
#include "TritonCoralToLLVM/Passes.h.inc"

} // namespace mlir::triton::coral

#endif // TRITON_CORAL_TO_LLVM_H
