#include "TritonCoralToLLVM/TritonCoralToLLVM.h"

#include "mlir/Conversion/FuncToLLVM/ConvertFuncToLLVM.h"
#include "mlir/Conversion/LLVMCommon/ConversionTarget.h"
#include "mlir/Conversion/LLVMCommon/LoweringOptions.h"
#include "mlir/Conversion/LLVMCommon/TypeConverter.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/IR/BuiltinOps.h"
#include "triton/Dialect/Triton/IR/Dialect.h"
#include "llvm/IR/DataLayout.h"

#include <optional>
#include <utility>

namespace mlir::triton::coral {
#define GEN_PASS_DEF_CONVERTTRITONCORALTOLLVM
#include "TritonCoralToLLVM/Passes.h.inc"

namespace tt = mlir::triton;

namespace {
// This matches the locked Coral SDK for riscv32-unknown-elf, ILP32.
constexpr llvm::StringLiteral coralTriple = "riscv32-unknown-unknown-elf";
constexpr llvm::StringLiteral coralDataLayout = "e-m:e-p:32:32-i64:64-n32-s128";

struct FuncLowering : OpConversionPattern<tt::FuncOp> {
  using OpConversionPattern::OpConversionPattern;

  LogicalResult
  matchAndRewrite(tt::FuncOp op, OpAdaptor,
                  ConversionPatternRewriter &rewriter) const override {
    if (!op.isPublic() || op.isExternal() || !op.getResultTypes().empty())
      return rewriter.notifyMatchFailure(op, "expected a public void kernel");

    const auto &converter = *getTypeConverter<LLVMTypeConverter>();
    auto function = convertFuncOpToLLVMFuncOp(op, rewriter, converter);

    if (failed(function))
      return failure();

    rewriter.eraseOp(op);
    return success();
  }
};

struct ReturnLowering : OpConversionPattern<tt::ReturnOp> {
  using OpConversionPattern::OpConversionPattern;

  LogicalResult
  matchAndRewrite(tt::ReturnOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    if (!adaptor.getOperands().empty())
      return rewriter.notifyMatchFailure(op, "expected a void return");

    rewriter.replaceOpWithNewOp<LLVM::ReturnOp>(op, ValueRange{});
    return success();
  }
};

struct ConvertTritonCoralToLLVM
    : impl::ConvertTritonCoralToLLVMBase<ConvertTritonCoralToLLVM> {
  void runOnOperation() override {
    MLIRContext &context = getContext();
    ModuleOp module = getOperation();
    LowerToLLVMOptions options(&context);
    options.dataLayout = llvm::DataLayout(coralDataLayout);
    LLVMTypeConverter converter(&context, options);
    converter.addConversion([](tt::PointerType type) -> std::optional<Type> {
      if (type.getAddressSpace() != tt::PtrAddrSpace::Global ||
          !type.getPointeeType().isInteger(32))
        return std::nullopt;
      return LLVM::LLVMPointerType::get(type.getContext());
    });

    LLVMConversionTarget target(context);
    target.addLegalOp<ModuleOp>();
    target.addIllegalDialect<tt::TritonDialect>();
    RewritePatternSet patterns(&context);
    patterns.add<FuncLowering, ReturnLowering>(converter, &context);

    if (failed(applyFullConversion(module, target, std::move(patterns)))) {
      signalPassFailure();
      return;
    }

    StringAttr triple = StringAttr::get(&context, coralTriple);
    StringAttr layout = StringAttr::get(&context, coralDataLayout);
    module->setAttr(LLVM::LLVMDialect::getTargetTripleAttrName(), triple);
    module->setAttr(LLVM::LLVMDialect::getDataLayoutAttrName(), layout);
  }
};

} // namespace

} // namespace mlir::triton::coral
