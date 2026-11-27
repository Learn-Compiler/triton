#include "TritonCoralToLLVM/TritonCoralToLLVM.h"

#include "mlir/Conversion/FuncToLLVM/ConvertFuncToLLVM.h"
#include "mlir/Conversion/LLVMCommon/ConversionTarget.h"
#include "mlir/Conversion/LLVMCommon/LoweringOptions.h"
#include "mlir/Conversion/LLVMCommon/TypeConverter.h"
#include "mlir/Conversion/ArithToLLVM/ArithToLLVM.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
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
    // The standard conversion rewrites the signature and moves the body.
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

// 签名转换和访存pattern共用这条约束
bool isGlobalI32Pointer(Type type) {
  auto pointer = dyn_cast<tt::PointerType>(type);
  return pointer && pointer.getAddressSpace() == tt::PtrAddrSpace::Global &&
         pointer.getPointeeType().isInteger(32);
}


struct LoadLowering : OpConversionPattern<tt::LoadOp> {
  using OpConversionPattern::OpConversionPattern;

  LogicalResult
    matchAndRewrite(tt::LoadOp op, OpAdaptor adaptor, 
                    ConversionPatternRewriter &rewriter) const override {
    if(!isGlobalI32Pointer(op.getPtr().getType()))
      return rewriter.notifyMatchFailure(op, "expected a global scalar i32 pointer");
    if(op.getMask() || op.getOther() || op.getCachePolicy() || op.getIsVolatile()) 
      return rewriter.notifyMatchFailure(op, "unsupported Coral scalar load options");
    
    // 用转换后的指针读取i32. 地址只保证1字节对齐。
    rewriter.replaceOpWithNewOp<LLVM::LoadOp>(
      op, op.getResult().getType(), adaptor.getPtr(), /*aligment*/1
    );
    return success();
  }
};


struct StoreLowering : OpConversionPattern<tt::StoreOp> {
  using OpConversionPattern::OpConversionPattern;

  LogicalResult 
  matchAndRewrite(tt::StoreOp op, OpAdaptor adaptor,
                    ConversionPatternRewriter &rewriter) const override {
    if(!isGlobalI32Pointer(op.getPtr().getType()))
      return rewriter.notifyMatchFailure(op, "expected a global scalar i32 pointer");
    if(op.getMask() || op.getCachePolicy() || op.getIgnoreCta())
      return rewriter.notifyMatchFailure(op, "unsupported Coral scalar store options");
    
    // adaptor 中的值和地址以及重映射， store接到改写后的load结果
    rewriter.replaceOpWithNewOp<LLVM::StoreOp>(
      op, adaptor.getValue(), adaptor.getPtr(), /*alignment=*/1
    );
    return success();
  }
};


struct ConvertTritonCoralToLLVM
    : impl::ConvertTritonCoralToLLVMBase<ConvertTritonCoralToLLVM> {
  void runOnOperation() override {
    MLIRContext &context = getContext();
    ModuleOp module = getOperation();

    // 标准转换库认识减法、浮点和额外类型， 只开放标量i32常量和无溢出承诺的加法
    // 先约束本节开放的Arith子集，再交给标准转换库
    WalkResult arithmetic = module.walk([](Operation *op) {
      if(op->getName().getDialectNamespace() != arith::ArithDialect::getDialectNamespace())
        return WalkResult::advance();
      if(auto constant = dyn_cast<arith::ConstantOp>(op)) {
        if(constant.getType().isInteger(32))
          return WalkResult::advance();
      }

      // 普通i32加法保留回绕语义，检查原操作的溢出标志
      if(auto add = dyn_cast<arith::AddIOp>(op)) {
        if(add.getType().isInteger(32) && 
           add.getOverflowFlags() == arith::IntegerOverflowFlags::none) 
          return WalkResult::advance();
      }
      op->emitOpError("Coral scalar arithmetic support i32 constant and unflagged addi");
      return WalkResult::advance();
    });
    if(arithmetic.wasInterrupted()) {
      signalPassFailure();
      return;
    }

    LowerToLLVMOptions options(&context);
    options.dataLayout = llvm::DataLayout(coralDataLayout);
    LLVMTypeConverter converter(&context, options);
    // This stage lowers global i32 pointers to opaque LLVM pointers
    // in address space zero.
    converter.addConversion([](tt::PointerType type) -> std::optional<Type> {
      if (type.getAddressSpace() != tt::PtrAddrSpace::Global ||
          !type.getPointeeType().isInteger(32))
        return std::nullopt;
      return LLVM::LLVMPointerType::get(type.getContext());
    });

    LLVMConversionTarget target(context);
    target.addLegalOp<ModuleOp>();
    // Full conversion reports any TTIR op that lacks a lowering pattern.
    target.addIllegalDialect<tt::TritonDialect>();
    RewritePatternSet patterns(&context);
    patterns.add<FuncLowering, ReturnLowering, LoadLowering, StoreLowering>(converter, &context);
    target.addIllegalDialect<arith::ArithDialect>();
    // 注册已有Atirth pattern, 将通过检查的常量和加法改写为LLVM操作
    arith::populateArithToLLVMConversionPatterns(converter, patterns);

    if (failed(applyFullConversion(module, target, std::move(patterns)))) {
      signalPassFailure();
      return;
    }

    // Preserve the RV32 target and layout when exporting LLVM IR.
    StringAttr triple = StringAttr::get(&context, coralTriple);
    StringAttr layout = StringAttr::get(&context, coralDataLayout);
    module->setAttr(LLVM::LLVMDialect::getTargetTripleAttrName(), triple);
    module->setAttr(LLVM::LLVMDialect::getDataLayoutAttrName(), layout);
  }
};


} // namespace

} // namespace mlir::triton::coral
