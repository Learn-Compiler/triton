import os
from pathlib import Path

# Triton 在导入时发现后端，这里让它扫描当前源码树里的 Coral。
os.environ.setdefault("TRITON_BACKENDS_IN_TREE", "1")

import triton
from triton.backends.compiler import GPUTarget
from triton.compiler import ASTSource


# 保留两个设备指针，先验证函数签名和空返回的转换。
@triton.jit
def empty_kernel(X, Y):
    pass


# ASTSource 提供参数类型，编译时可以直接检查生成的 IR。
source = ASTSource(empty_kernel, {"X": "*i32", "Y": "*i32"})
target = GPUTarget("coral", "coral-rv32", 1)
compiled = triton.compile(source, target=target)
print("\n####################### ttir ##########################")
print(compiled.asm["ttir"])
print("\n####################### llir ##########################")
print(compiled.asm["llir"])

# 用 kernel 名保存 LLVM IR，下一步把这个文件交给 llc。
llvm_ir = Path("/tmp/coral-ch02") / f"{compiled.metadata.name}.ll"
llvm_ir.parent.mkdir(parents=True, exist_ok=True)
llvm_ir.write_text(compiled.asm["llir"], encoding="utf-8")
print(f"Saved LLVM IR to {llvm_ir}")
