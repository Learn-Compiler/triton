import os
from pathlib import Path

os.environ.setdefault("TRITON_BACKENDS_IN_TREE", "1")

import triton
from triton.backends.compiler import GPUTarget
from triton.compiler import ASTSource


@triton.jit
def empty_kernel(X, Y):
    pass


source = ASTSource(empty_kernel, {"X": "*i32", "Y": "*i32"})
target = GPUTarget("coral", "coral-rv32", 1)
compiled = triton.compile(source, target=target)
print("\n####################### ttir ##########################")
print(compiled.asm["ttir"])
print("\n####################### llir ##########################")
print(compiled.asm["llir"])

llvm_ir = Path("/tmp/coral-ch02") / f"{compiled.metadata.name}.ll"
llvm_ir.parent.mkdir(parents=True, exist_ok=True)
llvm_ir.write_text(compiled.asm["llir"], encoding="utf-8")
print(f"Saved LLVM IR to {llvm_ir}")
