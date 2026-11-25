import re

import pytest
import triton
from triton.backends import backends
from triton.backends.compiler import GPUTarget
from triton.compiler import ASTSource


@triton.jit
def empty_kernel(X, Y):
    pass


@pytest.mark.skipif("coral" not in backends, reason="Coral backend is unavailable")
def test_coral_empty_kernel_lowers_to_llvm(monkeypatch, tmp_path):
    monkeypatch.setenv("TRITON_CACHE_DIR", str(tmp_path))
    source = ASTSource(empty_kernel, {"X": "*i32", "Y": "*i32"})
    compiled = triton.compile(source, target=GPUTarget("coral", "coral-rv32", 1))

    assert "tt.return" in compiled.asm["ttir"]
    llvm_ir = compiled.asm["llir"]
    assert 'target triple = "riscv32-unknown-unknown-elf"' in llvm_ir
    assert re.search(r"define void @empty_kernel\(ptr %[^,]+, ptr %[^)]+\) \{\s+ret void\s+\}", llvm_ir)
