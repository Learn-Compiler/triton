import os
os.environ.setdefault("TRITON_DEFAULT_BACKEND", "coral")

import triton
from triton.backends import backends
print('1. 注册的后端列表:', list(backends.keys()))

from triton._C.libtriton import coral
print('2. C++ 接口调用:', coral.hello())

import triton.language as tl
from triton.compiler import ASTSource, compile

@triton.jit
def add_kernel(x_ptr, y_ptr, BLOCK: tl.constexpr):
    x = tl.load(x_ptr)
    tl.store(y_ptr, x)

src = ASTSource(
    fn=add_kernel,
    signature={"x_ptr": "*fp32", "y_ptr": "*fp32"},
    constexprs={"BLOCK": 64},
)

compiled = compile(src, target=triton.runtime.driver.active.get_current_target())
print('3. 端到端编译成功! Kernel 名称:', compiled.name)
print('4. 生成的二进制 payload:', compiled.asm['elf'][:40])
