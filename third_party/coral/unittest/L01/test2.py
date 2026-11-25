from triton.backends import backends
from triton.backends.compiler import GPUTarget

accepts = backends["coral"].compiler.supports_target
assert accepts(GPUTarget("coral", "coral-rv32", 1))
assert accepts(GPUTarget("coral", "rv32imf_zve32x", 1))
assert not accepts(GPUTarget("coral", "coral-invalid", 1))
assert not accepts(GPUTarget("coral", "coral-rv32", 32))
print("PASS: Coral 目标检查通过")

