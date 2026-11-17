###-----------------------------------------------------------------------------
## Triton运行时驱动
###-----------------------------------------------------------------------------
from __future__ import annotations
import os
from triton.backends.compiler import GPUTarget
from triton.backends.driver import DriverBase


class CoralDriver(DriverBase):

    def __init__(self):
        super().__init__()

    @classmethod
    def is_active(cls) -> bool:
        # 当检测到Coral 仿真器或硬件运行时返回True，或通过 TRITON_DEFAULT_BACKEND=coral 显式指定
        return os.environ.get("TRITON_DEFAULT_BACKEND", "") == "coral" or True

    def get_current_target(self) -> GPUTarget:
        # Coral NPU是 32-bit RISCV(RV32IMF_Zve32x)
        return GPUTarget("coral", arch="rv32imf_zve32x", warp_size=1)

    def map_python_to_cpp_type(self, ty: str) -> str:
        mapping = {
            "i1": "bool",
            "i8": "int8_t",
            "i16": "int16_t",
            "i32": "int32_t",
            "i64": "int64_t",
            "u1": "bool",
            "u8": "uint8_t",
            "u16": "uint16_t",
            "u32": "uint32_t",
            "u64": "uint64_t",
            "fp16": "uint16_t",
            "bf16": "uint16_t",
            "fp32": "float",
            "fp64": "double",
        }
        if ty.startswith("*"):
            return "void*"
        return mapping.get(ty, "void*")

    def get_active_torch_device(self):
        try:
            import torch
            return torch.device("cpu")
        except Exception:
            return None

    def get_benchmarker(self):
        def _benchmark(kernel_call, *, quantiles=None, **kwargs):
            return [0.0] * (len(quantiles) if quantiles else 1)

        return _benchmark
