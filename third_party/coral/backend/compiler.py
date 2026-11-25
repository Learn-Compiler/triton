###-----------------------------------------------------------------------------
## Triton编译器核心
## 规范定义了定义了完整的add_stages流水线
## (TTIR优化->LLIR->ELF bytes二进制 以及元数据提取)
###-----------------------------------------------------------------------------
import hashlib
import re 
from dataclasses import dataclass
from typing import Any, Dict
from types import ModuleType

from triton.backends.compiler import BaseBackend, GPUTarget, Language
from triton._C.libtriton import ir, passes, llvm, coral 

@dataclass(frozen=True)
class CoralOptions:
  arch: str = "coral-rv32"
  num_warps: int = 1
  num_stages: int = 1
  debug: bool = False
  sanitize_overflow: bool = False

  def hash(self):
    key = "_".join([f"{k}-{v}" for k, v in self.__dict__.items()])
    return hashlib.sha256(key.encode("utf-8")).hexdigest()


class CoralBackend(BaseBackend):

  @staticmethod
  def supports_target(target: GPUTarget) -> bool:
    return (
      target.backend == "coral"
      and target.arch in ("coral-rv32", "rv32imf_zve32x")
      and target.warp_size == 1
    )

  def __init__(self, target: GPUTarget) -> None:
    super().__init__(target)
    # 声明最终二进制文件的后缀名
    self.binary_ext = "elf"

  def hash(self) -> str:
    return f"{self.target}"

  def parse_options(self, options: dict) -> CoralOptions:
    opts = {
      k: options[k]
      for k in CoralOptions.__dataclass_fields__.keys()
      if k in options and options[k] is not None
    } 
    return CoralOptions(**opts) 

  
  def load_dialects(self, context) -> None:
    # 后续加载Coral专有Dialect（tts, linalg）等
    pass 

  def pack_metadata(self, metadata):
    return (metadata.num_warps, getattr(metadata, "num_ctas", 1), getattr(metadata, "shared", 0))

  def get_codegen_implementation(self, options):
    return {}

  def get_module_map(self) -> Dict[str, ModuleType]:
    return {}

  # =========================================================================
  # Stage 流水线处理函数
  # =========================================================================
  @staticmethod 
  def make_ttir(mod, metadata, options):
    """阶段1: TTIR优化(内联 + 规范化) """
    pm = ir.pass_manager(mod.context)
    pm.enable_debug()
    passes.common.add_inliner(pm)
    pm.run(mod, "coral_make_ttir")

    # 提取kernel 函数名并记录到metadata(Triton缓存机制要求)
    mod_str = str(mod)
    names = re.findall(r"tt\.func\s+(?:public\s+)?@([a-zA-Z_][a-zA-Z0-9_]*)", mod_str)
    if names:
      metadata["name"] = names[0]
    else:
      metadata["name"] = "coral_kernel"

    metadata["num_warps"] = options.num_warps
    return mod

  @staticmethod
  def make_llir(src, metadata, options):
    '''阶段2: 降级到LLVM IR'''
    pm = ir.pass_manager(src.context)
    pm.enable_debug()
    coral.passes.add_to_llvmir(pm)
    # 在同一个 MLIR module 中把 TTIR 操作降低到 LLVM 方言。
    pm.run(src, "coral_make_llir")

    # 将 LLVM 方言翻译成 llc 可以读取的文本 LLVM IR。
    context = llvm.context()
    return str(llvm.to_module(src, context))


  @staticmethod
  def make_elf(src, metadata, options):
    '''阶段3: 最终二进制生成(铁律： 必须返回bytes类型)'''
    # 第一阶段先返回占位虚拟字节六，确保编译和缓存系统完整闭环
    dummy_elf = f"/* Coral RISC-V Binary: { metadata.get('name', 'kernel')} */\n".encode("utf-8")
    return dummy_elf


  def add_stages(self, stages: dict, options: CoralOptions, language: Language) -> None:
    # 按照顺序注册流水线
    stages["ttir"] = lambda src, metadata: self.make_ttir(src, metadata, options)
    stages["llir"] = lambda src, metadata: self.make_llir(src, metadata, options)
    stages["elf"]  = lambda src, metadata: self.make_elf(src, metadata, options)    
