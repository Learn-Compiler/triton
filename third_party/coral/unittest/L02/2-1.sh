#!/usr/bin/env bash
set -euo pipefail

# 回到 Triton 仓库根目录，让虚拟环境和 SDK 的相对路径从同一处解析。
cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../../../.."
sdk_bin="${LLVM_SYSPATH:-../coralnpu-compiler/out/llvm-zvt/install}/bin"
kernel=empty
echo "------------ LLVM IR ------------"
# 强制重编空 kernel，拿到当前转换 pass 生成的 LLVM IR。
TRITON_ALWAYS_COMPILE=1 .venv/bin/python third_party/coral/unittest/L02/${kernel}.py
echo "------------   ASM  ------------"
set -x
"$sdk_bin/llc" -mattr=+m,+f -target-abi=ilp32 -filetype=asm \
  /tmp/coral-ch02/${kernel}_kernel.ll -o /tmp/coral-ch02/${kernel}_kernel.s
cat /tmp/coral-ch02/${kernel}_kernel.s
echo "------------ OBJDUMP ------------"
"$sdk_bin/llc" -mattr=+m,+f -target-abi=ilp32 -filetype=obj \
  /tmp/coral-ch02/${kernel}_kernel.ll -o /tmp/coral-ch02/${kernel}_kernel.o
# 关闭指令别名，从机器码确认 ret 对应的 jalr。
"$sdk_bin/llvm-objdump" -d -M no-aliases /tmp/coral-ch02/${kernel}_kernel.o
