#!/usr/bin/env bash
set -euo pipefail

cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../../../.."
sdk_bin="${LLVM_SYSPATH:-../coralnpu-compiler/out/llvm-zvt/install}/bin"

echo "------------ LLVM IR ------------"
TRITON_ALWAYS_COMPILE=1 .venv/bin/python third_party/coral/unittest/L02/empty.py
echo "------------   ASM  ------------"
"$sdk_bin/llc" -mattr=+m,+f -target-abi=ilp32 -filetype=asm \
  /tmp/coral-ch02/empty_kernel.ll -o /tmp/coral-ch02/empty_kernel.s
cat /tmp/coral-ch02/empty_kernel.s
echo "------------ OBJDUMP ------------"
"$sdk_bin/llc" -mattr=+m,+f -target-abi=ilp32 -filetype=obj \
  /tmp/coral-ch02/empty_kernel.ll -o /tmp/coral-ch02/empty_kernel.o
"$sdk_bin/llvm-objdump" -d -M no-aliases /tmp/coral-ch02/empty_kernel.o
