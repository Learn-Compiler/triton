# Coral LLVM toolchain pin

This branch uses a source-built Coral SDK, not Triton's upstream prebuilt LLVM.
`cmake/llvm-info.json` is the build lock; `cmake/llvm-build-info.json`
records the same base revision for tooling. The AMD code-generation library has
its own pin in `cmake/amd-llvm-info.json`; do not replace that pin with Coral LLVM.

- LLVM repository: `iree-org/llvm-project`
- LLVM commit: `6cce5bca3b901ec32e250fb43e88eb0fe511af93`
- LLVM version: `24.0.0git`
- Coral compiler repository: `Learn-Compiler/coralnpu-compiler`
- Coral compiler commit: `5c68026757910f4764cf9660e406ab6f03829821`
- Zvt patch: `llvm-project-0001-add-zvt-support.patch`
- Patch SHA256: `47809e7cdd227e053878610bcf5a3fdbde756da97f5e85db2ca434130c52c9a0`
- Required targets: X86, RISCV, NVPTX, AMDGPU; assertions ON, RTTI OFF, EH OFF.

## Build the SDK

Use a separate, clean checkout of the pinned Coral compiler revision. Initialize
its `third_party/llvm-project` submodule, verify the patch checksum above, then
apply the patch with `git apply --check` followed by `git apply`. Do not reset an
existing checkout that contains local work. From the Coral compiler root:

```sh
SRC="$PWD/third_party/llvm-project"
OUT="$PWD/out/llvm-zvt"
cmake -S "$SRC/llvm" -B "$OUT/build" -G Ninja \
  -DCMAKE_BUILD_TYPE=Release -DCMAKE_INSTALL_PREFIX="$OUT/install" \
  -DCMAKE_C_COMPILER=clang -DCMAKE_CXX_COMPILER=clang++ \
  '-DLLVM_ENABLE_PROJECTS=clang;mlir;lld' \
  '-DLLVM_TARGETS_TO_BUILD=X86;RISCV;NVPTX;AMDGPU' \
  -DLLVM_ENABLE_ASSERTIONS=ON -DLLVM_ENABLE_RTTI=OFF -DLLVM_ENABLE_EH=OFF \
  -DLLVM_USE_LINKER=lld -DMLIR_ENABLE_BINDINGS_PYTHON=OFF \
  -DLLVM_INCLUDE_TESTS=ON -DLLVM_BUILD_TESTS=OFF \
  -DMLIR_INCLUDE_TESTS=OFF -DLLVM_PARALLEL_LINK_JOBS=2
cmake --build "$OUT/build" --parallel 24
cmake --install "$OUT/build"
cmake --build "$OUT/build" --target FileCheck not count split-file
for tool in FileCheck not count split-file; do
  install -m 755 "$OUT/build/bin/$tool" "$OUT/install/bin/$tool"
done
```

The compiler-1 SDK is already installed at
`/home/compiler-1/work/coral/coralnpu-compiler/out/llvm-zvt/install`.
Its additional local Zvt disassembler registration fix is independent of the
base revision and the original Zvt patch recorded here.

## Build Triton

Always specify `LLVM_SYSPATH`. Build helpers check the SDK's Git revision,
version, assertions, RTTI, targets, and assemble Zvt instructions. CMake also
rejects package and tablegen paths from another SDK, including stale cache
entries. A version string alone is insufficient. These checks establish
compatibility; they do not certify every local LLVM source modification.
There is no published Coral LLVM archive, so implicit upstream downloads are
disabled. The generic upstream LLVM build script stops with a pointer here.

For an editable Python build, install the normal repository build requirements
(including nanobind), then run from this repository's root:

```sh
export LLVM_SYSPATH=/path/to/coralnpu-compiler/out/llvm-zvt/install
export TRITON_BUILD_DIR="$PWD/build/coral-llvm24"
python -m pip install --no-build-isolation -e .
make BUILD_DIR="$TRITON_BUILD_DIR" PYTHON="$(command -v python)"
```

Direct CMake builds must also pass `-DLLVM_SYSPATH="$LLVM_SYSPATH"`.
Use a fresh build directory when switching SDKs. For offline builds, supply
`JSON_SYSPATH`, CUDA dependencies, and the independently pinned AMD codegen
library/cache as required by the existing build helpers.

On compiler-1 the validated build directory is
`/home/compiler-1/work/coral/coralnpu-compiler/out/llvm-zvt/triton-main-validation/build`.
Rebuild it from this repository with:

```sh
make BUILD_DIR=/home/compiler-1/work/coral/coralnpu-compiler/out/llvm-zvt/triton-main-validation/build \
  PYTHON="$PWD/.venv/bin/python"
```

## Backend status

The LLVM 24 NVIDIA unsigned-high-multiply adaptation is included in
`third_party/nvidia/lib/TritonNVIDIAGPUToLLVM/ElementwiseOpToLLVM.cpp`.
The conversion regression checks cover its 32-bit and 64-bit intrinsic lowering
through PTX generation. The compiler-1 build uses this committed adaptation.
Only `third_party/coral/` is excluded from the current commits and remains local.
No GPU execution or Coral hardware execution has been validated by this build.
