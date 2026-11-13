from __future__ import annotations

import argparse
import hashlib
import io
import json
import pytest
import tarfile
import tempfile

from pathlib import Path

import triton

from triton.runtime.build import compile_module_from_src

TEST_MODULE_C = """
#include <Python.h>
#include <string.h>

static PyObject* go(PyObject* self, PyObject* args) {
    const char *command;
    if (!PyArg_ParseTuple(args, "s", &command))
        return NULL;

    const char* res;
    if (strcmp(command, "hello") == 0) {
        res = "hiya";
    } else {
        res = "huh";
    }
    return PyUnicode_FromString(res);
}

static PyMethodDef ModuleMethods[] = {
  {"go", go, METH_VARARGS, "test_module.go for testing"},
  {NULL, NULL, 0, NULL}
};

static struct PyModuleDef ModuleDef = {
  PyModuleDef_HEAD_INIT,
  "test_module",
  NULL, //documentation
  -1, //size
  ModuleMethods
};

PyMODINIT_FUNC PyInit_test_module(void) {
  PyObject *m = PyModule_Create(&ModuleDef);
  if(m == NULL) {
    return NULL;
  }
  PyModule_AddFunctions(m, ModuleMethods);
  return m;
}
"""


def test_compile_module(fresh_triton_cache):
    mod = compile_module_from_src(TEST_MODULE_C, "test_module")

    with pytest.raises(Exception):
        mod.go()

    assert mod.go("huh") == "huh"
    assert mod.go("hello") == "hiya"

    # Make sure the module is cached
    mod2 = compile_module_from_src(TEST_MODULE_C, "test_module")
    assert mod2.__file__ == mod.__file__


def test_compile_module_bad_cache(fresh_knobs):
    with tempfile.TemporaryDirectory() as tmpd:
        tmp = Path(tmpd)
        called_get_file = False

        class InvalidFileCacheManager(triton.runtime.cache.FileCacheManager):

            def get_file(self, filename: str) -> str | None:
                nonlocal called_get_file
                called_get_file = True
                (tmp / filename).write_text("not an so")
                return str(tmp / filename)

        # First corrupt the cache
        fresh_knobs.cache.manager_class = InvalidFileCacheManager

        mod = compile_module_from_src(TEST_MODULE_C, "test_module")
        assert called_get_file

        with pytest.raises(Exception):
            mod.go()

        assert mod.go("huh") == "huh"
        assert mod.go("hello") == "hiya"


@pytest.mark.parametrize("artifact", ["bootstrap", "codegen"])
@pytest.mark.parametrize(
    "mirror", ["", "https://mirror.example/generic/oaiartifacts", "https://mirror.example/generic/oaiartifacts/"])
def test_amd_codegen_download_mirror(artifact, mirror, monkeypatch, tmp_path):
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[3]))
    import build_helpers

    monkeypatch.setenv("OAIARTIFACTS_BASE_URL", mirror)
    monkeypatch.delenv("TRITON_CACHE_DEPENDENCY_DOWNLOADS", raising=False)
    monkeypatch.setattr(build_helpers, "get_base_dir", lambda: str(tmp_path))
    parser = argparse.ArgumentParser()
    build_helpers.add_common_args(parser)
    args = build_helpers.normalize_parsed_args(
        parser.parse_args([
            "--triton-cache-path",
            str(tmp_path / "cache"),
            "--triton-llvm-system-suffix",
            "ubuntu-arm64",
            "--llvm-syspath",
            str(tmp_path / "unrelated-core-llvm"),
        ]))

    revision = "123456789abcdef"
    name = f"{'llvm' if artifact == 'bootstrap' else 'amd-codegen'}-{revision[:8]}-ubuntu-arm64-1"
    relative_path = "lib/cmake/llvm/LLVMConfig.cmake" if artifact == "bootstrap" else (
        f"lib/{build_helpers.amd_codegen_library_name()}")
    contents = b"test artifact"
    buffer = io.BytesIO()
    with tarfile.open(fileobj=buffer, mode="w:gz") as archive:
        member = tarfile.TarInfo(f"{name}/{relative_path}")
        member.size = len(contents)
        archive.addfile(member, io.BytesIO(contents))
    archive_bytes = buffer.getvalue()
    checksum = hashlib.sha256(archive_bytes).hexdigest()
    llvm_info = {"llvm_hash": revision, "build_number": 1, "sha256sum": {"ubuntu-arm64": checksum}}
    if artifact == "bootstrap":
        llvm_info["bootstrap_llvm"] = {"build_number": 1, "sha256sum": llvm_info.pop("sha256sum")}

    downloads = []

    def download(url, path, label):
        downloads.append(url)
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        Path(path).write_bytes(archive_bytes)

    monkeypatch.setattr(build_helpers, "_download_file", download)
    if artifact == "bootstrap":
        result = Path(build_helpers.download_codegen_llvm("amd", llvm_info, args)) / relative_path
    else:
        (tmp_path / "cmake").mkdir()
        (tmp_path / "cmake" / "amd-llvm-info.json").write_text(json.dumps(llvm_info))
        build_helpers.download_and_copy_amd_codegen(args)
        result = tmp_path / "third_party" / "amd" / "backend" / relative_path

    base = "https://mirror.example/generic/triton-llvm" if mirror else (
        "https://oaitriton.blob.core.windows.net/public/llvm-builds")
    assert downloads == [f"{base}/{name}.tar.gz"]
    assert result.read_bytes() == contents


@pytest.fixture
def coral_llvm_sdk(monkeypatch, tmp_path):
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[3]))
    import build_helpers

    info = json.loads((Path(build_helpers.get_base_dir()) / "cmake/llvm-info.json").read_text())
    prefix = tmp_path / "llvm"
    header = prefix / "include/llvm/Support/VCSRevision.h"
    header.parent.mkdir(parents=True)
    header.write_text(f'#define LLVM_REVISION R"({info["llvm_hash"]})"\n')
    return build_helpers, prefix, info


@pytest.mark.parametrize("failure", [None, "revision", "version", "assertions", "rtti", "targets", "zvt"])
def test_coral_llvm_sdk_validation(coral_llvm_sdk, monkeypatch, failure):
    helpers, prefix, info = coral_llvm_sdk
    if failure == "revision":
        (prefix / "include/llvm/Support/VCSRevision.h").write_text('#define LLVM_REVISION "wrong"')
    configuration = [info["version"], "ON", "NO", "X86 RISCV NVPTX AMDGPU"]
    if failure in ["version", "assertions", "rtti", "targets"]:
        configuration[["version", "assertions", "rtti", "targets"].index(failure)] = "wrong"
    monkeypatch.setattr(helpers.subprocess, "check_output", lambda *args, **kwargs: "\n".join(configuration))
    commands = []

    def assemble(command, **kwargs):
        commands.append((command, kwargs))
        if failure == "zvt":
            raise helpers.subprocess.CalledProcessError(1, command, stderr="Zvt instruction requires extension")

    monkeypatch.setattr(helpers.subprocess, "run", assemble)
    if failure is not None:
        with pytest.raises(RuntimeError, match="Coral LLVM"):
            helpers.verify_coral_llvm(prefix, info)
    else:
        helpers.verify_coral_llvm(prefix, info)
        assert commands[0][0][0] == str(prefix / "bin/llvm-mc")
        assert "+zvtbase" in commands[0][0][2]
        assert "vtmms.tvv" in commands[0][1]["input"]


def test_coral_llvm_requires_explicit_sdk(coral_llvm_sdk, monkeypatch):
    helpers, _, info = coral_llvm_sdk
    monkeypatch.delenv("LLVM_SYSPATH", raising=False)
    with pytest.raises(RuntimeError, match="LLVM_SYSPATH"):
        helpers.verify_coral_llvm(None, info)
    with pytest.raises(RuntimeError, match="No Coral LLVM archive"):
        helpers.get_llvm_archive("ubuntu-x64", info)
    parser = argparse.ArgumentParser()
    helpers.add_common_args(parser)
    args = helpers.normalize_parsed_args(parser.parse_args(["--triton-cache-path", "/unused"]))
    with pytest.raises(RuntimeError, match="LLVM_SYSPATH"):
        helpers.get_llvm_package_info(args)


@pytest.mark.parametrize("variable", ["LLVM_DIR", "MLIR_DIR", "LLD_DIR", "LLVM_LIBRARY_DIR", "LLVM_TABLEGEN_EXE"])
def test_coral_llvm_rejects_mixed_cmake_paths(coral_llvm_sdk, monkeypatch, tmp_path, variable):
    helpers, prefix, _ = coral_llvm_sdk
    monkeypatch.setattr(helpers, "get_thirdparty_cmake_vars", lambda *args: {variable: str(prefix / "expected")})
    generated = tmp_path / "paths.cmake"
    helpers.write_thirdparty_cmake_vars(str(generated), ["llvm"], None)
    # Matching paths must work; an explicit path into another SDK must fail.
    for path, succeeds in [(prefix / "expected", True), (tmp_path / "old-llvm", False)]:
        result = helpers.subprocess.run(
            ["cmake", f"-D{variable}={path}", "-P", str(generated)], capture_output=True, text=True)
        assert (result.returncode == 0) == succeeds
        if not succeeds:
            assert f"{variable} conflicts with pinned LLVM_SYSPATH" in result.stderr
