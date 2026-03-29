"""
Helper script for setup_env.bat.
Handles all complex operations (path finding, file patching)
to avoid cmd.exe quoting nightmares.

Usage:
    python setup_helper.py zlib_paths     -> prints ZLIB_LIBRARY;ZLIB_INCLUDE_DIR;ZLIB_ROOT
    python setup_helper.py python_paths   -> prints PYTHON_INCLUDE;PYTHON_LIBRARY
    python setup_helper.py patch <dir>    -> patches setup.py in <dir>
    python setup_helper.py cmake_args     -> prints full CMAKE_ARGS string
"""
import os
import sys
import sysconfig


def zlib_paths():
    prefix = sys.prefix
    lib = os.path.join(prefix, "Library", "lib", "zlib.lib")
    inc = os.path.join(prefix, "Library", "include")
    root = os.path.join(prefix, "Library")
    print(f"{lib};{inc};{root}")


def python_paths():
    inc = sysconfig.get_path("include")
    # On Windows conda, python3XX.lib is in PREFIX/libs/
    prefix = sys.prefix
    ver = f"{sys.version_info.major}{sys.version_info.minor}"
    lib = os.path.join(prefix, "libs", f"python{ver}.lib")
    if not os.path.exists(lib):
        # Fallback: try to find any python*.lib
        libs_dir = os.path.join(prefix, "libs")
        if os.path.isdir(libs_dir):
            for f in os.listdir(libs_dir):
                if f.startswith("python") and f.endswith(".lib"):
                    lib = os.path.join(libs_dir, f)
                    break
    print(f"{inc};{lib}")


def cmake_args():
    prefix = sys.prefix
    zlib_lib = os.path.join(prefix, "Library", "lib", "zlib.lib")
    zlib_inc = os.path.join(prefix, "Library", "include")
    zlib_root = os.path.join(prefix, "Library")
    py_inc = sysconfig.get_path("include")
    ver = f"{sys.version_info.major}{sys.version_info.minor}"
    py_lib = os.path.join(prefix, "libs", f"python{ver}.lib")

    args = [
        f'-DZLIB_LIBRARY="{zlib_lib}"',
        f'-DZLIB_INCLUDE_DIR="{zlib_inc}"',
        f'-DCMAKE_PREFIX_PATH="{zlib_root}"',
    ]
    # Only add Python hints if the files actually exist
    if os.path.isdir(py_inc):
        args.append(f'-DPython_INCLUDE_DIR="{py_inc}"')
    if os.path.exists(py_lib):
        args.append(f'-DPython_LIBRARY="{py_lib}"')

    print(" ".join(args))


def patch(build_dir):
    # Patch 1: setup.py — fix CMake generator AND build command
    setup_py = os.path.join(build_dir, "setup.py")
    if not os.path.exists(setup_py):
        print(f"ERROR: {setup_py} not found", file=sys.stderr)
        sys.exit(1)

    with open(setup_py, "r", encoding="utf-8") as f:
        content = f.read()

    patched = False

    # Fix 1a: Keep "Unix Makefiles" generator — the libretro cores require $(MAKE)
    # which ONLY works with Unix Makefiles. Ninja/NMake/VS generators all fail.
    # We use MinGW make (mingw32-make) on Windows to support this.
    # No generator change needed — just keep "Unix Makefiles".
    # But fix any previous patches:
    for old_gen in ['"NMake Makefiles"', '"Ninja"', '"Visual Studio 16 2019"', '"Visual Studio 17 2022"']:
        if old_gen in content:
            content = content.replace(old_gen, '"MinGW Makefiles"')
            patched = True
            print(f"  Patched setup.py: {old_gen} -> MinGW Makefiles")
            break
    if '"Unix Makefiles"' in content:
        content = content.replace('"Unix Makefiles"', '"MinGW Makefiles"')
        patched = True
        print("  Patched setup.py: Unix Makefiles -> MinGW Makefiles")

    # Remove any -A x64 flags from previous VS generator patches
    content = content.replace('            "-A", "x64",\n', '')

    # Fix 1b: Build command — keep make but use mingw32-make
    for old_build in [
        '["make", jobs, "stable_retro"]',
        '["nmake", "stable_retro"]',
        '["cmake", "--build", ".", "--target", "stable_retro"]',
        '["cmake", "--build", ".", "--target", "stable_retro", "--config", "Release"]',
    ]:
        if old_build in content:
            content = content.replace(
                old_build,
                '["mingw32-make", jobs, "stable_retro"]'
            )
            if '"mingw32-make"' not in old_build:
                patched = True
                print("  Patched setup.py: build -> mingw32-make")
            break

    if patched:
        with open(setup_py, "w", encoding="utf-8") as f:
            f.write(content)
    else:
        print("  setup.py: already patched")

    # Patch 2: CMakeLists.txt — remove GCC-only flags that break MSVC
    # Line 46-47: -Wall -Wextra etc. are invalid for MSVC cl.exe
    cmakelists = os.path.join(build_dir, "CMakeLists.txt")
    if os.path.exists(cmakelists):
        with open(cmakelists, "r", encoding="utf-8") as f:
            content = f.read()

        gcc_flags_line = "-Wall -Wextra -Wno-sign-compare -Wno-missing-field-initializers -fvisibility=hidden"
        # Replace GCC flags with MSVC-compatible equivalents
        # /W3 is reasonable warning level, /wd4267 /wd4244 suppress sign-compare warnings
        msvc_flags = "/W3 /wd4267 /wd4244 /wd4305"
        if gcc_flags_line in content:
            content = content.replace(gcc_flags_line, msvc_flags)
            # Also fix -mssse3 (GCC flag, not needed on MSVC)
            content = content.replace("-mssse3", "")
            with open(cmakelists, "w", encoding="utf-8") as f:
                f.write(content)
            print("  Patched CMakeLists.txt: GCC flags -> MSVC flags")
        else:
            print("  CMakeLists.txt: already patched or pattern not found")
    else:
        print("  WARNING: CMakeLists.txt not found")


def verify_prereqs():
    """Print diagnostic info about the build environment."""
    prefix = sys.prefix
    ver = f"{sys.version_info.major}{sys.version_info.minor}"

    checks = {
        "Python prefix": prefix,
        "Python include": sysconfig.get_path("include"),
        "Python lib": os.path.join(prefix, "libs", f"python{ver}.lib"),
        "zlib.lib": os.path.join(prefix, "Library", "lib", "zlib.lib"),
        "zlib include": os.path.join(prefix, "Library", "include", "zlib.h"),
    }

    all_ok = True
    for name, path in checks.items():
        exists = os.path.exists(path)
        status = "OK" if exists else "MISSING"
        print(f"  {status}: {name} -> {path}")
        if not exists:
            all_ok = False

    if not all_ok:
        print("\n  WARNING: Some prerequisites are missing!")
        print("  Try: conda install zlib python -y")
    return all_ok


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python setup_helper.py [zlib_paths|python_paths|cmake_args|patch <dir>|verify]")
        sys.exit(1)

    cmd = sys.argv[1]
    if cmd == "zlib_paths":
        zlib_paths()
    elif cmd == "python_paths":
        python_paths()
    elif cmd == "cmake_args":
        cmake_args()
    elif cmd == "patch":
        if len(sys.argv) < 3:
            print("ERROR: patch requires a directory argument", file=sys.stderr)
            sys.exit(1)
        patch(sys.argv[2])
    elif cmd == "verify":
        verify_prereqs()
    else:
        print(f"Unknown command: {cmd}", file=sys.stderr)
        sys.exit(1)
