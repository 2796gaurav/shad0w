"""Builds the optional C core (shad0w/_native/reflex.c) into the wheel as shad0w._reflex.

All metadata lives in pyproject.toml. If no C compiler is available the build still succeeds and shad0w
falls back to its numpy path, which makes identical decisions (about 15x slower).
"""
from setuptools import Extension, setup
from setuptools.command.build_ext import build_ext


class OptionalBuildExt(build_ext):
    def run(self):
        try:
            super().run()
        except Exception as e:  # noqa: BLE001
            print(f"shad0w: C core not built ({e}); using the numpy runtime")

    def build_extension(self, ext):
        try:
            super().build_extension(ext)
        except Exception as e:  # noqa: BLE001
            print(f"shad0w: C core not built ({e}); using the numpy runtime")


setup(
    ext_modules=[Extension("shad0w._reflex", ["shad0w/_native/reflex.c"], define_macros=[("SHAD0W_PYEXT", "1")],
                           extra_compile_args=[] if __import__("sys").platform == "win32" else ["-O3"],
                           libraries=[] if __import__("sys").platform == "win32" else ["m"])],
    cmdclass={"build_ext": OptionalBuildExt},
)
