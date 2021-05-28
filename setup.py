from setuptools import setup

setup(
    name="lleaves",
    version="0.0.1",
    packages=["lleaves", "lleaves.tree_compiler"],
    url="https://github.com/siboehm/LLeaVes",
    license="MIT",
    author="Simon Boehm",
    author_email="simon@siboehm.com",
    description="LLVM-based compiler for LightGBM models",
    install_requires=["llvmlite", "numpy"],
)
