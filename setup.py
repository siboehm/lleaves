from setuptools import setup

with open("README.md") as f:
    long_description = f.read()

setup(
    name="lleaves",
    version="0.0.1",
    packages=["lleaves", "lleaves.tree_compiler"],
    url="https://github.com/siboehm/LLeaVes",
    license="MIT",
    classifiers=[
        "Licence :: OSI Approved :: MIT License",
        "Development Status :: 4 - Beta",
        "Programming Language :: Python :: 3",
    ],
    author="Simon Boehm",
    author_email="simon@siboehm.com",
    description="LLVM-based compiler for LightGBM models",
    long_description=long_description,
    long_description_content_type="text/markdown",
    python_requires=">=3",
    install_requires=["llvmlite", "numpy"],
)
