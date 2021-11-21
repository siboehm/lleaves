from os import path

from setuptools import find_packages, setup

this_directory = path.abspath(path.dirname(__file__))
with open(path.join(this_directory, "README.md"), encoding="utf-8") as f:
    long_description = f.read()

setup(
    name="lleaves",
    use_scm_version=True,
    setup_requires=["setuptools_scm"],
    packages=find_packages(exclude=["benchmarks"]),
    url="https://github.com/siboehm/lleaves",
    project_urls={"Documentation": "https://lleaves.readthedocs.io/en/latest/"},
    license="MIT",
    classifiers=[
        "License :: OSI Approved :: MIT License",
        "Development Status :: 4 - Beta",
        "Programming Language :: Python :: 3",
    ],
    author="Simon Boehm",
    author_email="simon@siboehm.com",
    description="LLVM-based compiler for LightGBM models",
    long_description=long_description,
    long_description_content_type="text/markdown",
    python_requires=">=3.7",
    install_requires=["llvmlite>=0.36", "numpy"],
)
