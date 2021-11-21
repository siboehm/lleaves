import pytest
from lightgbm import Booster

from lleaves import Model


@pytest.fixture(scope="session")
def NYC_llvm():
    llvm_model = Model(model_file="tests/models/NYC_taxi/model.txt")
    llvm_model.compile()
    return llvm_model


@pytest.fixture(scope="session")
def NYC_lgbm():
    return Booster(model_file="tests/models/NYC_taxi/model.txt")


@pytest.fixture(scope="session")
def mtpl2_llvm():
    llvm_model = Model(model_file="tests/models/mtpl2/model.txt")
    llvm_model.compile()
    return llvm_model
