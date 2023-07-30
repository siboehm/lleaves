import llvmlite
from lleaves.compiler.ast import parse_to_ast
from lleaves.compiler.codegen import gen_forest

from lleaves.compiler.passes import fastmathpass


def test_fastmath():
    forest = parse_to_ast("tests/models/tiniest_single_tree/model.txt")
    forest.raw_score = False
    ir = llvmlite.ir.Module(name="forest")
    gen_forest(forest, ir, 34, "forest_root")

    # with open("forest_slow.ir", "w") as f:
    #     print(ir, file=f)

    fastmathpass.rewrite_module(ir)

    # with open("forest_fast.ir", "w") as f:
    #     print(ir, file=f)

    assert "nsz" in str(ir)
