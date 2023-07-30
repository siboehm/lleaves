from llvmlite import ir
from llvmlite.ir.transforms import CallVisitor


class Inliner(CallVisitor):
    def __init__(self) -> None:
        super().__init__()

    def visit_Call(self, inst: ir.CallInstr):
        exit(1)
