from llvmlite import ir
from llvmlite.ir.transforms import CallVisitor, Visitor

# Adapted from https://github.com/numba/numba/blob/master/numba/core/fastmathpass.py


class FastFloatBinOpVisitor(Visitor):
    """
    A pass to add fastmath flags to float-binop instruction if they don't have
    any flags.
    """

    float_binops = ["fadd", "fsub", "fmul", "fdiv", "frem", "fcmp"]

    def __init__(self, flags):
        self.flags = flags

    def visit_Instruction(self, instr):
        if instr.opname in self.float_binops and not instr.flags:
            for flag in self.flags:
                instr.flags.append(flag)


class FastFloatCallVisitor(CallVisitor):
    """
    A pass to change all floating point function calls to use fastmath.
    """

    def __init__(self, flags):
        self.flags = flags

    def visit_Call(self, instr):
        # Add to any call that has float/double return type
        if instr.type in (ir.FloatType(), ir.DoubleType()):
            for flag in self.flags:
                instr.fastmath.add(flag)


def rewrite_module(mod, flags):
    """
    Rewrite the given LLVM module to add fastmath flags everywhere.
    """
    FastFloatBinOpVisitor(flags).visit(mod)
    FastFloatCallVisitor(flags).visit(mod)
