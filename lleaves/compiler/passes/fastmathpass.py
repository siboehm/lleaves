from llvmlite import ir
from llvmlite.ir.transforms import CallVisitor, Visitor


class FastFloatOpVisitor(Visitor):
    float_binops = frozenset(["fneg", "fadd", "fsub", "fmul", "fdiv", "frem", "fcmp"])

    def __init__(self, flags):
        self.flags = flags

    def visit_Instruction(self, instr: ir.Instruction):
        if instr.opname in self.float_binops:
            if not instr.flags:
                for flag in self.flags:
                    instr.flags.append(flag)


class FastFloatCallVisitor(CallVisitor):
    def __init__(self, flags):
        self.flags = flags

    def visit_Call(self, instr):
        if instr.type in (ir.FloatType(), ir.DoubleType()):
            for flag in self.flags:
                instr.fastmath.add(flag)


class FastFloatSelectVisitor(Visitor):
    def __init__(self, flags):
        self.flags = flags

    def visit_Instruction(self, instr: ir.Instruction):
        if instr.opname == "select" and instr.type in (ir.FloatType(), ir.DoubleType()):
            if not instr.flags:
                for flag in self.flags:
                    instr.flags.append(flag)


def rewrite_module(mod):
    flags = ["nsz", "arcp", "contract", "afn", "reassoc"]
    # flags = ["fast"]
    FastFloatOpVisitor(flags).visit(mod)
    FastFloatCallVisitor(flags).visit(mod)
    FastFloatSelectVisitor(flags).visit(mod)
