from llvmlite import ir

import llvmlite.binding as llvm
import json

with open("model.json") as f:
    model_json = json.load(f)

tree_j = model_json["tree_info"]

# Create some useful types
double = ir.DoubleType()
dp = ir.PointerType(double)
fnty = ir.FunctionType(double, (double, double))

# Create an empty module...
module = ir.Module(name=__file__)
# and declare a function named "fpadd" inside it
func = ir.Function(module, fnty, name="tree")

# Now implement the function
block = func.append_basic_block(name="entry")
builder = ir.IRBuilder(block)
a, b = func.args
zero = ir.Constant(double, 0.0)
one = ir.Constant(double, 1.0)
res = builder.alloca(double)
comp = builder.fcmp_ordered("<=", a, b, name="compare")
with builder.if_else(comp) as (then, otherwise):
    with then:
        builder.store(zero, res)
    with otherwise:
        builder.store(one, res)
result = builder.load(res)
builder.ret(result)

# Print the module IR
print(module)

mod = llvm.parse_assembly(str(module))
mod.verify()

with open("tree.ll", "w") as file:
    file.write(str(module))
