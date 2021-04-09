from frontend import ir_from_json

# Now implement the function
block = func.append_basic_block(name="entry")
builder = ir.IRBuilder(block)
a, b = func.args
left = ir.Constant(double, 0.481)
right = ir.Constant(double, 0.497)
res = builder.alloca(double)
comp = builder.fcmp_ordered("<=", a, b, name="compare")
with builder.if_else(comp) as (then, otherwise):
    with then:
        builder.store(left, res)
    with otherwise:
        builder.store(right, res)
result = builder.load(res)
builder.ret(result)

# Print the module IR
print(module)
