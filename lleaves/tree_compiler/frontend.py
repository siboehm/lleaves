from lleaves.tree_compiler.tree_compiler import Forest


def ir_from_json(json):
    return Forest(json).get_ir()
