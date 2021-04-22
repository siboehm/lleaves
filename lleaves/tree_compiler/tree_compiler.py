from lleaves.tree_compiler.ast import parse_to_forest


def ir_from_model_file(file_path):
    forest = parse_to_forest(file_path)
    return forest.get_ir()
