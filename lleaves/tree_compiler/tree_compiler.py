from lleaves.tree_compiler.ast import parse_to_ast


def ir_from_model_file(file_path):
    forest = parse_to_ast(file_path)
    return forest.get_ir()
