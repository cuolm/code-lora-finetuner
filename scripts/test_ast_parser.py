import os
from tree_sitter import Node, Parser
from tree_sitter_language_pack import get_parser
from pathlib import Path
import numpy as np


def collect_logical_blocks(node: Node, source_code_utf8: bytes) -> list:
    logical_blocks = []
    block_types = {'function_definition'}

    if node.type in block_types:
        snippet = source_code_utf8[node.start_byte:node.end_byte]
        logical_blocks.append((snippet, node))

    for child in node.children:
        logical_blocks.extend(collect_logical_blocks(child, source_code_utf8))
    return logical_blocks

def collect_subblock_indices(node: Node, logical_subblock_types: set[str], base_offset: int) -> list:
    subblock_indices = []

    if node.type in logical_subblock_types:
        relative_start_byte = node.start_byte - base_offset
        relative_end_byte = node.end_byte - base_offset
        subblock_indices.append((relative_start_byte, relative_end_byte))
    for child in node.children:
        subblock_indices.extend(collect_subblock_indices(child, logical_subblock_types, base_offset))
    return subblock_indices

def generate_fim_examples(function_code: bytes, subblock_indices: list, fim_variants_per_subblock: int, bytes_per_code_block: int) -> list:
    fim_prefix_token = "<|fim_prefix|>".encode('utf8')
    fim_middle_token = "<|fim_middle|>".encode('utf8')
    fim_suffix_token = "<|fim_suffix|>".encode('utf8')
    fim_pad_token = "<|fim_pad|>".encode('utf8')

    rng = np.random.default_rng(seed=0)
    num_of_subblocks = len(subblock_indices)
    if (fim_variants_per_subblock > num_of_subblocks):
        fim_variants_per_subblock = num_of_subblocks

    unique_random_indices = rng.choice(len(subblock_indices), size=fim_variants_per_subblock, replace=False)

    fim_examples = [] 

    for idx in unique_random_indices:
        middle_start = subblock_indices[idx][0]
        middle_end = subblock_indices[idx][1]

        prefix = function_code[:middle_start]
        middle = function_code[middle_start:middle_end]
        suffix = function_code[middle_end:]

        fim_example = (
            fim_prefix_token + prefix +
            fim_suffix_token + suffix +
            fim_middle_token + middle
        )

        prefix_suffix_middle_token_length = 3
        pad_length = (bytes_per_code_block+ prefix_suffix_middle_token_length) - len(fim_example)

        if pad_length > 0:
            pad_bytes = pad_length // len(fim_pad_token)
            fim_example += fim_pad_token * pad_bytes

        fim_examples.append(fim_example)

    return fim_examples

def print_code_blocks_and_fim_examples(data_path: Path, python_parser: Parser, extensions: tuple, bytes_per_code_block: int, logical_subblock_types: set[str]) -> None:         
    for root, _, files in os.walk(data_path):
        for filename in files:
            if filename.endswith(extensions):
                filepath = os.path.join(root, filename)
                try:
                    with open(filepath, 'rb') as f:
                        source_code_utf8 = f.read()
                        source_code_utf8.decode('utf-8')  # Validate UTF-8, if not exception is raised, file skipped
                except UnicodeDecodeError:
                    print(f"Skipping file '{filename}': Not a valid UTF-8 file.")
                    continue
                
                tree = python_parser.parse(source_code_utf8)
                root_node = tree.root_node

                code_blocks = collect_logical_blocks(root_node, source_code_utf8)

                for code_utf8, node in code_blocks:
                    code_utf8 = code_utf8[:bytes_per_code_block]  # Trunctate code block if it is larger than bytes_per_code_block.
                    base_offset = node.start_byte
                    subblock_indices = collect_subblock_indices(node, logical_subblock_types, base_offset)
                    subblock_indices = sorted(subblock_indices, key=lambda x: x[1])

                    # Discard subblocks that have a larger end index than bytes_per_code_block.
                    i = 0
                    while (i < len(subblock_indices)) and (subblock_indices[i][1] <= bytes_per_code_block):
                        i += 1
                    subblock_indices = subblock_indices[:i]

                    print("=== Code Block Info ===")
                    print(f"Code:\n{code_utf8}")
                    print(f"Length: {len(code_utf8)}")
                    print(f"Subblock Indices: {subblock_indices}\n")

                    print("=== Subblocks ===")
                    for idxs in subblock_indices:
                        subblock_start, subblock_end = idxs
                        print(f"{code_utf8[subblock_start:subblock_end]}\n")

                    fim_variants_per_subblock = 2
                    fim_examples = generate_fim_examples(code_utf8, subblock_indices, fim_variants_per_subblock, bytes_per_code_block)

                    print("=== FIM Examples ===")
                    for example in fim_examples:
                        print(f"{example}\n")

def main():
    project_root_path = Path(__file__).resolve().parent.parent
    data_path = project_root_path / 'data'
    extensions = (".c", ".h")
    tree_sitter_parser= get_parser("c")
    token_to_byte_ration = 3 # Assume token to byte ration of 3
    max_bytes_per_func = token_to_byte_ration*50 

    logical_subblock_types = {
        "compound_statement",      # Block of code enclosed in braces `{ ... }` representing a function body or other scoped block
        "parameter_list",          # The list of parameters in a function declaration or definition (e.g., `(int a, int b)`)
        "declaration",             # Variable or other declarations (e.g., `int x;`)
        "expression_statement",    # Statements that are expressions terminated by a semicolon (e.g., `x = y + 1;`)
        "if_statement",            # If-else conditional statement with optional else branch
        "while_statement",         # While loop statement
        "for_statement",           # For loop statement, including header and body
        "switch_statement",        # Switch statement controlling multiple cases
        "case_statement",          # Individual case or default in a switch block
        "return_statement"         # Return statement returning an expression or void from a function
    }

    print_code_blocks_and_fim_examples(data_path, tree_sitter_parser, extensions, max_bytes_per_func, logical_subblock_types)

if __name__ == "__main__":
    main()
