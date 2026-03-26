# Tree-sitter Customization

## Add New Language Block Definitions
1. Navigate to the [tree-sitter-language-pack](https://github.com/Goldziher/tree-sitter-language-pack?tab=readme-ov-file#available-languages) and find your language's repository link.
2. Inside the language repository, search the `grammar.js` file for the required syntax node names.
3. Create a new file, e.g., called `tree_sitter_definitions.json`.
Add a new entry for your language, defining the following nodes:
    * `block_types`: The outer syntax nodes (e.g., functions, classes). These are the structural elements from which we create FIM examples.
    * `subblock_types`: The inner syntax nodes (e.g., statements, expressions). These are masked to form the predicted middle portion of the FIM example.
    
    Example entry for the C programming language:
      ```json
      "c": {
        "block_types": [
          "function_definition",
          "struct_specifier",
          "union_specifier",
          "enum_specifier"
        ],
        "subblock_types": [
          "compound_statement",
          "parameter_list",
          "declaration",
          "expression_statement",
          "if_statement",
          "while_statement",
          "for_statement",
          "switch_statement",
          "case_statement",
          "return_statement",
          "field_declaration_list",
          "field_declaration",
          "enumerator_list",
          "enumerator"
        ]
      }
    ```
4. Set the parameter in your config file `codefinetuner_config.yaml` to the path of that file:
    ```yaml
    tree_sitter_definitions_path: "path/to/your/tree_sitter_definitions.json"
    ```

## Build Custom Parser
If the language you want to use for fine-tuning is not present in the [tree-sitter-language-pack](https://github.com/Goldziher/tree-sitter-language-pack), you can build a tree-sitter language parser from source. Here is an example for the Mojo programming language:

#### 1. Add as Submodule and Build

Add the Tree-sitter language repository, e.g., to a `third_party` directory as a submodule.

```bash
git submodule add [https://github.com/lsh/tree-sitter-mojo.git](https://github.com/lsh/tree-sitter-mojo.git) third_party/tree-sitter-mojo     

cd third_party/tree-sitter-mojo

make
```
After executing these instructions, you should have a shared library file named `libtree-sitter-mojo.so` (Linux), `libtree-sitter-mojo.dylib` (macOS), or `libtree-sitter-mojo.dll` (Windows) in the root directory of the `tree-sitter-mojo` repository.

#### 2. Add Tree-sitter Language Block Definitions
Create or add to an existing Tree-sitter language block definitions file according to [this](#add-new-tree-sitter-language-block-definitions) guide. 

#### 3. Update Config File
Update the path in the YAML config file:
```yaml
tree_sitter_parser_path: "path/to/shared_library_file"  # e.g., "./third_party/tree-sitter-mojo/libtree-sitter-mojo.dylib"
```