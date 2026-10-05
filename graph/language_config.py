"""
language_config.py — the registry that makes build_dependency_graph.py work
on ANY language, not just Python.

HOW TO ADD A NEW LANGUAGE (this is the whole point of this file):
Add one entry to LANGUAGES below. You need 3 things:
  1. The file extension(s) that language uses
  2. The grammar name tree-sitter-language-pack knows it by (see
     https://github.com/Goldziher/tree-sitter-language-pack for the full
     list of 370+ supported languages/names)
  3. A tree-sitter query (a small pattern, not real code) that finds import
     statements in that language's syntax tree, with the import target
     captured as @module

Then tell it how to RESOLVE that import text into an actual file:
  - "dotted" style: the import looks like "package.sub.module" (Python,
    Java, Kotlin, Scala) — matched against file paths using the same
    right-aligned suffix matching already used for Python.
  - "path" style: the import looks like a file path, often relative
    ("./utils", "../lib/foo", "myheader.h") — resolved relative to the
    importing file's own folder, trying each extension in `path_extensions`.

No other code needs to change to add a language — build_dependency_graph.py
reads this file and handles any language listed here generically.
"""

LANGUAGES = {
    ".py": {
        "ts_name": "python",
        "style": "dotted",
        "import_query": """
            (import_statement (dotted_name) @module)
            (import_statement (aliased_import (dotted_name) @module))
            (import_from_statement module_name: (dotted_name) @module)
            (import_from_statement module_name: (relative_import) @module)
        """,
    },
    ".js": {
        "ts_name": "javascript",
        "style": "path",
        "path_extensions": [".js", ".jsx", "/index.js"],
        "import_query": """
            (import_statement source: (string) @module)
            (call_expression
              function: (identifier) @_fn
              arguments: (arguments (string) @module)
              (#eq? @_fn "require"))
            (export_statement source: (string) @module)
        """,
    },
    ".jsx": {  # same rules as .js
        "ts_name": "javascript",
        "style": "path",
        "path_extensions": [".js", ".jsx", "/index.js"],
        "import_query": """
            (import_statement source: (string) @module)
            (call_expression
              function: (identifier) @_fn
              arguments: (arguments (string) @module)
              (#eq? @_fn "require"))
        """,
    },
    ".ts": {
        "ts_name": "typescript",
        "style": "path",
        "path_extensions": [".ts", ".tsx", "/index.ts"],
        "import_query": """
            (import_statement source: (string) @module)
            (export_statement source: (string) @module)
        """,
    },
    ".tsx": {
        "ts_name": "tsx",
        "style": "path",
        "path_extensions": [".ts", ".tsx", "/index.tsx"],
        "import_query": """
            (import_statement source: (string) @module)
        """,
    },
    ".java": {
        "ts_name": "java",
        "style": "dotted",
        "import_query": """
            (import_declaration (scoped_identifier) @module)
        """,
    },
    ".go": {
        "ts_name": "go",
        "style": "path",  # Go import paths aren't local files, but we treat
                           # them the same way the C++ resolver does: only
                           # same-repo paths will ever match anything.
        "path_extensions": [".go"],
        "import_query": """
            (import_spec path: (interpreted_string_literal) @module)
        """,
    },
    ".c": {
        "ts_name": "c",
        "style": "path",
        "path_extensions": [".h", ".c"],
        "import_query": """
            (preproc_include path: (_) @module)
        """,
    },
    ".h": {
        "ts_name": "c",
        "style": "path",
        "path_extensions": [".h", ".c"],
        "import_query": """
            (preproc_include path: (_) @module)
        """,
    },
    ".cpp": {
        "ts_name": "cpp",
        "style": "path",
        "path_extensions": [".hpp", ".h", ".cpp"],
        "import_query": """
            (preproc_include path: (_) @module)
        """,
    },
    ".hpp": {
        "ts_name": "cpp",
        "style": "path",
        "path_extensions": [".hpp", ".h", ".cpp"],
        "import_query": """
            (preproc_include path: (_) @module)
        """,
    },
    ".rb": {
        "ts_name": "ruby",
        "style": "path",
        "path_extensions": [".rb"],
        "import_query": """
            (call
              method: (identifier) @_fn
              arguments: (argument_list (string (string_content) @module))
              (#match? @_fn "^(require|require_relative)$"))
        """,
    },
    ".rs": {
        "ts_name": "rust",
        "style": "dotted",
        "import_query": """
            (use_declaration argument: (_) @module)
        """,
    },
    ".php": {
        "ts_name": "php",
        "style": "path",
        "path_extensions": [".php"],
        "import_query": """
            (include_expression (string) @module)
            (require_expression (string) @module)
            (require_once_expression (string) @module)
            (include_once_expression (string) @module)
        """,
    },
    ".cs": {
        "ts_name": "csharp",
        "style": "dotted",
        "import_query": """
            (using_directive (qualified_name) @module)
            (using_directive (identifier) @module)
        """,
    },
    ".kt": {
        "ts_name": "kotlin",
        "style": "dotted",
        "import_query": """
            (import_header (identifier) @module)
        """,
    },
}

# Folders to never walk into, regardless of language (dependencies,
# generated code, VCS internals — never real project source).
IGNORE_DIRS = {
    "tests", "test", "vendor", "node_modules", ".git", "venv", "env",
    "__pycache__", "build", "dist", "docs", "examples", "target",
    ".next", "bin", "obj", "out",
}