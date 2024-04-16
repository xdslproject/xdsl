# The xDSL book

This directory contains the source code of the xDSL book.

To compile the book, first [install mdbook](https://rust-lang.github.io/mdBook/guide/installation.html) then run:

```
$ mdbook serve --open
```

This will open the book it in your default browser and set up a local mdbook server that will hot-reload your changes to the book.

## Including source code

In order to include source code in the book, create an accompanying Python file for your section. Then, include sections of the Python file in the markdown using [mdbook partial file inclusion features](https://rust-lang.github.io/mdBook/format/mdbook.html#including-portions-of-a-file).

The Python file should be able to run on its own and must be tested by CI (any Python file added to this directory will be ran by CI for errors). Prefer importing from adjacent Python files instead of redefining when your section depends on previous sections' definitions.
