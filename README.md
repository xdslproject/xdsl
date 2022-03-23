# xDSL

TODO

## Prerequisits

To install all required dependencies, execute the following command:

```
pip install -r requirements.txt
```

TODO: check if PYTHONPATH is required or if there exists an easy fix for it.

## Testing

This project includes pytest unit test and llvm-style filecheck tests. They can be executed using to following commands from within the root directory of the project:

```
# Executes pytests which are located in tests/
pytest

# Executes filecheck tests
lit tests/filecheck
```

## Formatting

All python code used in xDSL use yapf to format the code in a uniform manner. 

https://github.com/google/yapf

To automate the formatting within vim, one can use https://github.com/vim-autoformat/vim-autoformat and trigger a `:Autoformat` on save.
