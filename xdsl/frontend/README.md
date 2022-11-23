# Frontend framework

This document briefly describes the frontend framework.

## How to install everything

Probably you found the README in the main directory misleading and incomplete.
In this case, here is another attempt.

First, make sure you use **Python 3.10**.

```
$ python3 --version
Python 3.10.0
```

Next, clone the code and checkout the right branch.

```
$ git clone https://github.com/xdslproject/xdsl.git
$ cd xdsl
$ git checkout frontend
```

Setup you virtual enironement and download/install all dependencies.

```
$ python3 -m venv env
$ source env/bin/activate 
$ pip install -e .
```

You should be ready to write frontend programs!

## How to compile and run programs

To compile your program, first create a new `FrontendProgram`. Then, using
`CodeContext` and Python-like syntax one can write any program. Below, there
is an example of such a program with a single function `num2bits`

```python
from typing import Literal, Tuple
from xdsl.frontend.program import FrontendProgram
from xdsl.frontend.context import CodeContext
from xdsl.frontend.dialects.builtin import i1, i64, TensorType

p = FrontendProgram()
with CodeContext(p):

    def num2bits(inp: i64) -> TensorType[i1, Tuple[Literal[64],]]:
        assert inp < (1 << 64)
        out: TensorType[i1, Tuple[Literal[64],]] = [0 for i in range(64)]
        for i in range(64):
            out[i] = ((inp >> i) & 1)
        return out
```

To compile a program, just call `compile` on your `FrontendProgram` instance.
To produce SSA output, make sure you call `desymref`!
any time 

```python
# Compile the program.
p.compile()

# Apply desymref pass.
p.desymref()
```

IR can be printend either as xDSL (call `xdsl`), or as MLIR (call `mlir`).

```python
print(p.xdsl())
print(p.mlir())
```

Additionally, if you have MLIR built and installed on your machine you can
run `mlir-opt` directly from Python. All you need is to specify the arguments
the binary should take, e.g. `--verify-each`, etc.

```python
MLIR_OPT_PATH = "../llvm-project/build/bin/mlir-opt"
mlir_output = p.mlir_roundtrip(MLIR_OPT_PATH, args=["--verify-each"])
print(mlir_output)
```

The output of the program above is shown below. Full source code can be found in
the directory of this README in file `example.py`.

```mlir
module {
  func.func private @num2bits(%arg0: i64) -> tensor<64xi1> {
    %c64_i32 = arith.constant 64 : i32
    %c1_i32 = arith.constant 1 : i32
    %0 = arith.shli %c1_i32, %c64_i32 : i32
    %1 = arith.extsi %0 : i32 to i64
    %2 = arith.cmpi slt, %arg0, %1 : i64
    cf.assert %2, ""
    %c0_i32 = arith.constant 0 : i32
    %3 = tensor.splat %c0_i32 : tensor<64xi32>
    %4 = arith.trunci %3 : tensor<64xi32> to tensor<64xi1>
    %5 = affine.for %arg1 = 0 to 64 iter_args(%arg2 = %4) -> (tensor<64xi1>) {
      %c1_i32_0 = arith.constant 1 : i32
      %6 = arith.index_cast %arg1 : index to i64
      %7 = arith.shrsi %arg0, %6 : i64
      %8 = arith.extsi %c1_i32_0 : i32 to i64
      %9 = arith.andi %7, %8 : i64
      %10 = arith.trunci %9 : i64 to i1
      %11 = tensor.insert %10 into %arg2[%arg1] : tensor<64xi1>
      affine.yield %11 : tensor<64xi1>
    }
    return %5 : tensor<64xi1>
  }
}
```

Now, if you want to store the output as MLIR, simply pipe the output into the
file, e.g.

```bash
python xdsl/frontend/example.py > your_file.mlir
```

Make sure that you do not print xDSL though! We recommend only prinitng MLIR from
the roundtrip for that purpose since it will be verified on MLIR side and optimized,
if needed.

## How to test code generation

To test frontend, we use `filecheck`. In particular, all tests are stored under
`tests/filecheck/*` directory, and to run them, just use
```bash
# Runs desymref tests. 
$ lit tests/filecheck/desymref/

# Runs frontend code generation tests. 
$ lit tests/filecheck/frontend/
```
