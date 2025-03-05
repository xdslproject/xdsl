# Equality Saturation Dialect

## Tutorial 

In this note, we will present the `eqsat` dialect with examples.
The `eqsat` dialect refers to [e-graph](https://egraphs-good.github.io/), a graph representation of multiple functionally equivalent programs.

### How does a e-graph look like in MLIR?

In our example, we will saturate the expression `lambda x: x + x` with the following rules:

1. `a + b -> b + a`
2. `x + x -> x * 2`
3. `x * 2 -> x << 1`

Ultimately, we want to extract `lambda x: x << 1` as the final expression.

We start with the input:

```mlir
func.func @test(%x : index) -> (index) {
    %res = arith.addi %x, %x : index
    func.return %res : index
}
```

And the rewrites expressed as PDL: (not yet tested)

```mlir
// 1. a + b -> b + a
pdl.pattern @a_plus_b_to_b_plus_a : benefit(1) {
  %type = pdl.type : index
  %a = pdl.operand : %type
  %b = pdl.operand : %type
  %root = pdl.operation "arith.addi"(%a : !pdl.value, %b : !pdl.value)
  pdl.rewrite %root {
    %new = pdl.operation "arith.addi"(%b : !pdl.value, %a : !pdl.value)
    pdl.replace %root with %new
  }
}

// 2. x + x -> x * 2
pdl.pattern @x_plus_x_to_x_times_2 : benefit(1) {
  // x
  %type = pdl.type : index
  %x = pdl.operand : %type
  // x + x
  %root = pdl.operation "arith.addi"(%x : !pdl.value, %x : !pdl.value)
  pdl.rewrite %root {
    // 2
    %two_attr = pdl.attribute = 2 : index
    %two_op = pdl.operation "arith.constant" { "value" = %two_attr }
    // x * 2
    %new = pdl.operation "arith.muli"(%x : !pdl.value, %two_op : !pdl.value)
    pdl.replace %root with %new
  }
}


// 3. x * 2 -> x << 1
pdl.pattern @x_plus_x_to_x_times_2 : benefit(1) {
  // x
  %type = pdl.type : index
  %x = pdl.operand : %type
  // 2
  %two_attr = pdl.attribute = 2 : index
  %two_op = pdl.operation "arith.constant" { "value" = %two_attr }
  // x * 2
  %root = pdl.operation "arith.muli"(%x : !pdl.value, %two_op : !pdl.value)
  pdl.rewrite %root {
    // 1
    %one_attr = pdl.attribute = 1 : index
    %one_op = pdl.operation "arith.constant" { "value" = %one_attr }
    // x << 1
    %new = pdl.operation "arith.shli"(%x : !pdl.value, %one_op : !pdl.value)
    pdl.replace %root with %new
  }
}
```

Let's start with a simple example `2 * x`, and the MLIR code is as follows:

```mlir
func.func @test(%x : index) -> (index) {
    %c2 = arith.constant 2 : index
    %res = arith.muli %x, %c2 : index
    func.return %res : index
}
```

An initial e-graph is then a graph of e-classes, and each e-class collects a set of equivalent values, also known as e-nodes.
At the beginning, each e-class contains a e-node:

```mlir
func.func @test(%x : index) -> (index) {
    %x_eq = eqsat.eclass %x : index
    %c2 = arith.constant 2 : index
    %c2_eq = eqsat.eclass %c2 : index
    %res = arith.muli %x_eq, %c2_eq : index
    %res_eq = eqsat.eclass %res : index
    func.return %res_eq : index
}
```

Now say I want to rewrite it to `x << 1`, which leads to an e-graph that contains two expressions:

```mlir
func.func @test(%x : index) -> (index) {
    %x_eq = eqsat.eclass %x : index
    %c2 = arith.constant 2 : index
    %c2_eq = eqsat.eclass %c2 : index
    %c1 = arith.constant 1 : index
    %c1_eq = eqsat.eclass %c1 : index
    %shift = arith.shli %x_eq, %c1_eq : index
    %res = arith.muli %x_eq, %c2_eq : index
    // Here the e-class contains two e-nodes:
    // (2 * x, x << 1)
    %res_eq = eqsat.eclass %res, %shift : index
    func.return %res_eq : index
}
```

Another example is to add `x + x` to the e-graph:

```mlir
func.func @test(%x : index) -> (index) {
    %x_eq = eqsat.eclass %x : index
    %c2 = arith.constant 2 : index
    %c2_eq = eqsat.eclass %c2 : index
    %c1 = arith.constant 1 : index
    %c1_eq = eqsat.eclass %c1 : index
    %shift = arith.shli %x_eq, %c1_eq : index
    %res = arith.muli %x_eq, %c2_eq : index
    %add = arith.add %x_eq, %x_eq : index
    %res_eq = eqsat.eclass %res, %shift, %add : index
    func.return %res_eq : index
}
```

For extraction, we could add costs to these e-nodes:

```mlir
func.func @test(%x : index) -> (index) {
    %x_eq = eqsat.eclass %x : index
    %c2 = arith.constant 2 attributes {cost=1} : index
    %c2_eq = eqsat.eclass %c2 : index
    %c1 = arith.constant 1 attributes {cost=1} : index
    %c1_eq = eqsat.eclass %c1 : index
    %shift = arith.shli %x_eq, %c1_eq attributes {cost=3} : index  // add: lambda c0, c1: c0 + c1 + 2
    %res = arith.muli %x_eq, %c2_eq attributes {cost=5} : index  // add: lambda c0, c1: c0 + c1 + 4
    %add = arith.add %x_eq, %x_eq attributes {cost=3} : index  // add: lambda c0, c1: c0 + c1 + 2
    %res_eq = eqsat.eclass %res, %shift, %add : index
    func.return %res_eq : index
}

// Extract: minimal area (* = 10, + = 1, << = 0)
func.func @test(%x : index) -> (index) {
    %c1 = arith.constant 1 : index
    %shift = arith.shli %x_eq, %c1_eq : index
    func.return %res : index
}
```

### Getting Started

An end-to-end example is provided under `xdsl/tests/filecheck/projects/eqsat/identity.mlir`.
To run the test case:

```
xdsl-opt -p 'eqsat-create-eclasses,apply-eqsat-pdl,eqsat-add-costs,eqsat-extract' \
xdsl/tests/filecheck/projects/eqsat/identity.mlir
```

The command involves four passes:
* `eqsat-create-eclasses`: translates an MLIR program into an initial e-graph, where each e-class contains an e-node.
* `apply-eqsat-pdl`: explores a given set of rewrites on the e-graph and expands e-classes with more e-nodes. Particularly for this example, we have a single rewrite rule `x * y == y * x`.
* `eqsat-add-costs`: assigns a numerical cost to each e-node in an e-class, indicating which one is preferred to be presented in the output.
* `eqsat-extract`: extracts an e-node from each e-class based on the cost function (often taking the one with the lowest cost), and returns the transformed MLIR program.

PS: In the given example, the output program is identical to the input program but that is expected.
If you run these passes seperately and observe the intermediate MLIR, we can find a new representation is created and unioned into the e-graph, but did not get extracted from the graph as it has the same cost as the original program. By default, the first found e-node with the lowest cost is extracted - but we can improve this using a different cost function as a future work.

## Development Notes 


### How do we define scope of an e-graph?

JC: @Sasha proposes an new op to define a specific region/block for exploring equality saturation.
This improves the scalability of optimization so we can skip well-optimized code.

Question: Should this be a region or block? Personally I think region is more useful.
Before the `eqsat` dialect is available, I would use a function to define such design space.
A region is useful when we explore rewrites such as if conversion and complex loop transformation.

### How does the e-graph handle arbitrary operations?

We face the following problems when adding the `eqsat` dialect:

1. An e-class collects the result of an operation as an e-node. What if an operation does not have a result, such as `memref.store`?
2. An operation may have multiple results. How do we represent them in an e-graph?
3. An e-graph usually express one data-path expression. How do we express a program with control flows?

Here is Jianyi's attempt:

#### 1. Operations without any result

Here is an example of function that does not have a return value, because the value is stored in a memref.

```mlir
func.func @test(%x : index, %y : memref<32xindex>) -> () {
    %c2 = arith.constant 2 : index
    %res = arith.muli %x, %c2 : index
    memref.store %y[0], %res
}
```

What we can do is to consider the `memref.store` returns a `none` value.
Then we can rewrite the example into the following e-graph:

```mlir
func.func @test(%x : index, %y : memref<32xindex>) -> () {
    %y_eq = eqsat.eclass %y : memref<32xindex>
    %x_eq = eqsat.eclass %x : index
    %c0 = arith.constant 0 : index
    %c0_eq = eqsat.eclass %c0 : index
    %c2 = arith.constant 2 : index
    %c2_eq = eqsat.eclass %c2 : index
    %res = arith.muli %x_eq, %c2_eq : index
    %res_eq = eqsat.eclass %res : index
    %m = memref.store %y_eq[%c0_eq], %res_eq : none
    %m_eq = eqsat.eclass %m : none
}
```

Note that a `none` value is not equivalent to another `none` value. This means that the e-class `%m_eq` cannot have more than one e-node.

#### 2. Operations with multiple results

The following example returns two values:

```mlir
func.func @test(%x : index) -> (index, index) {
    %c2 = arith.constant 2 : index
    %res2 = arith.muli %x, %c2 : index
    %c3 = arith.constant 3 : index
    %res3 = arith.muli %x, %c3 : index
    func.return %res2, %res3 : index, index
}
```

That should be straightforward if MLIR already supports tuples:

```mlir
func.func @test(%x : index) -> (index) {
    %x_eq = eqsat.eclass %x : index
    %c2 = arith.constant 2 : index
    %c2_eq = eqsat.eclass %c2 : index
    %res2 = arith.muli %x_eq, %c2_eq : index
    %res2_eq = eqsat.eclass %res2 : index
    %c3 = arith.constant 3 : index
    %c3_eq = eqsat.eclass %c3 : index
    %res3 = arith.muli %x_eq, %c3_eq : index
    %res3_eq = eqsat.eclass %res3 : index
    func.return %res2_eq, %res3_eq : index
}
```

#### 3. Multiple expressions

A more interesting case is where we may have multiple expressions. A common case is when we have memory statements.

```mlir
// y[0] = y[1];
// y[2] = x * 2;
func.func @test(%x : index, %y : memref<32xindex>) -> () {
    %temp = memref.load %y[1] : index
    memref.store %y[0], %temp
    %c2 = arith.constant 2 : index
    %res = arith.muli %x, %c2 : index
    memref.store %y[2], %res
}
```

Because we express the e-graph in a graph region, the program order information is lost.
Here we use `eqsat.seq` to preserve the program order in the graph so we can do some scheduling rewrites.

```mlir
func.func @test(%x : index, %y : memref<32xindex>) -> () {
    %y_eq = eqsat.eclass %y : memref<32xindex>
    %x_eq = eqsat.eclass %x : index
    %c0 = arith.constant 0 : index
    %c0_eq = eqsat.eclass %c0 : index
    %c1 = arith.constant 1 : index
    %c1_eq = eqsat.eclass %c1 : index
    %temp = memref.load %y_eq[%c1_eq] : index
    %temp_eq = eqsat.eclass %temp : indetemp
    %m0 = memref.store %y_eq[%c0_eq], %temp_eq : none
    %m0_eq = eqsat.eclass %m0 : none
    %c2 = arith.constant 2 : index
    %c2_eq = eqsat.eclass %c2 : index
    %res = arith.muli %x_eq, %c2_eq : index
    %res_eq = eqsat.eclass %res : index
    %m = memref.store %y_eq[%c2_eq], %res_eq : none
    %m_eq = eqsat.eclass %m : none
    %seq = eqsat.seq %m0_eq, %m_eq : none
}
```

Question:
The `seq` op seems a list of non-return operations, but I am not sure if this covers all the cases?

### How does the e-graph handle functions?

TODO

### How does the e-graph control flow ops, such as loops?

TODO

### How does the e-graph handle blocks?

TODO

## Meeting Notes

### 2024-09-16

Attendees: Janyi, Samuel, Sasha

How do we handle congruence? Like adding `a = b` to the following:

```python
x = f(a)
y = f(b)
# expect x to be equality saturated to y
```

TODO:

- Sasha to open PR to merge eqsat dialect
- Janyi open PR to convert standard MLIR function to version with eclass ops
- Sasha to create GH Project and small tasks
- Samuel to try extraction and maybe cost model
