uv run xdsl-opt -p decompose-softmax testlowering.mlir


uv run xdsl-opt -p decompose-softmax,lower-exp-to-polynomial testlowering.mlir


uv run xdsl-opt -p decompose-softmax,lower-exp-to-polynomial,expand-polynomial-eval testlowering.mlir
