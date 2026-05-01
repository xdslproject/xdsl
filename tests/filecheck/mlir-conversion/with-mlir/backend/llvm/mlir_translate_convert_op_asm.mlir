// REQUIRES: llc
// sed: (1) strip llc comments (e.g. '# %bb.0:'), (2) collapse '.file "<name>"', and (3) trim trailing ws
// RUN: xdsl-opt -t llvm %S/../../../../backend/llvm/convert_op.mlir | %llc -O0 -mtriple=x86_64-unknown-linux-gnu | sed -E \
// RUN:     -e 's/[[:space:]]*#.*$//' \
// RUN:     -e 's/\.file.*$/.file/' \
// RUN:     -e 's/[[:space:]]+$//' \
// RUN:     > %t.xdsl.s
// RUN: mlir-translate --mlir-to-llvmir %S/../../../../backend/llvm/convert_op.mlir | %llc -O0 -mtriple=x86_64-unknown-linux-gnu | sed -E \
// RUN:     -e 's/[[:space:]]*#.*$//' \
// RUN:     -e 's/\.file.*$/.file/' \
// RUN:     -e 's/[[:space:]]+$//' \
// RUN:     > %t.mlir.s
// RUN: diff %t.xdsl.s %t.mlir.s
