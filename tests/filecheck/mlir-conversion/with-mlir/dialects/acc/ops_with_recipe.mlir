// RUN: xdsl-opt --allow-unregistered-dialect --disable-verify %s --print-op-generic | mlir-opt --allow-unregistered-dialect --mlir-print-op-generic --mlir-print-local-scope | xdsl-opt --allow-unregistered-dialect --disable-verify | filecheck %s

// MLIR interop coverage for the privatization data-clause ops that
// upstream's verifier requires a `recipe` SymbolRefAttr to resolve to a
// real `acc.private.recipe` / `acc.firstprivate.recipe` /
// `acc.reduction.recipe` op (per `checkRecipe` in OpenACC.cpp): the var
// type is memref (`PointerLikeType`, not `MappableType`), so a recipe is
// mandatory.
//
// The recipe ops themselves are deferred to a later PR
// so xDSL has no IRDL classes for them yet. To still
// exercise the data-op interop, we declare the three recipes inline in
// *generic form* and pass `--allow-unregistered-dialect --disable-verify`
// on each xdsl-opt invocation:
//   - `--allow-unregistered-dialect` lets xdsl-opt print/parse the
//     recipe ops as unregistered (they're in the registered acc dialect
//     but xDSL has no classes yet).
//   - `--disable-verify` skips `acc.yield`'s `HasParent` check, since
//     the parent is one of the unregistered recipe ops.
// `mlir-opt` parses the recipes as the real registered ops (it has
// classes for them) and runs its full verifier over the data ops + their
// `recipe` symbol references; the round-trip back through xdsl-opt then
// re-emits the recipes in generic form. When PR 13 lands, this file
// should be folded into `ops.mlir` with the standard MLIR_ROUNDTRIP /
// MLIR_GENERIC_ROUNDTRIP substitutions.

"acc.private.recipe"() <{sym_name = "priv_recipe", type = memref<10xf32>}> ({
^bb0(%arg0: memref<10xf32>):
  "acc.yield"(%arg0) : (memref<10xf32>) -> ()
}, {
^bb0(%arg0: memref<10xf32>):
  "acc.yield"(%arg0) : (memref<10xf32>) -> ()
}) : () -> ()
// CHECK:      "acc.private.recipe"() <{sym_name = "priv_recipe", type = memref<10xf32>}> ({
// CHECK-NEXT:   ^{{.*}}(%{{.*}}: memref<10xf32>):
// CHECK-NEXT:     acc.yield %{{.*}} : memref<10xf32>
// CHECK-NEXT:   }, {
// CHECK-NEXT:   ^{{.*}}(%{{.*}}: memref<10xf32>):
// CHECK-NEXT:     acc.yield %{{.*}} : memref<10xf32>
// CHECK-NEXT:   }) : () -> ()

"acc.firstprivate.recipe"() <{sym_name = "fp_recipe", type = memref<10xf32>}> ({
^bb0(%arg0: memref<10xf32>):
  "acc.yield"(%arg0) : (memref<10xf32>) -> ()
}, {
^bb0(%arg0: memref<10xf32>, %arg1: memref<10xf32>):
  "acc.yield"(%arg1) : (memref<10xf32>) -> ()
}, {
^bb0(%arg0: memref<10xf32>):
  "acc.terminator"() : () -> ()
}) : () -> ()
// CHECK:      "acc.firstprivate.recipe"() <{sym_name = "fp_recipe", type = memref<10xf32>}> ({
// CHECK-NEXT:   ^{{.*}}(%{{.*}}: memref<10xf32>):
// CHECK-NEXT:     acc.yield %{{.*}} : memref<10xf32>
// CHECK-NEXT:   }, {
// CHECK-NEXT:   ^{{.*}}(%{{.*}}: memref<10xf32>, %{{.*}}: memref<10xf32>):
// CHECK-NEXT:     acc.yield %{{.*}} : memref<10xf32>
// CHECK-NEXT:   }, {
// CHECK-NEXT:   ^{{.*}}(%{{.*}}: memref<10xf32>):
// CHECK-NEXT:     "acc.terminator"() : () -> ()
// CHECK-NEXT:   }) : () -> ()

"acc.reduction.recipe"() <{sym_name = "red_recipe", type = memref<i64>, reductionOperator = #acc.reduction_operator<add>}> ({
^bb0(%arg0: memref<i64>):
  "acc.yield"(%arg0) : (memref<i64>) -> ()
}, {
^bb0(%arg0: memref<i64>, %arg1: memref<i64>):
  "acc.yield"(%arg0) : (memref<i64>) -> ()
}, {
^bb0(%arg0: memref<i64>):
  "acc.terminator"() : () -> ()
}) : () -> ()
// `reductionOperator` lives in the recipe op's properties dict alongside
// `sym_name` / `type`; mlir-opt's generic-form printer sorts them
// alphabetically, hence the `reductionOperator` slot prints first.
// CHECK:      "acc.reduction.recipe"() <{reductionOperator = #acc.reduction_operator<add>, sym_name = "red_recipe", type = memref<i64>}> ({
// CHECK-NEXT:   ^{{.*}}(%{{.*}}: memref<i64>):
// CHECK-NEXT:     acc.yield %{{.*}} : memref<i64>
// CHECK-NEXT:   }, {
// CHECK-NEXT:   ^{{.*}}(%{{.*}}: memref<i64>, %{{.*}}: memref<i64>):
// CHECK-NEXT:     acc.yield %{{.*}} : memref<i64>
// CHECK-NEXT:   }, {
// CHECK-NEXT:   ^{{.*}}(%{{.*}}: memref<i64>):
// CHECK-NEXT:     "acc.terminator"() : () -> ()
// CHECK-NEXT:   }) : () -> ()

func.func @privatization_data_ops(%a : memref<10xf32>, %b : memref<i64>) {
  %p = acc.private varPtr(%a : memref<10xf32>) -> memref<10xf32> {recipe = @priv_recipe}
  %fp = acc.firstprivate varPtr(%a : memref<10xf32>) -> memref<10xf32> {recipe = @fp_recipe}
  %r = acc.reduction varPtr(%b : memref<i64>) -> memref<i64> {recipe = @red_recipe}
  func.return
}
// CHECK:      func.func @privatization_data_ops(
// CHECK:        %{{.*}} = acc.private varPtr(%{{.*}} : memref<10xf32>) -> memref<10xf32> {recipe = @priv_recipe}
// CHECK-NEXT:   %{{.*}} = acc.firstprivate varPtr(%{{.*}} : memref<10xf32>) -> memref<10xf32> {recipe = @fp_recipe}
// CHECK-NEXT:   %{{.*}} = acc.reduction varPtr(%{{.*}} : memref<i64>) -> memref<i64> {recipe = @red_recipe}
