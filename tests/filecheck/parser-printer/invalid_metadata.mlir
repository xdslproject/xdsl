// RUN: xdsl-opt %s --parsing-diagnostics --split-input-file | filecheck %s

// CHECK: Expected a resource type key
{-#

// -----

// CHECK: expected `:`
{-#
  key
#-}

// -----

// CHECK: got an unexpected key in file metadata: some_key
{-#
  some_key: {}
#-}

// -----

//===----------------------------------------------------------------------===//
// `dialect_resources`
//===----------------------------------------------------------------------===//

// CHECK: '{' expected
{-#
  dialect_resources: "value"
#-}

// -----

// CHECK: Expected a dialect name
{-#
  dialect_resources: {
    10
  }
#-}

// -----

// CHECK: expected `:`
{-#
  dialect_resources: {
    entry "value"
  }
#-}

// -----

// CHECK: dialect foobar is not registered
{-#
  dialect_resources: {
    foobar: {
      entry: "foo"
    }
  }
#-}

// -----

// CHECK: string literal expected
{-#
  dialect_resources: {
    test: {
      invalid_blob: 10
    }
  }
#-}

// -----

// CHECK: got an error when parsing a resource: Blob must be a hex string, got: abc
{-#
  dialect_resources: {
    test: {
      invalid_blob: "abc"
    }
  }
#-}

// -----

// CHECK: dialect arith doesn't have an OpAsmDialectInterface interface
{-#
  dialect_resources: {
    arith: {
      key: "0x1"
    }
  }
#-}
