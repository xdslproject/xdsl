// RUN: XDSL_ROUNDTRIP
// RUN: MLIR_ROUNDTRIP
// RUN: XDSL_GENERIC_ROUNDTRIP
// RUN: MLIR_GENERIC_ROUNDTRIP

llvm.func @func_vis0_unnameaddr0()

// CHECK: llvm.func @func_vis0_unnameaddr0()
// CHECK-GENERIC: "llvm.func"() <{
// CHECK-GENERIC-DAG: sym_name = "func_vis0_unnameaddr0"
// CHECK-GENERIC-DAG: visibility_ = 0 : i64
// CHECK-GENERIC-DAG: unnamed_addr = 0 : i64
// CHECK-GENERIC: }>

llvm.func hidden @func_vis1_unnameaddr0()

// CHECK: llvm.func hidden @func_vis1_unnameaddr0()
// CHECK-GENERIC: "llvm.func"() <{
// CHECK-GENERIC-DAG: sym_name = "func_vis1_unnameaddr0"
// CHECK-GENERIC-DAG: visibility_ = 1 : i64
// CHECK-GENERIC-DAG: unnamed_addr = 0 : i64
// CHECK-GENERIC: }>

llvm.func protected @func_vis2_unnameaddr0()

// CHECK: llvm.func protected @func_vis2_unnameaddr0()
// CHECK-GENERIC: "llvm.func"() <{
// CHECK-GENERIC-DAG: sym_name = "func_vis2_unnameaddr0"
// CHECK-GENERIC-DAG: visibility_ = 2 : i64
// CHECK-GENERIC-DAG: unnamed_addr = 0 : i64
// CHECK-GENERIC: }>

llvm.func local_unnamed_addr @func_vis0_unnameaddr1()

// CHECK: llvm.func local_unnamed_addr @func_vis0_unnameaddr1()
// CHECK-GENERIC: "llvm.func"() <{
// CHECK-GENERIC-DAG: sym_name = "func_vis0_unnameaddr1"
// CHECK-GENERIC-DAG: visibility_ = 0 : i64
// CHECK-GENERIC-DAG: unnamed_addr = 1 : i64
// CHECK-GENERIC: }>

llvm.func unnamed_addr @func_vis0_unnameaddr2()

// CHECK: llvm.func unnamed_addr @func_vis0_unnameaddr2()
// CHECK-GENERIC: "llvm.func"() <{
// CHECK-GENERIC-DAG: sym_name = "func_vis0_unnameaddr2"
// CHECK-GENERIC-DAG: visibility_ = 0 : i64
// CHECK-GENERIC-DAG: unnamed_addr = 2 : i64
// CHECK-GENERIC: }>

llvm.func hidden local_unnamed_addr @func_vis1_unnameaddr1()

// CHECK: llvm.func hidden local_unnamed_addr @func_vis1_unnameaddr1()
// CHECK-GENERIC: "llvm.func"() <{
// CHECK-GENERIC-DAG: sym_name = "func_vis1_unnameaddr1"
// CHECK-GENERIC-DAG: visibility_ = 1 : i64
// CHECK-GENERIC-DAG: unnamed_addr = 1 : i64
// CHECK-GENERIC: }>

llvm.func hidden unnamed_addr @func_vis1_unnameaddr2()

// CHECK: llvm.func hidden unnamed_addr @func_vis1_unnameaddr2()
// CHECK-GENERIC: "llvm.func"() <{
// CHECK-GENERIC-DAG: sym_name = "func_vis1_unnameaddr2"
// CHECK-GENERIC-DAG: visibility_ = 1 : i64
// CHECK-GENERIC-DAG: unnamed_addr = 2 : i64
// CHECK-GENERIC: }>

llvm.func protected local_unnamed_addr @func_vis2_unnameaddr1()

// CHECK: llvm.func protected local_unnamed_addr @func_vis2_unnameaddr1()
// CHECK-GENERIC: "llvm.func"() <{
// CHECK-GENERIC-DAG: sym_name = "func_vis2_unnameaddr1"
// CHECK-GENERIC-DAG: visibility_ = 2 : i64
// CHECK-GENERIC-DAG: unnamed_addr = 1 : i64
// CHECK-GENERIC: }>

llvm.func protected unnamed_addr @func_vis2_unnameaddr2()

// CHECK: llvm.func protected unnamed_addr @func_vis2_unnameaddr2()
// CHECK-GENERIC: "llvm.func"() <{
// CHECK-GENERIC-DAG: sym_name = "func_vis2_unnameaddr2"
// CHECK-GENERIC-DAG: visibility_ = 2 : i64
// CHECK-GENERIC-DAG: unnamed_addr = 2 : i64
// CHECK-GENERIC: }>

//
// generic format tests
// 
// the comments above each test encode the following values: 
// <in_vis> ┆ <in_unnamed> ┆ <out_vis> ┆ <out_unnamed>
//

// - | - | 0 | -
"llvm.func"() <{
  sym_name = "gen_none_none",
  function_type = !llvm.func<void ()>,
  linkage = #llvm.linkage<external>,
  CConv = #llvm.cconv<ccc>
}> ({
  "llvm.return"() : () -> ()
}) : () -> ()

// CHECK: llvm.func @gen_none_none()
// CHECK-GENERIC: "llvm.func"() <{
// CHECK-GENERIC-DAG: sym_name = "gen_none_none"
// CHECK-GENERIC-DAG: visibility_ = 0 : i64
// CHECK-GENERIC-NOT: unnamed_addr =
// CHECK-GENERIC: }>

// - | 0 | 0 | 0
"llvm.func"() <{
  sym_name = "gen_none_0",
  unnamed_addr = 0 : i64,
  function_type = !llvm.func<void ()>,
  linkage = #llvm.linkage<external>,
  CConv = #llvm.cconv<ccc>
}> ({
  "llvm.return"() : () -> ()
}) : () -> ()

// CHECK: llvm.func @gen_none_0()
// CHECK-GENERIC: "llvm.func"() <{
// CHECK-GENERIC-DAG: sym_name = "gen_none_0"
// CHECK-GENERIC-DAG: visibility_ = 0 : i64
// CHECK-GENERIC-DAG: unnamed_addr = 0 : i64
// CHECK-GENERIC: }>

// - | 1 | 0 | 1
"llvm.func"() <{
  sym_name = "gen_none_1",
  unnamed_addr = 1 : i64,
  function_type = !llvm.func<void ()>,
  linkage = #llvm.linkage<external>,
  CConv = #llvm.cconv<ccc>
}> ({
  "llvm.return"() : () -> ()
}) : () -> ()

// CHECK: llvm.func local_unnamed_addr @gen_none_1()
// CHECK-GENERIC: "llvm.func"() <{
// CHECK-GENERIC-DAG: sym_name = "gen_none_1"
// CHECK-GENERIC-DAG: visibility_ = 0 : i64
// CHECK-GENERIC-DAG: unnamed_addr = 1 : i64
// CHECK-GENERIC: }>

// 0 | - | 0 | -
"llvm.func"() <{
  sym_name = "gen_0_none",
  visibility_ = 0 : i64,
  function_type = !llvm.func<void ()>,
  linkage = #llvm.linkage<external>,
  CConv = #llvm.cconv<ccc>
}> ({
  "llvm.return"() : () -> ()
}) : () -> ()

// CHECK: llvm.func @gen_0_none()
// CHECK-GENERIC: "llvm.func"() <{
// CHECK-GENERIC-DAG: sym_name = "gen_0_none"
// CHECK-GENERIC-DAG: visibility_ = 0 : i64
// CHECK-GENERIC-NOT: unnamed_addr =
// CHECK-GENERIC: }>

// 0 | 0 | 0 | 0
"llvm.func"() <{
  sym_name = "gen_0_0",
  visibility_ = 0 : i64,
  unnamed_addr = 0 : i64,
  function_type = !llvm.func<void ()>,
  linkage = #llvm.linkage<external>,
  CConv = #llvm.cconv<ccc>
}> ({
  "llvm.return"() : () -> ()
}) : () -> ()

// CHECK: llvm.func @gen_0_0()
// CHECK-GENERIC: "llvm.func"() <{
// CHECK-GENERIC-DAG: sym_name = "gen_0_0"
// CHECK-GENERIC-DAG: visibility_ = 0 : i64
// CHECK-GENERIC-DAG: unnamed_addr = 0 : i64
// CHECK-GENERIC: }>

// 0 | 1 | 0 | 1
"llvm.func"() <{
  sym_name = "gen_0_1",
  visibility_ = 0 : i64,
  unnamed_addr = 1 : i64,
  function_type = !llvm.func<void ()>,
  linkage = #llvm.linkage<external>,
  CConv = #llvm.cconv<ccc>
}> ({
  "llvm.return"() : () -> ()
}) : () -> ()

// CHECK: llvm.func local_unnamed_addr @gen_0_1()
// CHECK-GENERIC: "llvm.func"() <{
// CHECK-GENERIC-DAG: sym_name = "gen_0_1"
// CHECK-GENERIC-DAG: visibility_ = 0 : i64
// CHECK-GENERIC-DAG: unnamed_addr = 1 : i64
// CHECK-GENERIC: }>

// 1 | - | 1 | -
"llvm.func"() <{
  sym_name = "gen_1_none",
  visibility_ = 1 : i64,
  function_type = !llvm.func<void ()>,
  linkage = #llvm.linkage<external>,
  CConv = #llvm.cconv<ccc>
}> ({
  "llvm.return"() : () -> ()
}) : () -> ()

// CHECK: llvm.func hidden @gen_1_none()
// CHECK-GENERIC: "llvm.func"() <{
// CHECK-GENERIC-DAG: sym_name = "gen_1_none"
// CHECK-GENERIC-DAG: visibility_ = 1 : i64
// CHECK-GENERIC-NOT: unnamed_addr =
// CHECK-GENERIC: }>

// 1 | 0 | 1 | 0
"llvm.func"() <{
  sym_name = "gen_1_0",
  visibility_ = 1 : i64,
  unnamed_addr = 0 : i64,
  function_type = !llvm.func<void ()>,
  linkage = #llvm.linkage<external>,
  CConv = #llvm.cconv<ccc>
}> ({
  "llvm.return"() : () -> ()
}) : () -> ()

// CHECK: llvm.func hidden @gen_1_0()
// CHECK-GENERIC: "llvm.func"() <{
// CHECK-GENERIC-DAG: sym_name = "gen_1_0"
// CHECK-GENERIC-DAG: visibility_ = 1 : i64
// CHECK-GENERIC-DAG: unnamed_addr = 0 : i64
// CHECK-GENERIC: }>

// 1 | 1 | 1 | 1
"llvm.func"() <{
  sym_name = "gen_1_1",
  visibility_ = 1 : i64,
  unnamed_addr = 1 : i64,
  function_type = !llvm.func<void ()>,
  linkage = #llvm.linkage<external>,
  CConv = #llvm.cconv<ccc>
}> ({
  "llvm.return"() : () -> ()
}) : () -> ()

// CHECK: llvm.func hidden local_unnamed_addr @gen_1_1()
// CHECK-GENERIC: "llvm.func"() <{
// CHECK-GENERIC-DAG: sym_name = "gen_1_1"
// CHECK-GENERIC-DAG: visibility_ = 1 : i64
// CHECK-GENERIC-DAG: unnamed_addr = 1 : i64
// CHECK-GENERIC: }>

//
// custom format tests
//

// - | - | 0 | 0
llvm.func @cust_none_none() {
  llvm.return
}

// CHECK: llvm.func @cust_none_none()
// CHECK-GENERIC: "llvm.func"() <{
// CHECK-GENERIC-DAG: sym_name = "cust_none_none"
// CHECK-GENERIC-DAG: visibility_ = 0 : i64
// CHECK-GENERIC-DAG: unnamed_addr = 0 : i64
// CHECK-GENERIC: }>

// - | 0 | 0 | 0
// Same input as row 10 (no keyword for 0)
llvm.func @cust_none_0() {
  llvm.return
}

// CHECK: llvm.func @cust_none_0()
// CHECK-GENERIC: "llvm.func"() <{
// CHECK-GENERIC-DAG: sym_name = "cust_none_0"
// CHECK-GENERIC-DAG: visibility_ = 0 : i64
// CHECK-GENERIC-DAG: unnamed_addr = 0 : i64
// CHECK-GENERIC: }>

// - | 1 | 0 | 1
llvm.func local_unnamed_addr @cust_none_1() {
  llvm.return
}

// CHECK: llvm.func local_unnamed_addr @cust_none_1()
// CHECK-GENERIC: "llvm.func"() <{
// CHECK-GENERIC-DAG: sym_name = "cust_none_1"
// CHECK-GENERIC-DAG: visibility_ = 0 : i64
// CHECK-GENERIC-DAG: unnamed_addr = 1 : i64
// CHECK-GENERIC: }>

// 0 | - | 0 | 0
// Same input as row 10 (no keyword for 0)
llvm.func @cust_0_none() {
  llvm.return
}

// CHECK: llvm.func @cust_0_none()
// CHECK-GENERIC: "llvm.func"() <{
// CHECK-GENERIC-DAG: sym_name = "cust_0_none"
// CHECK-GENERIC-DAG: visibility_ = 0 : i64
// CHECK-GENERIC-DAG: unnamed_addr = 0 : i64
// CHECK-GENERIC: }>

// 0 | 0 | 0 | 0
// Same input as row 10
llvm.func @cust_0_0() {
  llvm.return
}

// CHECK: llvm.func @cust_0_0()
// CHECK-GENERIC: "llvm.func"() <{
// CHECK-GENERIC-DAG: sym_name = "cust_0_0"
// CHECK-GENERIC-DAG: visibility_ = 0 : i64
// CHECK-GENERIC-DAG: unnamed_addr = 0 : i64
// CHECK-GENERIC: }>

// 0 | 1 | 0 | 1
// Same input as row 12
llvm.func local_unnamed_addr @cust_0_1() {
  llvm.return
}

// CHECK: llvm.func local_unnamed_addr @cust_0_1()
// CHECK-GENERIC: "llvm.func"() <{
// CHECK-GENERIC-DAG: sym_name = "cust_0_1"
// CHECK-GENERIC-DAG: visibility_ = 0 : i64
// CHECK-GENERIC-DAG: unnamed_addr = 1 : i64
// CHECK-GENERIC: }>

// 1 | - | 1 | 0
llvm.func hidden @cust_1_none() {
  llvm.return
}

// CHECK: llvm.func hidden @cust_1_none()
// CHECK-GENERIC: "llvm.func"() <{
// CHECK-GENERIC-DAG: sym_name = "cust_1_none"
// CHECK-GENERIC-DAG: visibility_ = 1 : i64
// CHECK-GENERIC-DAG: unnamed_addr = 0 : i64
// CHECK-GENERIC: }>

// 1 | 0 | 1 | 0
// Same input as row 16
llvm.func hidden @cust_1_0() {
  llvm.return
}

// CHECK: llvm.func hidden @cust_1_0()
// CHECK-GENERIC: "llvm.func"() <{
// CHECK-GENERIC-DAG: sym_name = "cust_1_0"
// CHECK-GENERIC-DAG: visibility_ = 1 : i64
// CHECK-GENERIC-DAG: unnamed_addr = 0 : i64
// CHECK-GENERIC: }>

// 1 | 1 | 1 | 1
llvm.func hidden local_unnamed_addr @cust_1_1() {
  llvm.return
}

// CHECK: llvm.func hidden local_unnamed_addr @cust_1_1()
// CHECK-GENERIC: "llvm.func"() <{
// CHECK-GENERIC-DAG: sym_name = "cust_1_1"
// CHECK-GENERIC-DAG: visibility_ = 1 : i64
// CHECK-GENERIC-DAG: unnamed_addr = 1 : i64
// CHECK-GENERIC: }>
