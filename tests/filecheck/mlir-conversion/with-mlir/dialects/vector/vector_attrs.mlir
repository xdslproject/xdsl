// RUN: XDSL_ROUNDTRIP

// CHECK: #vector.kind<add>
"test.op"() { attrs = [#vector.kind<add>] }: () -> ()
// CHECK: #vector.kind<mul>
"test.op"() { attrs = [#vector.kind<mul>] }: () -> ()
// CHECK: #vector.kind<minui>
"test.op"() { attrs = [#vector.kind<minui>] }: () -> ()
// CHECK: #vector.kind<minsi>
"test.op"() { attrs = [#vector.kind<minsi>] }: () -> ()
// CHECK: #vector.kind<minnumf>
"test.op"() { attrs = [#vector.kind<minnumf>] }: () -> ()
// CHECK: #vector.kind<maxui>
"test.op"() { attrs = [#vector.kind<maxui>] }: () -> ()
// CHECK: #vector.kind<maxsi>
"test.op"() { attrs = [#vector.kind<maxsi>] }: () -> ()
// CHECK: #vector.kind<maxnumf>
"test.op"() { attrs = [#vector.kind<maxnumf>] }: () -> ()
// CHECK: #vector.kind<and>
"test.op"() { attrs = [#vector.kind<and>] }: () -> ()
// CHECK: #vector.kind<or>
"test.op"() { attrs = [#vector.kind<or>] }: () -> ()
// CHECK: #vector.kind<xor>
"test.op"() { attrs = [#vector.kind<xor>] }: () -> ()
// CHECK: #vector.kind<maximumf>
"test.op"() { attrs = [#vector.kind<maximumf>] }: () -> ()
// CHECK: #vector.kind<minimumf>
"test.op"() { attrs = [#vector.kind<minimumf>] }: () -> ()
