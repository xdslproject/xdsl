
%v0, %v1 = "test.op"() : () -> (index, index)

%c0 = arith.constant 0 : index
%c1 = arith.constant 1 : index
%c2 = arith.constant 2 : index
%c3 = arith.constant 3 : index

%r00, %r01 = scf.for %i0 = %v0 to %v1 step %v0 iter_args(%arg00 = %v0, %arg01 = %v1) -> (index, index) {
    yield %arg00, %arg01 : index, index
}
"test.op"(%r00, %r01) : (index, index) -> ()

%r10, %r11 = scf.for %i0 = %c1 to %c1 step %v0 iter_args(%arg10 = %v0, %arg11 = %v1) -> (index, index) {
    yield %arg10, %arg11 : index, index
}
"test.op"(%r10, %r11) : (index, index) -> ()

%r20, %r21 = scf.for %i0 = %c2 to %c1 step %v0 iter_args(%arg20 = %v0, %arg21 = %v1) -> (index, index) {
    yield %arg20, %arg21 : index, index
}
"test.op"(%r20, %r21) : (index, index) -> ()

%r30, %r31 = scf.for %i0 = %c1 to %c3 step %c2 iter_args(%arg30 = %v0, %arg31 = %v1) -> (index, index) {
    "test.op"() {"hello"} : () -> ()
    yield %arg30, %arg31 : index, index
}
"test.op"(%r30, %r31) : (index, index) -> ()
