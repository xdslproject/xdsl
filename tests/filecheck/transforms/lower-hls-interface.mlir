// RUN: xdsl-opt %s -p lower-hls | filecheck %s

module {
  func.func @kernel(%in: !llvm.ptr, %out: !llvm.ptr) -> i32 {
    "hls.interface"(%in) {
      mode = "m_axi",
      bundle = "gmem0",
      offset = "slave",
      depth = 16 : i32
    } : (!llvm.ptr) -> ()
    "hls.interface"(%out) {
      mode = "m_axi",
      bundle = "gmem1",
      offset = "slave",
      depth = 16 : i32
    } : (!llvm.ptr) -> ()
    %c0 = arith.constant 0 : i32
    return %c0 : i32
  }
}

// CHECK:      func.func @kernel(%in: !llvm.ptr, %out: !llvm.ptr) -> i32 {
// CHECK-NEXT:   func.call @_interface_maxi_gmem0(%in) : (!llvm.ptr) -> ()
// CHECK-NEXT:   func.call @_interface_maxi_gmem1(%out) : (!llvm.ptr) -> ()
// CHECK:      func.func private @_interface_maxi_gmem0(!llvm.ptr) -> ()
// CHECK-NEXT: func.func private @_interface_maxi_gmem1(!llvm.ptr) -> ()
