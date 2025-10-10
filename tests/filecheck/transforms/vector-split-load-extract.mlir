// RUN: xdsl-opt -p vector-split-load-extract %s | filecheck %s

// CHECK-LABEL: @do_split
func.func @do_split(%ptr : !ptr_xdsl.ptr) {
// CHECK-NEXT:      %vector = arith.constant 12 : index
// CHECK-NEXT:      %vector_1 = ptr_xdsl.ptradd %ptr, %vector : (!ptr_xdsl.ptr, index) -> !ptr_xdsl.ptr
// CHECK-NEXT:      %vector_2 = ptr_xdsl.load %vector_1 : !ptr_xdsl.ptr -> f32
// CHECK-NEXT:      %vector_3 = arith.constant 8 : index
// CHECK-NEXT:      %vector_4 = ptr_xdsl.ptradd %ptr, %vector_3 : (!ptr_xdsl.ptr, index) -> !ptr_xdsl.ptr
// CHECK-NEXT:      %vector_5 = ptr_xdsl.load %vector_4 : !ptr_xdsl.ptr -> f32
// CHECK-NEXT:      %vector_6 = arith.constant 4 : index
// CHECK-NEXT:      %vector_7 = ptr_xdsl.ptradd %ptr, %vector_6 : (!ptr_xdsl.ptr, index) -> !ptr_xdsl.ptr
// CHECK-NEXT:      %vector_8 = ptr_xdsl.load %vector_7 : !ptr_xdsl.ptr -> f32
    %vector = ptr_xdsl.load %ptr : !ptr_xdsl.ptr -> vector<4xf32>
    %f0 = vector.extract %vector[1] : f32 from vector<4xf32>
    %f1 = vector.extract %vector[2] : f32 from vector<4xf32>
    %f2 = vector.extract %vector[3] : f32 from vector<4xf32>

// CHECK-NEXT:      "test.op"(%vector_8, %vector_5, %vector_2) : (f32, f32, f32) -> ()
    "test.op"(%f0, %f1, %f2) : (f32, f32, f32) -> ()
    return
}

// CHECK-LABEL: @do_not_split
func.func @do_not_split(%ptr : !ptr_xdsl.ptr) {
// CHECK-NEXT:      %vector = ptr_xdsl.load %ptr : !ptr_xdsl.ptr -> vector<4xf32>
// CHECK-NEXT:      %f0 = vector.extract %vector[1] : f32 from vector<4xf32>
    %vector = ptr_xdsl.load %ptr : !ptr_xdsl.ptr -> vector<4xf32>
    %f0 = vector.extract %vector[1] : f32 from vector<4xf32>

// CHECK-NEXT:      "test.op"(%f0, %vector) : (f32, vector<4xf32>) -> ()
    "test.op"(%f0, %vector) : (f32, vector<4xf32>) -> ()
    return
}
