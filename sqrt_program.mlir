func.func @f(%x : f32) -> f32 {
  %x_1 = eqsat.eclass %x : f32
  %one = arith.constant 1.000000e+00 : f32
  %one_1 = eqsat.eclass %one : f32
  %x_add_one = arith.addf %x_1, %one_1 : f32
  %x_add_one_1 = eqsat.eclass %x_add_one : f32
  %sqrt_x_add_one = math.sqrt %x_add_one_1 : f32
  %sqrt_x_add_one_1 = eqsat.eclass %sqrt_x_add_one : f32
  %sqrt_x = math.sqrt %x_1 : f32
  %sqrt_x_1 = eqsat.eclass %sqrt_x : f32
  %res = arith.subf %sqrt_x_add_one_1, %sqrt_x_1 : f32
  %res_1 = eqsat.eclass %res : f32
  func.return %res_1 : f32
}
