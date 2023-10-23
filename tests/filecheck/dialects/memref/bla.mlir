
  // Private variable with an initial value.
  memref.global "private" @x : memref<2xf32> = dense<0.0>

memref.global "private" @a : memref<2xf32> = dense<0.0> {alignment = 64}

  // Declaration of an external variable.
  memref.global "private" @y : memref<4xi32>

  // Uninitialized externally visible variable.
  memref.global @z : memref<3xf16> = uninitialized

  // Externally visible constant variable.
  memref.global constant @c : memref<2xi32> = dense<1>
