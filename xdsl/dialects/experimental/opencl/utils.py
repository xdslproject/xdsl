def get_ptr_arith_functions(dim: int):
    _1D_to_3D = """
float ***_1D_to_3D(float *array, int D, int R, int C) {
    // Allocate memory for the 3D pointer-to-pointer structure (D x R x C)
    float ***ptr_array = (float ***)malloc(D * sizeof(float **));

    // Allocate memory for each slice (2D array)
    for (int i = 0; i < D; i++) {
        ptr_array[i] = (float **)malloc(R * sizeof(float *));

        // Allocate memory for each row (1D array of columns)
        for (int j = 0; j < R; j++) {
            ptr_array[i][j] = &array[(i * R * C) + (j * C)]; // Point to the correct memory in the 1D array
        }
    }

    // Return the 3D pointer structure
    return ptr_array;
}
"""

    _1D_to_4D = """
float ****_1D_to_4D(float *array, int D, int R, int C, int L) {
    // Allocate memory for the 4D pointer-to-pointer-to-pointer structure (D x R x C x L)
    float ****ptr_array = (float ****)malloc(D * sizeof(float ***));

    // Allocate memory for each depth slice (3D array)
    for (int i = 0; i < D; i++) {
        ptr_array[i] = (float ***)malloc(R * sizeof(float **));

        // Allocate memory for each row slice (2D array)
        for (int j = 0; j < R; j++) {
            ptr_array[i][j] = (float **)malloc(C * sizeof(float *));

            // Allocate memory for each column (1D array)
            for (int k = 0; k < C; k++) {
                ptr_array[i][j][k] = &array[(i * R * C * L) + (j * C * L) + (k * L)];
            }
        }
    }

    // Return the 4D pointer structure
    return ptr_array;
}
"""
    if dim == 3:
        return _1D_to_3D
    elif dim == 4:
        return _1D_to_4D
