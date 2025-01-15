program = """
    // Load kernel binary
    size_t binary_size;
    #ifdef SW_EMU
    unsigned char* binary = read_binary_file("nodes-sw_emu.xclbin", &binary_size);
    #elif HW_EMU
    unsigned char* binary = read_binary_file("nodes-hw_emu.xclbin", &binary_size);
    #endif

    cl_program program = clCreateProgramWithBinary(
            context,
            1,
            &device,
            &binary_size,
            (const unsigned char**)&binary,
            NULL,
            &status
        );
"""

####################################################################################################

platform = """
cl_platform_id platform;
clGetPlatformIDs(1, &platform, NULL);


char platform_name[1024]; // Buffer to hold the platform name

// Get platform ID
err = clGetPlatformIDs(1, &platform, NULL);
if (err != CL_SUCCESS) {
    printf("Error getting platform ID: %d\\n", err);
    return 1;
}

// Get the platform name
err = clGetPlatformInfo(platform, CL_PLATFORM_NAME, sizeof(platform_name), platform_name, NULL);
if (err != CL_SUCCESS) {
    printf("Error getting platform info: %d\\n", err);
    return 1;
}

// Print the platform name
printf("Platform Name: %s\\n", platform_name);
"""

####################################################################################################

device = """
    cl_device_id device;
    clGetDeviceIDs(platform, CL_DEVICE_TYPE_ACCELERATOR, 1, &device, NULL);
"""

####################################################################################################

context = """
cl_context context = clCreateContext(NULL, 1, &device, NULL, NULL, &err);
"""
