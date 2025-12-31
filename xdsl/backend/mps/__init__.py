"""
Apple GPU Backend.

This module provides a complete, system-level compiler backend targeting Apple's
proprietary AIR (Apple Intermediate Representation) bitcode format.

It functions as the final code generation stage in an MLIR-based pipeline,
serving as a "native" GPU backend for Apple Silicon (M-series) processors,
analogous to NVPTX for NVIDIA or AMDGPU for AMD.

## 1. Architecture: The MPS Dialect

Unlike standard approaches, this backend introduces a dedicated **MPS Dialect**
that acts as a structured, in-memory representation of Apple's proprietary IR.

    [ High-Level MLIR ]   (Linalg, GPU, Vector)
            |
            v
    [   MPS Dialect   ]   (Reverse-Engineered AIR Model)
            |
            v
    [  AIR Bitcode   ]    (Serialized LLVM IR + Metadata)
            .
            . (External Linkage via xcrun)
            v
    [  Metallib      ]    (Standard Metal Library Resource)

## 2. Rationale: Why Bypass MSL?

The standard Metal toolchain enforces a "black box" compilation step (MSL -> AIR)
that obscures the hardware reality from the compiler frontend. By generating AIR
directly, this backend reclaims control:

*   (1) Instruction Selection: We can emit specific opcodes (e.g., simd_shuffle,
    threadgroup_barrier) that might not have direct MSL equivalents or whose
    generation from MSL is unpredictable.
*   (2) Predictability: Bypassing the aggressive high-level MSL optimizer guarantees
    that the code structure defined in MLIR is preserved in the final binary,
    crucial for performance tuning.
*   (3) Compilation Latency: Skipping the C++ parsing and frontend phases of the
    Metal compiler significantly reduces JIT compilation time for dynamic workloads.

## 3. The Target: Apple Intermediate Representation (AIR)

AIR is an undocumented format based on LLVM IR. The backend emits this bitcode
directly. The user is responsible for packaging it into a `.metallib` using
Apple's command-line tools: `xcrun metallib` or `xcrun metal`.

It differs from standard LLVM in three critical ways:

*   (1) Intrinsics: It relies heavily on `llvm.air.*` intrinsics for GPU-specific operations.
*   (2) Metadata: It uses a complex schema of named metadata nodes to describe kernel
    signatures, threadgroup dimensions and argument bindings.
*   (3) Conventions: It enforces strict ABI rules regarding address spaces and types.

This module implements the logic to synthesize correct AIR modules, including
all necessary metadata and bitcode encoding, based on clean-room reverse
engineering.
"""
