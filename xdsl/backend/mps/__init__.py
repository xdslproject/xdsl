"""
Apple GPU Backend.

This module provides a complete, system-level compiler backend targeting Apple's
proprietary AIR (Apple Intermediate Representation) bitcode format.

It functions as the final code generation stage in an MLIR-based pipeline,
serving as a "native" GPU backend for Apple Silicon (M-series) processors,
analogous to NVPTX for NVIDIA or AMDGPU for AMD.

## 1. Architecture: Direct-to-Binary

Unlike standard approaches that transpile MLIR to high-level Metal Shading
Language (MSL) source code, this backend lowers directly to the binary format
consumed by the metal driver.

    [ High-Level MLIR ]   (Linalg, GPU, Vector)
            |
            v
    [  LLVM Dialect  ]    (with Apple-specific intrinsics)
            |
            v
    [  AIR Bitcode   ]    (LLVM IR + Metadata)
            |
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

AIR is an undocumented format based on LLVM IR. It differs from standard LLVM
in three critical ways:

*   (1) Intrinsics: It relies heavily on `llvm.air.*` intrinsics for GPU-specific operations.
*   (2) Metadata: It uses a complex schema of named metadata nodes to describe kernel
    signatures, threadgroup dimensions and argument bindings.
*   (3) Conventions: It enforces strict ABI rules regarding address spaces and types.

This module implements the logic to synthesize correct AIR modules, including
all necessary metadata and bitcode encoding, based on clean-room reverse
engineering.
"""
