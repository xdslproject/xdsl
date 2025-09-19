// RUN: xdsl-opt %s --split-input-file --verify-diagnostics --allow-unregistered-dialect | filecheck %s

mesh.mesh @mesh0(shape = [])

// CHECK: 'mesh.mesh' op rank of mesh is expected to be a positive integer 
