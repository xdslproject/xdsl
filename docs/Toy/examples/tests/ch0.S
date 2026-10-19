# RUN: riscemu -o unlimited_regs,libc,ignore_exit_code %s | filecheck %s

.text
.globl main
.p2align 2
main:
    li j_0, 48                                   # memref alloc size
    mv a0, j_0
    jal malloc
    mv j_0, a0
    li j_1, 1
    fcvt.d.w fj_5, j_1
    li j_1, 2
    fcvt.d.w fj_4, j_1
    li j_1, 3
    fcvt.d.w fj_3, j_1
    li j_1, 4
    fcvt.d.w fj_2, j_1
    li j_1, 5
    fcvt.d.w fj_1, j_1
    li j_1, 6
    fcvt.d.w fj_0, j_1
    li j_1, 48                                   # memref alloc size
    mv a0, j_1
    jal malloc
    mv j_1, a0
    li j_2, 1
    fcvt.d.w fj_11, j_2
    li j_2, 2
    fcvt.d.w fj_10, j_2
    li j_2, 3
    fcvt.d.w fj_9, j_2
    li j_2, 4
    fcvt.d.w fj_8, j_2
    li j_2, 5
    fcvt.d.w fj_7, j_2
    li j_2, 6
    fcvt.d.w fj_6, j_2
    li j_2, 48                                   # memref alloc size
    mv a0, j_2
    jal malloc
    mv j_2, a0
    li j_3, 1
    fcvt.d.w fj_17, j_3
    li j_3, 2
    fcvt.d.w fj_16, j_3
    li j_3, 3
    fcvt.d.w fj_15, j_3
    li j_3, 4
    fcvt.d.w fj_14, j_3
    li j_3, 5
    fcvt.d.w fj_13, j_3
    li j_3, 6
    fcvt.d.w fj_12, j_3
    li j_3, 48                                   # memref alloc size
    mv a0, j_3
    jal malloc
    mv j_3, a0
    mv j_5, j_3
    fsd fj_17, 0(j_5)                            # store double value to memref of shape (2, 3)
    fsd fj_16, 8(j_3)                            # store double value to memref of shape (2, 3)
    fsd fj_15, 16(j_3)                           # store double value to memref of shape (2, 3)
    fsd fj_14, 24(j_3)                           # store double value to memref of shape (2, 3)
    fsd fj_13, 32(j_3)                           # store double value to memref of shape (2, 3)
    fsd fj_12, 40(j_3)                           # store double value to memref of shape (2, 3)
    mv j_5, j_2
    fsd fj_11, 0(j_5)                            # store double value to memref of shape (3, 2)
    fsd fj_10, 8(j_2)                            # store double value to memref of shape (3, 2)
    fsd fj_9, 16(j_2)                            # store double value to memref of shape (3, 2)
    fsd fj_8, 24(j_2)                            # store double value to memref of shape (3, 2)
    fsd fj_7, 32(j_2)                            # store double value to memref of shape (3, 2)
    fsd fj_6, 40(j_2)                            # store double value to memref of shape (3, 2)
    mv j_5, j_2
    fld fj_6, 0(j_5)                             # load double from memref of shape (3, 2)
    fld fj_7, 8(j_2)                             # load double from memref of shape (3, 2)
    fld fj_8, 16(j_2)                            # load double from memref of shape (3, 2)
    fld fj_9, 24(j_2)                            # load double from memref of shape (3, 2)
    fld fj_10, 32(j_2)                           # load double from memref of shape (3, 2)
    fld fj_11, 40(j_2)                           # load double from memref of shape (3, 2)
    printf "[[{}, {}], [{}, {}], [{}, {}]]", fj_6, fj_7, fj_8, fj_9, fj_10, fj_11
    mv j_5, j_1
    fsd fj_5, 0(j_5)                             # store double value to memref of shape (2, 3)
    fsd fj_4, 8(j_1)                             # store double value to memref of shape (2, 3)
    fsd fj_3, 16(j_1)                            # store double value to memref of shape (2, 3)
    fsd fj_2, 24(j_1)                            # store double value to memref of shape (2, 3)
    fsd fj_1, 32(j_1)                            # store double value to memref of shape (2, 3)
    fsd fj_0, 40(j_1)                            # store double value to memref of shape (2, 3)
    li j_5, 2
    li j_6, 1
    mv j_4, zero
    bge j_4, j_5, scf_body_end_1_for
scf_body_1_for:
    li j_8, 3
    li j_9, 1
    mv j_7, zero
    bge j_7, j_8, scf_body_end_0_for
scf_body_0_for:
    li j_10, 3
    mul j_10, j_4, j_10
    add j_10, j_10, j_7
    li j_11, 8
    mul j_10, j_10, j_11                         # multiply by element size
    add j_10, j_3, j_10
    fld fj_0, 0(j_10)                            # load double from memref of shape (2, 3)
    li j_10, 3
    mul j_10, j_4, j_10
    add j_10, j_10, j_7
    li j_11, 8
    mul j_10, j_10, j_11                         # multiply by element size
    add j_10, j_1, j_10
    fld fj_1, 0(j_10)                            # load double from memref of shape (2, 3)
    fadd.d fj_0, fj_0, fj_1
    li j_10, 3
    mul j_10, j_4, j_10
    add j_10, j_10, j_7
    li j_11, 8
    mul j_10, j_10, j_11                         # multiply by element size
    add j_10, j_0, j_10
    fsd fj_0, 0(j_10)                            # store double value to memref of shape (2, 3)
    add j_7, j_7, j_9
    blt j_7, j_8, scf_body_0_for
scf_body_end_0_for:
    add j_4, j_4, j_6
    blt j_4, j_5, scf_body_1_for
scf_body_end_1_for:
    mv j_4, j_0
    fld fj_0, 0(j_4)                             # load double from memref of shape (2, 3)
    fld fj_1, 8(j_0)                             # load double from memref of shape (2, 3)
    fld fj_2, 16(j_0)                            # load double from memref of shape (2, 3)
    fld fj_3, 24(j_0)                            # load double from memref of shape (2, 3)
    fld fj_4, 32(j_0)                            # load double from memref of shape (2, 3)
    fld fj_5, 40(j_0)                            # load double from memref of shape (2, 3)
    printf "[[{}, {}, {}], [{}, {}, {}]]", fj_0, fj_1, fj_2, fj_3, fj_4, fj_5
    mv a0, j_3
    jal free
    mv a0, j_2
    jal free
    mv a0, j_1
    jal free
    mv a0, j_0
    jal free
    li a7, 93
    ecall
    ret

#      CHECK:  [
# CHECK-SAME:   [1.0, 2.0], [3.0, 4.0], [5.0, 6.0]]
#      CHECK:  [
# CHECK-SAME:   [2.0, 4.0, 6.0], [8.0, 10.0, 12.0]]
