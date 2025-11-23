from abc import ABC
from collections.abc import Sequence
from dataclasses import dataclass

from xdsl.context import Context
from xdsl.dialects import arith, builtin, riscv, riscv_snitch, snitch_runtime
from xdsl.dialects.builtin import IntegerAttr
from xdsl.ir import Operation, SSAValue
from xdsl.passes import ModulePass
from xdsl.pattern_rewriter import (
    GreedyRewritePatternApplier,
    PatternRewriter,
    PatternRewriteWalker,
    RewritePattern,
    op_type_rewrite_pattern,
)


@dataclass(frozen=True)
class SnrtConstants(ABC):
    """
    Constants used when compiling the snitch runtime, depend on the exact snitch
    architecture target.
    """

    cluster_num: int = 4
    """
    Number of snitch clusters on the chip
    """
    cluster_core_num: int = 9
    """
    Number of cores per cluster (1 dma + 8 compute)
    """
    base_hartid: int = 0
    """
    SNRT expects the snitch cores to be numbered contiguously starting at base_hartid
    """
    cluster_dm_core_num: int = 1
    """
    Number of dm cores per cluster. Compute core num = cluster_core_num - cluster_dm_core_num
    """


class LowerClusterHWBarrier(RewritePattern):
    @op_type_rewrite_pattern
    def match_and_rewrite(
        self, op: snitch_runtime.ClusterHwBarrierOp, rewriter: PatternRewriter, /
    ):
        """
        Lowers to:

        /// Synchronize cores in a cluster with a hardware barrier
        inline void snrt_cluster_hw_barrier() {
            asm volatile("csrr x0, 0x7C2" ::: "memory");
        }
        """
        rewriter.replace_op(
            op,
            [
                zero := riscv.GetRegisterOp(riscv.Registers.ZERO),
                riscv.CsrrsOp(
                    rd=riscv.Registers.ZERO,
                    rs1=zero,
                    csr=IntegerAttr(0x7C2, 12),
                ),
            ],
            [],
        )


class LowerSSRDisable(RewritePattern):
    @op_type_rewrite_pattern
    def match_and_rewrite(
        self, op: snitch_runtime.SsrDisableOp, rewriter: PatternRewriter, /
    ):
        """
        Lowers to:

            /// Disable SSR.
            inline void snrt_ssr_disable() {
            #ifdef __TOOLCHAIN_LLVM__
                __builtin_ssr_disable();
            #else
                asm volatile("csrci 0x7C0, 1\n");
            #endif
            }

        P.S. This specific rewrite ignores the LLVM case and goes
                straight to the generic one.
        """
        rewriter.replace_op(
            op,
            [
                riscv.CsrrciOp(csr=IntegerAttr(0x7C0, 12), immediate=IntegerAttr(1, 4)),
            ],
            [],
        )


class LowerDMAStart1D(RewritePattern):
    @op_type_rewrite_pattern
    def match_and_rewrite(
        self, op: snitch_runtime.DmaStart1DOp, rewriter: PatternRewriter, /
    ):
        """
        Lowers to:

        /// Initiate an asynchronous 1D DMA transfer.
        inline snrt_dma_txid_t snrt_dma_start_1d(void *dst, const void *src,
                                                size_t size) {
            return snrt_dma_start_1d_wideptr((size_t)dst, (size_t)src, size);
        }
        """
        reg_t = riscv.Registers.UNALLOCATED_INT
        rewriter.replace_op(
            op,
            [
                zero := riscv.GetRegisterOp(riscv.Registers.ZERO),
                # "Take a void* (assumed 32bit) and make it a 32 bit-wide RISC-V register"
                i32_dst := builtin.UnrealizedConversionCastOp.get(
                    [op.dst],
                    [reg_t],
                ),
                # "Take a void* (assumed 32bit) and make it a 32 bit-wide RISC-V register"
                i32_src := builtin.UnrealizedConversionCastOp.get(
                    [op.src],
                    [reg_t],
                ),
                # "Convert an IR-level i32 to a RISC-V register"
                i32_size := builtin.UnrealizedConversionCastOp.get(
                    [op.size],
                    [reg_t],
                ),
                riscv_snitch.DMSourceOp(i32_src, zero),
                riscv_snitch.DMDestinationOp(i32_dst, zero),
                copy_imm := riscv_snitch.DMCopyImmOp(i32_size, 0),
                tx_id := builtin.UnrealizedConversionCastOp.get(
                    [copy_imm], [builtin.i32]
                ),
            ],
            new_results=tx_id.results,
        )


class LowerDMAStart1DWidePtr(RewritePattern):
    @op_type_rewrite_pattern
    def match_and_rewrite(
        self, op: snitch_runtime.DmaStart1DWideptrOp, rewriter: PatternRewriter, /
    ):
        """
        Lowers to:

        /// Initiate an asynchronous 1D DMA transfer with wide 64-bit pointers.
        inline snrt_dma_txid_t snrt_dma_start_1d_wideptr(uint64_t dst, uint64_t src,
                                                        size_t size) {
            // Current DMA does not allow transfers with size == 0 (blocks)
            // TODO(colluca) remove this check once new DMA is integrated
            if (size > 0) {
                register uint32_t reg_dst_low asm("a0") = dst >> 0;    // 10
                register uint32_t reg_dst_high asm("a1") = dst >> 32;  // 11
                register uint32_t reg_src_low asm("a2") = src >> 0;    // 12
                register uint32_t reg_src_high asm("a3") = src >> 32;  // 13
                register uint32_t reg_size asm("a4") = size;           // 14

                // dmsrc a2, a3
                asm volatile(
                    ".word (0b0000000 << 25) | \
                        (     (13) << 20) | \
                        (     (12) << 15) | \
                        (    0b000 << 12) | \
                        (0b0101011 <<  0)   \n" ::"r"(reg_src_high),
                    "r"(reg_src_low));

                // dmdst a0, a1
                asm volatile(
                    ".word (0b0000001 << 25) | \
                        (     (11) << 20) | \
                        (     (10) << 15) | \
                        (    0b000 << 12) | \
                        (0b0101011 <<  0)   \n" ::"r"(reg_dst_high),
                    "r"(reg_dst_low));

                // dmcpyi a0, a4, 0b00
                register uint32_t reg_txid asm("a0");  // 10
                asm volatile(
                    ".word (0b0000010 << 25) | \
                        (  0b00000 << 20) | \
                        (     (14) << 15) | \
                        (    0b000 << 12) | \
                        (     (10) <<  7) | \
                        (0b0101011 <<  0)   \n"
                    : "=r"(reg_txid)
                    : "r"(reg_size));

                return reg_txid;
            } else {
                return -1;
            }
        }

        P.S. We only implement taking the top branch for now.
        """
        reg_t = riscv.Registers.UNALLOCATED_INT
        rewriter.replace_op(
            op,
            [
                # "Take an ui64 and split it in two 32 bit-wide RISC-V registers"
                split_i64_dst := builtin.UnrealizedConversionCastOp.get(
                    [op.dst],
                    [reg_t, reg_t],
                ),
                # "Take an ui64 and split it in two 32 bit-wide RISC-V registers"
                split_i64_src := builtin.UnrealizedConversionCastOp.get(
                    [op.src],
                    [reg_t, reg_t],
                ),
                # "Convert an IR-level i32 to a RISC-V register"
                i32_size := builtin.UnrealizedConversionCastOp.get(
                    [op.size],
                    [reg_t],
                ),
                riscv_snitch.DMSourceOp(
                    split_i64_src.results[0], split_i64_src.results[1]
                ),
                riscv_snitch.DMDestinationOp(
                    split_i64_dst.results[0], split_i64_dst.results[1]
                ),
                copy_imm := riscv_snitch.DMCopyImmOp(i32_size, 0),
                tx_id := builtin.UnrealizedConversionCastOp.get(
                    [copy_imm], [builtin.i32]
                ),
            ],
            new_results=tx_id.results,
        )


class LowerDMAStart2DBase(RewritePattern, ABC):
    any_reg = riscv.Registers.UNALLOCATED_INT

    def generate_dma_instructions(
        self,
        dst_low: SSAValue | Operation,
        dst_high: SSAValue | Operation,
        src_low: SSAValue | Operation,
        src_high: SSAValue | Operation,
        size: SSAValue | Operation,
        dst_stride: SSAValue | Operation,
        src_stride: SSAValue | Operation,
        repeat: SSAValue | Operation,
    ) -> tuple[Sequence[Operation], Sequence[SSAValue]]:
        """
        Common function to generate the following sequence of operations:
            dmsrc %src_low, %src_high
            dmdst %dst_low, %dst_high
            dmstr %src_stride, %dst_stride
            dmrep %repeat
            %tx_id = dmcpyi %size, 0b10
            %tx_id_i32 unrealized_conversion_cast %tx_id to i32
        """

        return [
            riscv_snitch.DMSourceOp(src_low, src_high),
            riscv_snitch.DMDestinationOp(dst_low, dst_high),
            riscv_snitch.DMStrideOp(src_stride, dst_stride),
            riscv_snitch.DMRepOp(repeat),
            cpy_op := riscv_snitch.DMCopyImmOp(size, 0b10),
            tx_id := builtin.UnrealizedConversionCastOp.get([cpy_op], [builtin.i32]),
        ], tx_id.results

    def cast_i32(self, input_val: SSAValue):
        """
        Cast an i32 to riscv registers
        """
        return builtin.UnrealizedConversionCastOp.get([input_val], [self.any_reg])

    def cast_i64(self, input_val: SSAValue):
        """
        Cast an i64 to two riscv registers
        """
        return builtin.UnrealizedConversionCastOp.get(
            [input_val], [self.any_reg, self.any_reg]
        )


class LowerDMAStart2DWideptr(LowerDMAStart2DBase):
    @op_type_rewrite_pattern
    def match_and_rewrite(
        self, op: snitch_runtime.DmaStart2DWideptrOp, rewriter: PatternRewriter, /
    ):
        """
        Lowers to the equivalent of the snitch_runtime implementation:

        inline snrt_dma_txid_t snrt_dma_start_2d_wideptr(uint64_t dst, uint64_t src,
                                                         size_t size, size_t dst_stride,
                                                         size_t src_stride,
                                                         size_t repeat) {
            // Current DMA does not allow transfers with size == 0 (blocks)
            // TODO(colluca) remove this check once new DMA is integrated
            if (size > 0) {
                register uint32_t reg_dst_low asm("a0") = dst >> 0;       // 10
                register uint32_t reg_dst_high asm("a1") = dst >> 32;     // 11
                register uint32_t reg_src_low asm("a2") = src >> 0;       // 12
                register uint32_t reg_src_high asm("a3") = src >> 32;     // 13
                register uint32_t reg_size asm("a4") = size;              // 14
                register uint32_t reg_dst_stride asm("a5") = dst_stride;  // 15
                register uint32_t reg_src_stride asm("a6") = src_stride;  // 16
                register uint32_t reg_repeat asm("a7") = repeat;          // 17

                // dmsrc a0, a1
                asm volatile(
                    ".word (0b0000000 << 25) | \
                        (     (13) << 20) | \
                        (     (12) << 15) | \
                        (    0b000 << 12) | \
                        (0b0101011 <<  0)   \n" ::"r"(reg_src_high),
                    "r"(reg_src_low));

                // dmdst a0, a1
                asm volatile(
                    ".word (0b0000001 << 25) | \
                        (     (11) << 20) | \
                        (     (10) << 15) | \
                        (    0b000 << 12) | \
                        (0b0101011 <<  0)   \n" ::"r"(reg_dst_high),
                    "r"(reg_dst_low));

                // dmstr a5, a6
                asm volatile(
                    ".word (0b0000110 << 25) | \
                        (     (15) << 20) | \
                        (     (16) << 15) | \
                        (    0b000 << 12) | \
                        (0b0101011 <<  0)   \n"
                    :
                    : "r"(reg_dst_stride), "r"(reg_src_stride));

                // dmrep a7
                asm volatile(
                    ".word (0b0000111 << 25) | \
                        (     (17) << 15) | \
                        (    0b000 << 12) | \
                        (0b0101011 <<  0)   \n"
                    :
                    : "r"(reg_repeat));

                // dmcpyi a0, a4, 0b10
                register uint32_t reg_txid asm("a0");  // 10
                asm volatile(
                    ".word (0b0000010 << 25) | \
                        (  0b00010 << 20) | \
                        (     (14) << 15) | \
                        (    0b000 << 12) | \
                        (     (10) <<  7) | \
                        (0b0101011 <<  0)   \n"
                    : "=r"(reg_txid)
                    : "r"(reg_size));

                return reg_txid;
            } else {
                return -1;
            }
        }
        """
        rewriter.insert_op(
            [
                dst := self.cast_i64(op.dst),
                src := self.cast_i64(op.src),
                src_stride := self.cast_i32(op.src_stride),
                dst_stride := self.cast_i32(op.dst_stride),
                size := self.cast_i32(op.size),
                repeat := self.cast_i32(op.size),
            ]
        )
        rewriter.replace_op(
            op,
            *self.generate_dma_instructions(
                dst.results[0],
                dst.results[1],
                src.results[0],
                src.results[1],
                size,
                dst_stride,
                src_stride,
                repeat,
            ),
        )


class LowerDMAStart2D(LowerDMAStart2DBase):
    @op_type_rewrite_pattern
    def match_and_rewrite(
        self, op: snitch_runtime.DmaStart2DOp, rewriter: PatternRewriter, /
    ):
        """
        Lower to the equivalent snitch_runtime implementation:

        /// Initiate an asynchronous 2D DMA transfer.
        inline snrt_dma_txid_t snrt_dma_start_2d(void *dst, const void *src,
                                                 size_t size, size_t dst_stride,
                                                 size_t src_stride, size_t repeat) {
            return snrt_dma_start_2d_wideptr((size_t)dst, (size_t)src, size, dst_stride,
                                             src_stride, repeat);
        }
        """
        rewriter.insert_op(
            [
                # we use zero register for the ptr_high registers
                zero := riscv.GetRegisterOp(riscv.Registers.ZERO),
                dst := self.cast_i32(op.dst),
                src := self.cast_i32(op.src),
                src_stride := self.cast_i32(op.src_stride),
                dst_stride := self.cast_i32(op.dst_stride),
                size := self.cast_i32(op.size),
                repeat := self.cast_i32(op.size),
            ]
        )
        # generate the dma setup instructions with `zero` for the ptr_high values
        rewriter.replace_op(
            op,
            *self.generate_dma_instructions(
                dst,
                zero,
                src,
                zero,
                size,
                dst_stride,
                src_stride,
                repeat,
            ),
        )


@dataclass
class LowerGlobalCoreBaseHartid(RewritePattern):
    constants: SnrtConstants

    @op_type_rewrite_pattern
    def match_and_rewrite(
        self, op: snitch_runtime.GlobalCoreBaseHartidOp, rewriter: PatternRewriter, /
    ):
        rewriter.replace_op(
            op,
            [
                arith.ConstantOp.from_int_and_width(
                    self.constants.base_hartid, builtin.i32
                )
            ],
        )


@dataclass
class LowerGlobalCoreNum(RewritePattern):
    constants: SnrtConstants

    @op_type_rewrite_pattern
    def match_and_rewrite(
        self, op: snitch_runtime.GlobalCoreNumOp, rewriter: PatternRewriter, /
    ):
        rewriter.replace_op(
            op,
            [
                arith.ConstantOp.from_int_and_width(
                    self.constants.cluster_num * self.constants.cluster_core_num,
                    builtin.i32,
                )
            ],
        )


@dataclass
class LowerClusterCoreNum(RewritePattern):
    constants: SnrtConstants

    @op_type_rewrite_pattern
    def match_and_rewrite(
        self, op: snitch_runtime.ClusterCoreNumOp, rewriter: PatternRewriter, /
    ):
        rewriter.replace_op(
            op,
            [
                arith.ConstantOp.from_int_and_width(
                    self.constants.cluster_core_num, builtin.i32
                )
            ],
        )


@dataclass
class LowerClusterNum(RewritePattern):
    constants: SnrtConstants

    @op_type_rewrite_pattern
    def match_and_rewrite(
        self, op: snitch_runtime.ClusterNumOp, rewriter: PatternRewriter, /
    ):
        rewriter.replace_op(
            op,
            [
                arith.ConstantOp.from_int_and_width(
                    self.constants.cluster_num, builtin.i32
                )
            ],
        )


@dataclass
class LowerClusterDmCoreNum(RewritePattern):
    constants: SnrtConstants

    @op_type_rewrite_pattern
    def match_and_rewrite(
        self, op: snitch_runtime.ClusterDmCoreNumOp, rewriter: PatternRewriter, /
    ):
        rewriter.replace_op(
            op,
            [
                arith.ConstantOp.from_int_and_width(
                    self.constants.cluster_dm_core_num, builtin.i32
                )
            ],
        )


class LowerIsComputeCore(RewritePattern):
    @op_type_rewrite_pattern
    def match_and_rewrite(
        self, op: snitch_runtime.IsComputeCoreOp, rewriter: PatternRewriter, /
    ):
        """
        inline int __attribute__((const)) snrt_is_compute_core() {
            return snrt_cluster_core_idx() < snrt_cluster_compute_core_num();
        }
        """
        rewriter.replace_op(
            op,
            [
                cluster_core_idx := snitch_runtime.ClusterCoreIdxOp(),
                compute_core_num := snitch_runtime.ClusterComputeCoreNumOp(),
                arith.CmpiOp(cluster_core_idx, compute_core_num, "slt"),
            ],
        )


class LowerIsDmCore(RewritePattern):
    @op_type_rewrite_pattern
    def match_and_rewrite(
        self, op: snitch_runtime.IsDmCoreOp, rewriter: PatternRewriter, /
    ):
        """
        inline int __attribute__((const)) snrt_is_compute_core() {
            return snrt_cluster_core_idx() < snrt_cluster_compute_core_num();
        }

        inline int __attribute__((const)) snrt_is_dm_core() {
            return !snrt_is_compute_core();
        }
        """
        rewriter.replace_op(
            op,
            [
                cluster_core_idx := snitch_runtime.ClusterCoreIdxOp(),
                compute_core_num := snitch_runtime.ClusterComputeCoreNumOp(),
                arith.CmpiOp(cluster_core_idx, compute_core_num, "sge"),
            ],
        )


class LowerClusterCoreIdx(RewritePattern):
    @op_type_rewrite_pattern
    def match_and_rewrite(
        self, op: snitch_runtime.ClusterCoreIdxOp, rewriter: PatternRewriter, /
    ):
        """
        inline uint32_t __attribute__((const)) snrt_cluster_core_idx() {
            return snrt_global_core_idx() % snrt_cluster_core_num();
        }
        """
        rewriter.replace_op(
            op,
            [
                global_core_idx := snitch_runtime.GlobalCoreIdxOp(),
                cluster_core_num := snitch_runtime.ClusterCoreNumOp(),
                arith.RemSIOp(global_core_idx, cluster_core_num),
            ],
        )


@dataclass
class LowerClusterComputeCoreNum(RewritePattern):
    constants: SnrtConstants

    @op_type_rewrite_pattern
    def match_and_rewrite(
        self, op: snitch_runtime.ClusterComputeCoreNumOp, rewriter: PatternRewriter, /
    ):
        """
        Implementation:

        inline uint32_t __attribute__((const)) snrt_cluster_compute_core_num() {
            return snrt_cluster_core_num() - snrt_cluster_dm_core_num();
        }

        inline uint32_t __attribute__((const)) snrt_cluster_core_num() {
            return SNRT_CLUSTER_CORE_NUM;
        }

        inline uint32_t __attribute__((const)) snrt_cluster_dm_core_num() {
            return SNRT_CLUSTER_DM_CORE_NUM;
        }
        """
        rewriter.replace_op(
            op,
            [
                arith.ConstantOp.from_int_and_width(
                    self.constants.cluster_core_num
                    - self.constants.cluster_dm_core_num,
                    builtin.i32,
                )
            ],
        )


@dataclass
class LowerGlobalCoreIdx(RewritePattern):
    constants: SnrtConstants

    @op_type_rewrite_pattern
    def match_and_rewrite(
        self, op: snitch_runtime.GlobalCoreIdxOp, rewriter: PatternRewriter, /
    ):
        """
        Implementation:

        inline uint32_t __attribute__((const)) snrt_hartid() {
            uint32_t hartid;
            asm("csrr %0, mhartid" : "=r"(hartid));
            return hartid;
        }

        inline uint32_t __attribute__((const)) snrt_global_core_idx() {
            return snrt_hartid() - snrt_global_core_base_hartid();
        }
        """
        rewriter.replace_op(
            op,
            [
                zero := riscv.GetRegisterOp(riscv.Registers.ZERO),
                hartid := riscv.CsrrsOp(zero, IntegerAttr(0xF14, 12), readonly=True),
                hartid_i32 := builtin.UnrealizedConversionCastOp.get(
                    [hartid], [builtin.i32]
                ),
                base_hartid := arith.ConstantOp.from_int_and_width(
                    self.constants.base_hartid, builtin.i32
                ),
                arith.SubiOp(hartid_i32, base_hartid),
            ],
        )


class LowerClusterIdx(RewritePattern):
    @op_type_rewrite_pattern
    def match_and_rewrite(
        self, op: snitch_runtime.ClusterIdxOp, rewriter: PatternRewriter, /
    ):
        """
        Implementation:

        inline uint32_t __attribute__((const)) snrt_cluster_idx() {
            return snrt_global_core_idx() / snrt_cluster_core_num();
        }
        """
        rewriter.replace_op(
            op,
            [
                cluster_core_num := snitch_runtime.ClusterCoreNumOp(),
                core_idx := snitch_runtime.GlobalCoreIdxOp(),
                arith.DivSIOp(
                    core_idx,
                    cluster_core_num,
                ),
            ],
        )


@dataclass(frozen=True)
class InlineSnrtPass(SnrtConstants, ModulePass):
    """
    Inline operations of the snrt dialect to their definitions.

    Uses arith operations where possible, and emit `riscv.csr*` and `riscv_snitch` ops where required.
    """

    name = "inline-snrt"

    def apply(self, ctx: Context, op: builtin.ModuleOp) -> None:
        PatternRewriteWalker(
            GreedyRewritePatternApplier(
                [
                    LowerClusterHWBarrier(),
                    LowerSSRDisable(),
                    LowerDMAStart1D(),
                    LowerDMAStart1DWidePtr(),
                    LowerDMAStart2D(),
                    LowerDMAStart2DWideptr(),
                    # information getting ops:
                    LowerClusterIdx(),
                    LowerClusterNum(self),
                    LowerClusterCoreIdx(),
                    LowerClusterCoreNum(self),
                    LowerClusterDmCoreNum(self),
                    LowerClusterComputeCoreNum(self),
                    LowerGlobalCoreNum(self),
                    LowerGlobalCoreIdx(self),
                    LowerGlobalCoreBaseHartid(self),
                    LowerIsComputeCore(),
                    LowerIsDmCore(),
                ]
            )
        ).rewrite_module(op)
