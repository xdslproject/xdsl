from xdsl.dialects import builtin, riscv, riscv_snitch, snitch_runtime
from xdsl.dialects.builtin import IntegerAttr
from xdsl.ir import MLContext
from xdsl.passes import ModulePass
from xdsl.pattern_rewriter import (
    GreedyRewritePatternApplier,
    PatternRewriter,
    PatternRewriteWalker,
    RewritePattern,
    op_type_rewrite_pattern,
)


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
        rewriter.replace_matched_op(
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
        rewriter.replace_matched_op(
            [riscv.CsrrciOp(csr=IntegerAttr(0x7C0, 12), immediate=IntegerAttr(1, 4))],
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
        reg_t = riscv.IntRegisterType.unallocated()
        rewriter.replace_matched_op(
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
        reg_t = riscv.IntRegisterType.unallocated()
        rewriter.replace_matched_op(
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


class ConvertSnrtToRISCV(ModulePass):
    name = "convert-snrt-to-riscv-asm"

    def apply(self, ctx: MLContext, op: builtin.ModuleOp) -> None:
        PatternRewriteWalker(
            GreedyRewritePatternApplier(
                [
                    LowerClusterHWBarrier(),
                    LowerSSRDisable(),
                    LowerDMAStart1D(),
                    LowerDMAStart1DWidePtr(),
                ]
            )
        ).rewrite_module(op)
