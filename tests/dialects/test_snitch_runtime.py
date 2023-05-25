from xdsl.dialects import snitch_runtime


def test_snrt_op():
    cluster_num = snitch_runtime.ClusterNumOp()
    cluster_hw_barrier = snitch_runtime.ClusterHwBarrierOp()
    print(cluster_num)
    print(cluster_hw_barrier)
