# Write the benchmarking functions here.
# See "Writing benchmarks" in the asv docs for more information.


class ImportXDSLOptSuite:
    """
    A benchmark that measures the time to import xdsl_opt_main. This is most of the
    constant cost of running xdsl-opt.
    """

    def time_import_xdsl_opt_main(self):
        from xdsl import xdsl_opt_main as _
