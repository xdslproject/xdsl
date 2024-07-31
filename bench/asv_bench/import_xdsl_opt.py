# Write the benchmarking functions here.
# See "Writing benchmarking" in the asv docs for more information.


def timeraw_import_inspect():
    """
    A benchmark that measures the time to import xdsl_opt_main. This is most of the
    constant cost of running xdsl-opt.
    """
    return """
    from xdsl.xdsl_opt_main import xDSLOptMain
    """
