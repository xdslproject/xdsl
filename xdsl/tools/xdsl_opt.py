import warnings

from xdsl.xdsl_opt_main import xDSLOptMain


def main():
    # Python ignores DeprecationWarnings and some others by default:
    # https://docs.python.org/3/library/warnings.html#default-warning-filter
    warnings.filterwarnings("default", category=DeprecationWarning)
    xDSLOptMain().run()


if "__main__" == __name__:
    main()
