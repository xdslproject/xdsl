from xdsl.utils.exceptions import ParseError
from xdsl.xdsl_opt_main import xDSLOptMain


def main():
    try:
        xDSLOptMain().run()
    except ParseError as pe:
        print(pe.with_context())
        return 1
