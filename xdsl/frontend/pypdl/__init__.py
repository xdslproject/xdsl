from dataclasses import dataclass

from xdsl.frontend.pyast.context import PyASTContext
from xdsl.frontend.pypdl.transforms.func_to_pdl_rewrite import FuncToPdlRewrite
from xdsl.transforms.desymref import FrontendDesymrefyPass


@dataclass
class PyPDLContext(PyASTContext):
    """Encapsulate the mapping between Python and IR types and operations."""

    def __init__(self):
        super().__init__(
            post_transforms=[FrontendDesymrefyPass(), FuncToPdlRewrite()],
            post_callback=None,
        )
