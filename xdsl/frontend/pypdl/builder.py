from dataclasses import dataclass, field

from xdsl.frontend.pyast.utils.builder import PyASTBuilder
from xdsl.frontend.pypdl.transforms.func_to_pdl_rewrite import FuncToPdlRewrite
from xdsl.passes import ModulePass
from xdsl.transforms.desymref import FrontendDesymrefyPass


@dataclass
class PyPDLRewriteBuilder(PyASTBuilder):
    """Builder for PDL rewrites from aspects of a Python function."""

    post_transforms: list[ModulePass] = field(
        default_factory=lambda: [FrontendDesymrefyPass(), FuncToPdlRewrite()]
    )
    """An ordered list of passes to apply to the built module."""

    post_verify: bool = False
    """Whether to verify each post processing transformation pass."""
