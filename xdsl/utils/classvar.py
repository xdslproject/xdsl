from typing import Any, ClassVar, get_origin

from xdsl.utils.exceptions import PyRDLError


def is_classvar(annotation: Any) -> bool:
    """
    The type annotation can be one of
     * `ClassVar[MyType]`,
     * `ClassVar`, or
     * `"ClassVar[MyType]"`.
    """
    return (
        get_origin(annotation) is ClassVar
        or annotation is ClassVar
        or (isinstance(annotation, str) and annotation.startswith("ClassVar"))
    )


def is_const_classvar(
    field_name: str, annotation: Any, error_type: type[PyRDLError]
) -> bool:
    """
    Operation definitions may only have `*_def` fields or constant class variables,
    where the constness is defined by convention with an UPPER_CASE name and enforced by
    pyright.
    The type annotation can be one of
     * `ClassVar[MyType]`,
     * `ClassVar`, or
     * `"ClassVar[MyType]"`.
    This function throws an exception on a non UPPER_CASE class variable.
    """
    if not is_classvar(annotation):
        return False
    if not field_name.isupper():
        raise error_type(f'Invalid ClassVar name "{field_name}", must be uppercase.')
    return True
