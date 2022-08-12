
import xdsl.fronted


def get_fqcn(cn: object) -> str:
    """
    Get fully qualified class name
    """
    fqcn = ""
    if hasattr(cn, '__package__'):
        fqcn += cn.__package__
    if hasattr(cn, '__module__'):
        fqcn += cn.__module__
    if hasattr(cn, '__qualname__'):
        fqcn += f".{cn.__qualname__}"  # type: ignore
    elif hasattr(cn, '_name'):
        fqcn += f".{cn._name}"  # type: ignore

    return fqcn


def is_frontend_obj(obj: object) -> object:
    """
    Checks whether a Python object is a frontend object.
    :returns: frontend object or None.
    """
    # TODO: find a better way to detect frontend objects (this relies on sensible naming)
    if obj \
            and hasattr(obj, "__module__") \
            and (obj.__module__.startswith(get_fqcn(xdsl.fronted)) or ".frontend" in obj.__module__):
        return obj
    return None


def frontend_module_name_to_xdsl_name(frontend_module_name: str):
    xdsl.fronted_name = get_fqcn(xdsl.fronted)
    if frontend_module_name.startswith(xdsl.fronted_name):
        return frontend_module_name.replace(xdsl.fronted_name, "xdsl")
    # TODO: find a better way to detect frontend objects (this relies on sensible naming)
    return frontend_module_name.replace(".frontend", "")
