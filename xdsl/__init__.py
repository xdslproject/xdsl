class LazyVersion:
    """
    Resolving the version dynamically is slow, hence this lazy wrapper to get the version
    only when needed.
    """

    def __init__(self):
        self._version = None

    def __str__(self):
        if self._version is None:
            import os

            if "XDSL_VERSION_OVERRIDE" in os.environ:
                self._version = os.environ["XDSL_VERSION_OVERRIDE"]
            else:
                from . import _version

                self._version = _version.get_versions()["version"]
        return self._version


__version__ = LazyVersion()
