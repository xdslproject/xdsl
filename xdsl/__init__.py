class LazyVersion:
    """
    Resolving the version dynamically is slow, hence this lazy wrapper to get the version
    only when needed.
    """

    _version: str | None

    def __init__(self):
        self._version = None

    def __str__(self):
        if self._version is None:
            from importlib.metadata import PackageNotFoundError, version

            try:
                self._version = version("xdsl")
            except PackageNotFoundError:
                # package is not installed
                self._version = "Error: xDSL not installed"

        return self._version


__version__ = LazyVersion()
