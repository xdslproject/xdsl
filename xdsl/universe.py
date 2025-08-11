"""
xDSL leverages Python [plugins](https://packaging.python.org/en/latest/guides/creating-and-discovering-plugins/) for user-provided compiler infrastructure.

Python packages that want to extend the passes and dialects available in `xdsl-opt` can
implement their own `Universe`, which will be discovered by `Multiverse` at runtime.

To opt into this behavior first add a Universe instance somewhere in your project,
for example in `your_project/universe.py`:

```py
def get_dialect():
    ...

def get_pass():
    ...

YOUR_UNIVERSE = Universe(
    all_dialects={"your_dialect": get_dialect},
    all_passes={"your-pass": get_pass},
)
```

Then, add the following entry in your `pyproject.toml` file:

```toml
[project.entry-points.'xdsl.universe']
your_project = 'your_project.universe:YOUR_UNIVERSE'
```

Note that any clashes in names of dialects or passes will result in an error.
"""  # noqa: E501

from __future__ import annotations

from collections.abc import Callable

from xdsl.dialects import get_all_dialects
from xdsl.ir import Dialect
from xdsl.passes import ModulePass
from xdsl.transforms import get_all_passes


class Universe:
    """
    Contains all the dialects and passes from modules in the environment.
    """

    all_dialects: dict[str, Callable[[], Dialect]]
    """Dialects from all modules in the current environment."""
    all_passes: dict[str, Callable[[], type[ModulePass]]]
    """Passes from all modules in the current environment."""

    def __init__(
        self,
        *,
        all_dialects: dict[str, Callable[[], Dialect]] | None = None,
        all_passes: dict[str, Callable[[], type[ModulePass]]] | None = None,
    ) -> None:
        self.all_dialects = {} if all_dialects is None else all_dialects
        self.all_passes = {} if all_passes is None else all_passes

    @staticmethod
    def get_multiverse() -> Universe:
        """
        Traverses the current environment looking for entry points in the
        `xdsl.universe` group.
        """
        from importlib.metadata import entry_points

        discovered_plugins = entry_points(group="xdsl.universe")
        all_universes = {plugin.name: plugin.load() for plugin in discovered_plugins}
        sorted_universes = sorted(all_universes.items(), key=lambda t: t[0])

        all_dialects: dict[str, Callable[[], Dialect]] = {}
        all_passes: dict[str, Callable[[], type[ModulePass]]] = {}

        for universe_name, universe in sorted_universes:
            if not isinstance(universe, Universe):
                raise ValueError(
                    f"Entry point {universe} for plugin {universe_name} is not an "
                    "instance of Universe."
                )
            for dialect_name, dialect in universe.all_dialects.items():
                if dialect_name in all_dialects:
                    universes = [
                        n for n, u in sorted_universes if dialect_name in u.all_dialects
                    ]
                    raise ValueError(
                        f"Duplicate definition of {dialect_name} in {universes}."
                    )
                all_dialects[dialect_name] = dialect

            for pass_name, _pass in universe.all_passes.items():
                if pass_name in all_passes:
                    universes = [
                        n for n, u in sorted_universes if pass_name in u.all_passes
                    ]
                    raise ValueError(
                        f"Duplicate definition of {pass_name} in {universes}."
                    )
                all_passes[pass_name] = _pass

        return Universe(all_dialects=all_dialects, all_passes=all_passes)


XDSL_UNIVERSE = Universe(all_dialects=get_all_dialects(), all_passes=get_all_passes())
