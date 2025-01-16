from collections.abc import Mapping
from typing import cast

from setuptools import Command, find_packages, setup

import versioneer

version = versioneer.get_version()


setup(
    version=version,
    cmdclass=cast(Mapping[str, type[Command]], versioneer.get_cmdclass()),
    packages=find_packages(),
)
