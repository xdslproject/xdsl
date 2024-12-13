import os
from collections.abc import Mapping
from typing import cast

from setuptools import Command, find_packages, setup

import versioneer

if "XDSL_VERSION_OVERRIDE" in os.environ:
    version = os.environ["XDSL_VERSION_OVERRIDE"]
else:
    version = versioneer.get_version()


setup(
    version=version,
    cmdclass=cast(Mapping[str, type[Command]], versioneer.get_cmdclass()),
    packages=find_packages(),
)
