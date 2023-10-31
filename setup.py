import re
from collections.abc import Mapping
from pathlib import Path
from typing import cast

from setuptools import Command, find_packages, setup

import versioneer

# Add README.md as long description
this_directory = Path(__file__).parent
long_description = (this_directory / "README.md").read_text()

git_regex = r"git\+(?P<url>https:\/\/github.com\/[\w]+\/[\w]+\.git)(@(?P<version>[\w]+))?(#egg=(?P<name>[\w]+))?"

with open("requirements.txt") as f:
    required = f.read().splitlines()

with open("requirements-optional.txt") as f:
    optionals = f.read().splitlines()

reqs: list[str] = []
for ir in required:
    if ir[0:3] == "git":
        name = ir.split("/")[-1]
        reqs += [f"{name} @ {ir}@main"]
    else:
        reqs += [ir]

extras_require = {}
for mreqs, mode in zip(
    [
        optionals,
    ],
    [
        "extras",
    ],
):
    opt_reqs: list[str] = []
    for ir in mreqs:
        # For conditionals like pytest=2.1; python == 3.6
        if ";" in ir:
            entries = ir.split(";")
            extras_require[entries[1]] = entries[0]
        elif ir[0:3] == "git":
            m = re.match(git_regex, ir)
            assert m is not None
            items = m.groupdict()
            name = items["name"]
            url = items["url"]
            version = items.get("version")
            if version is None:
                version = "main"

            opt_reqs += [f"{name} @ git+{url}@{version}"]
        else:
            opt_reqs += [ir]
    extras_require[mode] = opt_reqs

setup(
    version=versioneer.get_version(),
    cmdclass=cast(Mapping[str, type[Command]], versioneer.get_cmdclass()),
    packages=find_packages(),
    install_requires=reqs,
    extras_require=extras_require,
)
