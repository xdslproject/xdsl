import re
from pathlib import Path
from typing import Mapping, cast

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
    name="xdsl",
    version=cast(str, versioneer.get_version()),
    cmdclass=cast(Mapping[str, type[Command]], versioneer.get_cmdclass()),
    description="xDSL",
    long_description=long_description,
    long_description_content_type="text/markdown",
    entry_points={
        "console_scripts": [
            "xdsl-opt = xdsl.tools.xdsl_opt:main",
            "irdl-to-pyrdl = xdsl.tools.irdl_to_pyrdl:main",
        ]
    },
    project_urls={
        "Source Code": "https://github.com/xdslproject/xdsl",
        "Issue Tracker": "https://github.com/xdslproject/xdsl/issues",
    },
    url="https://xdsl.dev/",
    platforms=["Linux", "Mac OS-X", "Unix"],
    test_suite="pytest",
    author="Mathieu Fehr",
    author_email="mathieu.fehr@ed.ac.uk",
    license="MIT",
    packages=find_packages(),
    package_data={"xdsl": ["py.typed"]},
    install_requires=reqs,
    extras_require=extras_require,
    zip_safe=False,
)
