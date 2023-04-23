import versioneer
from setuptools import find_packages, setup
from pathlib import Path

# Add README.md as long description
this_directory = Path(__file__).parent
long_description = (this_directory / "README.md").read_text()

with open("requirements.txt") as f:
    required = f.read().splitlines()

with open("requirements-optional.txt") as f:
    optionals = f.read().splitlines()

reqs = []
for ir in required:
    if ir[0:3] == "git":
        name = ir.split("/")[-1]
        reqs += ["%s @ %s@main" % (name, ir)]
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
    opt_reqs = []
    for ir in mreqs:
        # For conditionals like pytest=2.1; python == 3.6
        if ";" in ir:
            entries = ir.split(";")
            extras_require[entries[1]] = entries[0]
        # Git repos, install master
        if ir[0:3] == "git":
            name = ir.split("/")[-1]
            opt_reqs += ["%s @ %s@main" % (name, ir)]
        else:
            opt_reqs += [ir]
    extras_require[mode] = opt_reqs

setup(
    name="xdsl",
    version=versioneer.get_version(),
    cmdclass=versioneer.get_cmdclass(),
    description="xDSL",
    long_description=long_description,
    long_description_content_type="text/markdown",
    entry_points={"console_scripts": ["xdsl-opt = xdsl.tools.xdsl_opt:main"]},
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
