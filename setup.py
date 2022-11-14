from setuptools import setup, find_packages
from version import get_git_version

with open("README.md", "r", encoding = "utf-8") as fh:
    long_description = fh.read()

with open('requirements.txt') as f:
    required = f.read().splitlines()

setup(
    name = "xdsl",
    version = get_git_version(),
    author = "Mathieu Fehr",
    author_email = "mathieu.fehr@gmail.com",
    description = "xDSL",
    long_description = long_description,
    long_description_content_type = "text/markdown",
    url = "https://xdsl.dev/",
    project_urls={
        'Source Code': 'https://github.com/xdslproject/xdsl',
        'Issue Tracker': 'https://github.com/xdslproject/xdsl/issues',
    },
    classifiers = [
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    platforms=["Linux", "Mac OS-X", "Unix"],
    license='MIT',
    install_requires=required,
    packages = find_packages(),
    python_requires = ">=3.10"
)
