#!/usr/bin/python3

# This script updates the meta.yaml file used by Pyodide to bundle and use xDSL
# Takes the .yaml file and the xDSL directory as arguments

import hashlib
import os
import sys

import yaml

meta_yaml_path = sys.argv[1]
xdsl_directory = sys.argv[2]

# Parse the auto-generated one
with open(meta_yaml_path) as f:
    yaml_doc = yaml.safe_load(f)

# Find the built source distribution. This assumes it is the only thing in xdsl/dist
xdsl_sdist = os.listdir(os.path.join(xdsl_directory, "dist"))[0]
xdsl_sdist = os.path.abspath(os.path.join(xdsl_directory, "dist", xdsl_sdist))
sha256_hash = hashlib.sha256()
with open(xdsl_sdist, "rb") as sdist:
    # Read and update hash string value in blocks of 4K
    for byte_block in iter(lambda: sdist.read(4096), b""):
        sha256_hash.update(byte_block)

# Make it build the local xDSL, not the PyPi release. The pyodide build still requires the SHA256 sum.
yaml_doc["source"] = {"url": f"file://{xdsl_sdist}", "sha256": sha256_hash.hexdigest()}
yaml_doc["requirements"] = {"run": ["typing-extensions"]}
with open(meta_yaml_path, "w") as f:
    yaml.dump(yaml_doc, f)
