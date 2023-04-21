import os

f = open("requirements-optional.txt", "r")

line = f.readlines()[-1]

version = line.split("==")[1].strip()

version_file = open("pyright-version.txt", "w")
version_file.write(version)
