f = open("requirements-optional.txt")

line = f.readlines()[-1]

version = line.split("==")[1].strip()

version_file = open(".github/pyright-version.txt", "w")
version_file.write(version)
