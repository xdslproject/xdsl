import versioneer

from setuptools import setup, find_packages

with open('requirements.txt') as f:
    required = f.read().splitlines()

with open('requirements-optional.txt') as f:
    optionals = f.read().splitlines()

reqs = []
for ir in required:
    if ir[0:3] == 'git':
        name = ir.split('/')[-1]
        reqs += ['%s @ %s@master' % (name, ir)]
    else:
        reqs += [ir]

extras_require = {}
for mreqs, mode in zip([optionals, ], ['extras', ]):
    opt_reqs = []
    for ir in mreqs:
        # For conditionals like pytest=2.1; python == 3.6
        if ';' in ir:
            entries = ir.split(';')
            extras_require[entries[1]] = entries[0]
        # Git repos, install master
        if ir[0:3] == 'git':
            name = ir.split('/')[-1]
            opt_reqs += ['%s @ %s@master' % (name, ir)]
        else:
            opt_reqs += [ir]
    extras_require[mode] = opt_reqs

setup(name='xdsl',
      version=versioneer.get_version(),
      cmdclass=versioneer.get_cmdclass(),
      description="xDSL.",
      long_description="""
      Add long description here.""",
      project_urls={
          'Documentation': 'https://www....html',
          'Source Code': 'https://github.com/xdslproject/xdsl',
          'Issue Tracker': 'https://github.com/xdslproject/xdsl/issues',
      },
      url='https://xdsl.dev/',
      platforms=["Linux", "Mac OS-X", "Unix"],
      test_suite='pytest',
      author="-",
      author_email='-',
      license='MIT',
      packages=find_packages(),
      install_requires=reqs,
      extras_require=extras_require)