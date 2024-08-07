#!/usr/bin/env python
from setuptools import setup

setup(name='bronte',
      description='Bronte - the single eye MCAO system',
      version='0.1',
      classifiers=['Development Status :: 4 - Beta',
                   'Operating System :: POSIX :: Linux',
                   'Programming Language :: Python :: 3',
                   ],
      long_description=open('README.md').read(),
      url='',
      author_email='lorenzo.busoni@inaf.it',
      author='Arcetri Adaptive Optics team',
      license='',
      keywords='adaptive optics',
      packages=['bronte',
                ],
      install_requires=["numpy",
                        "arte",
                        "plico_dm",
                        "pysilico",
                        ],
      package_data={
          'bronte': ['data/*'],
      },
      include_package_data=True,
      test_suite='test',
      )
