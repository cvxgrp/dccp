from setuptools import setup

setup(
    name="dccp",
    version="1.0.4",
    author="Xinyue Shen, Steven Diamond, Stephen Boyd",
    author_email="xinyues@stanford.edu, diamond@cs.stanford.edu, boyd@stanford.edu",
    packages=["dccp"],
    license="GPLv3",
    zip_safe=False,
    install_requires=["cvxpy >= 0.3.5"],
    url="http://github.com/cvxgrp/dccp/",
    description="A CVXPY extension for difference of convex programs.",
)
