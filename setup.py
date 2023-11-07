from setuptools import setup, find_packages
import codecs
import os

here = os.path.abspath(os.path.dirname(__file__))

with codecs.open(os.path.join(here, "README.md"), encoding="utf-8") as fh:
    long_description = "\n" + fh.read()

VERSION = '0.0.1'
DESCRIPTION = 'Unconstrained Optimisation Algorithms and Matrix Operations'
LONG_DESCRIPTION = 'A Python package integrating around ten unconstrained optimization algorithms, inclusive of 2D/3D visualizations for comparative analysis, and incorporated matrix operations.'

# Setting up
setup(
    name="PyOptim",
    version=VERSION,
    author="Heyyassinesedjari",
    author_email="<yassinesmsedjari@gmail.com>",
    description=DESCRIPTION,
    long_description_content_type="text/markdown",
    long_description=LONG_DESCRIPTION,
    packages=find_packages(),
    install_requires=["matplotlib", "numdifftools", "scipy", "sympy", "numpy"],
    keywords=['python', 'optimization', "matrix operations", "2D", "3D"],
    classifiers=[
        "Development Status :: 1 - Planning",
        "Intended Audience :: Developers",
        "Programming Language :: Python :: 3",
        "Operating System :: Unix",
        "Operating System :: MacOS :: MacOS X",
        "Operating System :: Microsoft :: Windows",
    ]
)
