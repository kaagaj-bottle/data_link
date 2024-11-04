
from setuptools import find_packages, setup

setup(
    name="content_insights",
    packages=find_packages(),
    install_requires=["dedupe",""],
    extrax_require={"develop": ["pytest"]},
)
