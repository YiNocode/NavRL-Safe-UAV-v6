"""Installation script for the ``navrl_uav_v5`` Isaac Lab extension."""

from pathlib import Path

import toml
from setuptools import find_packages, setup


EXTENSION_ROOT = Path(__file__).resolve().parent
EXTENSION_DATA = toml.load(EXTENSION_ROOT / "config" / "extension.toml")

setup(
    name="navrl_uav_v5",
    version=EXTENSION_DATA["package"]["version"],
    description=EXTENSION_DATA["package"]["description"],
    author=EXTENSION_DATA["package"]["author"],
    maintainer=EXTENSION_DATA["package"]["maintainer"],
    packages=find_packages(),
    include_package_data=True,
    python_requires=">=3.11",
    zip_safe=False,
)

