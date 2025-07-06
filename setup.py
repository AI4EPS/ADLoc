from setuptools import setup

setup(
    name="ADLoc",
    version="1.0.3",
    long_description="ADLoc",
    long_description_content_type="text/markdown",
    packages=["adloc"],
    install_requires=["numpy", "h5py", "matplotlib", "pandas", "tqdm", "pyproj", "numba", "scipy"],
)
