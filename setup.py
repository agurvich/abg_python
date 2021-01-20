import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="abg_python",
    version="0.1",
    author="Alex Gurvich",
    author_email="agurvich@u.northwestern.edu",
    description="common python utilities",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/agurvich/abg_python",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    install_requires=[
        'pandas>=1.2.0',
        'scipy>=1.5.0',
        'numpy>=1.19.1',
        'palettable>=3.3.0',
        'matplotlib>=3.2.2',
        'numba>=0.51.2',
        'firefly_api>=0.0.2',
        'h5py>=3.1.0',
        'psutil>=5.8.0',
    ],
)
