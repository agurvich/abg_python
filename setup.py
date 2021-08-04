import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="abg_python",
    version="1.0.0",
    author = 'Alex Gurvich',
    author_email = 'agurvich@u.northwestern.edu',
    description="shared python utilities for http://github.com/agurvich 's packages.",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/agurvich/abg_python",
    project_urls={
        "Bug Tracker": "https://github.com/agurvich/abg_python/issues",
    },
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    package_dir={"": "src"},
    packages=setuptools.find_packages(where="src"),
    python_requires=">=3.6",
    install_requires=[            
          'scipy',
          'numpy',
          'h5py',
          'pandas',
          'matplotlib'
      ],
    include_package_data=True,
)

