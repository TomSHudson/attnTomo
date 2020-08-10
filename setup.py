import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="attnTomo",
    version="0.0.1",
    author="Tom Hudson",
    author_email="tsh37@alumni.cam.ac.uk",
    description="An attenuation tomography package for microseismic studies",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/TomSHudson/attnTomo",
    packages=setuptools.find_packages(),
    classifiers=[
        "Development Status :: 4 - Beta",
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
)
