import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()


setuptools.setup(
    name="my-project-ieBoytsov",
    version="0.0.1",
    author="Ilya Boytsov",
    author_email="ilyaboytsov1805@gmail.com",
    description="A new project",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/ieBoytsov/my_project",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: Apache Software License",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.7",
)
