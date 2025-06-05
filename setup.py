from setuptools import setup

with open("README.md", "r") as fh:
    long_description = fh.read()

setup(
    name="GaitAnalyzer",
    version=1.0,
    author="laboratoireIRISSE",
    author_email="eve.charbonneau@univ-reunion.fr",
    description="A toolbox for biomechanical analysis of gait",
    long_description=long_description,
    url="https://github.com/laboratoireIRISSE/GaitAnalyzer",
    packages=[
        ".",
        "gait_analyzer",
        "examples",
    ],
    license="LICENSE",
    keywords=[
        "Walking",
        "Biomechanics",
        "Gait parameters",
        "Functional Electrical Stimulation",
        "Motion capture",
        "Ground reaction force",
    ],
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
    ],
    include_package_data=True,
    python_requires=">=3.7",
    zip_safe=False,
)
