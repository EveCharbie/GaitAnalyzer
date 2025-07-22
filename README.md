# GaitAnalyzer

This report is for gait analysis.


## Installation
If you need help setting up your environment, you can follow the instructions in the section setup.
* Please note that python 3.10 must be used because of OpenSim's API. 

For now the GaitAnalyzer library is only installable from source files. 
If you only want to use it, you can now clone the repository:
```bash
git clone https://github.com/laboratoireIRISSE/GaitAnalyzer.git
```
But if you would like to contribute, you can fork the repository and clone your fork instead.
To be able to use the GaitAnalyzer properly, you can install the following dependency packages from conda-forge
```bash
conda install -c conda-forge numpy matplotlib plotly biorbd=1.10.0 ezc3d scipy lxml pandas openpyxl gitpython pyorerun pyomeca opensim-org::opensim=4.5.1 rerun-sdk=0.21.0
```
You must also install the following libraries:
- BioBuddy (https://github.com/pyomeca/biobuddy) #TODO: add to conda-forge


If you are a developer, you can install the following libraries as well:
```bash
conda install -c conda-forge pytest black
```

## Setup
1. First, you will need conda. You can install it through miniconda3 (https://docs.anaconda.com/miniconda/).
To confirm that conda is installed, you can run the following command:
```bash
conda --version
```
Then, you can create a new environment with the following command:
```bash
conda create --name [name of your environment] python=3.10
```
The python version should be 3.11 because of opensim that will be installed after.

Then, you can activate you environment using the following command:
```bash
conda activate [name of your environment]
```
You can then install the packages listed in the installation section.


2. Before stepping into the code, you will need a code editor. We recommend using PyCharm (https://www.jetbrains.com/help/pycharm/installation-guide.html).
In PyCharm, you will need to open the project folder GaitAnalyser. You can do so by clicking on File > Open and selecting the folder where you cloned the repository. 
Then, you will need to set up the Python interpreter. You can do so by clicking on PyCharm/File > Settings > Project: GaitAnalyser > Python Interpreter > Gearwheel (top right) > Add... > Conda environment > Existing environment > [name of your environment].
You are good to go! Have fun with your GaitAnalyser projects!

3. If you want to contribute, we also recommend downloading GitKraken (https://www.gitkraken.com/).