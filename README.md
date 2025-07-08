# hod_madrid_py

<a target="_blank" href="https://cookiecutter-data-science.drivendata.org/">
    <img src="https://img.shields.io/badge/CCDS-Project%20template-328F97?logo=cookiecutter" />
</a>

Code to construct mock catalogues bwith an Halo Occupation Distribution (HOD) model. The code is based on [Avila+2020](https://arxiv.org/abs/2007.09012). Please cite this paper if you use the code.

## Installing and running the code

1. Download the code. In the command line you can clone the repository:
   git clone git@github.com:computationalAstroUAM/hod_madrid_py.git
2. Get to the repository. In the command line:
   '''cd [PATH TO REPOSITORY]/hod_madrid_py/'''
3. Get the adequate libraries. For this, you might want to create a conda environment from the repository's *environment.yml* (this is not neccessary if you already have the needed libraries):
   '''conda env create -f environment.yml'''
   1. Activate the environment:
     '''conda activate hod_madrid_py'''
   2. To deactivate the environment when you are done: 
     '''conda deactivate'''
4. Run the code using the provided example:
   '''python produce_hod_mock.py'''

### Test run

This repository comes with an example file, data/example/UNIT_haloes_logMmin13.500_logMmax14.500.txt. The file contains [UNIT](https://arxiv.org/abs/1811.02111) dark matter haloes, identified with [ROCKSTAR](https://arxiv.org/abs/1110.4372), with masses $13.5<log_{10}M/M_{odot}<14.5$ at $z=0.86$ from [Vos Gines+2024](https://arxiv.org/abs/2310.18189).

Running the example code, python produce_hod_mock.py, will generate an HOD mock catalogue in the output folder (this is set not to be part of the repository).

## Project Organization

```
├── produce_hod_mock_updated.py <- Example to run the code
├── README.md          <- The top-level README for developers using this project.
├── data
│   ├── example        <- File with UNIT haloes to make a test run
│
├── docs               <- A default mkdocs project; see www.mkdocs.org for details
│
├── pyproject.toml     <- Project configuration file with package metadata for 
│                         hod_madrid_py and configuration for tools like black
│
├── environment.yml    <- The requirements file for reproducing the analysis environment
│
│
└── src                <- Source code for use in this project.
    │
    ├── __init__.py    <- Makes src a Python module
    │
    ├── hod_model.py   <- Main code for the HOD modelling
    │    │
    ├── hod_io.py      <- Read input parameters and file properties
    │
    ├── hod_shape.py   <- Shape of the average HOD
    │
    ├── hod_pdf.py     <- Probability distributions
    │
    ├── hod_radial_profile.py  <- Radial profile for satellite galaxies 
    │
    ├── hod_v_profile.py       <- Velocity profile for satellite galaxies 
    │
    ├── hod_cosmology.py       <- Codes for cosmological parameters
    │
    └── hod_plots.py   <- Code to create visualizations
```

--------

