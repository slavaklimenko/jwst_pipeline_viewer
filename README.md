# jwst_Viewer

This code provides a graphical interface for visualizing and analyzing MIRI MRS data processed with the official JWST pipeline stages 1, 2, and 3.

## Installation and Usage

### 1. Download the code

Either download the repository directly or clone it using:

```bash
git clone https://github.com/slavaklimenko/jwst_Viewer
```

### 2. Create a Python environment and install dependencies

Create a new Conda environment:

```bash
conda create -n jwst_viewer python=3.13.13
conda activate jwst_viewer
```
Install the required JWST pipeline version manually:

```bash
conda install jwst=2.0.0
```

Install the required packages using the provided environment file:

```bash
conda env update -n jwst_viewer -f environment.yml
```


### 3. Configure the input data path

Edit the paths to your uncalibrated JWST data and CRDS data directory in the `init.dat` file.

### 4. Run the program

Start the viewer by running:

```bash
python main.py
```

This launches the detector image viewer for inspecting JWST pipeline Stage 1 and Stage 2 products, as well as the MRS cube analysis tools for creating and analyzing Stage 3 spectral cubes.
