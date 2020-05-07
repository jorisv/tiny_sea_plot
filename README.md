# PyTinySea Plot

PyTinySea Bokeh interface to compute sailing boat routing

<!-- vim-markdown-toc GitLab -->

* [Installation](#installation)
	* [Python](#python)
* [Run](#run)

<!-- vim-markdown-toc -->

## Installation

Before doing any of theses steps you must have installed py\_tiny\_sea in your [Python environment](https://github.com/jorisv/py_tiny_sea#build-python).

PyTinySea Plot need CfgGrib that need [eccodes library](https://github.com/ecmwf/cfgrib#binary-dependencies) to run.

### Python

```bash
git clone https://github.com/jorisv/tiny_sea_plot.git
cd tiny_sea_plot
python -m pip install .
```

## Run

```bash
tiny_sea_plot grib_file polar_file
```
