from setuptools import setup

with open("README.md", "r") as fh:
    long_description = fh.read()

setup(
    name="tinyseaplot",
    version="0.1.0",
    author="Joris Vaillant",
    author_email="joris.vaillant@gmail.com",
    description="TinySea plot tools",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="nullspace.fr:/home/trollboy/tiny_sea_plot",
    packages=["tinyseaplot"],
    entry_points={"console_scripts": ["plot_grib=tinyseaplot.plot_grib:main_func",]},
    install_requires=[
        "pytinysea",
        "bokeh>=2.0.0",
        "cfgrib",
        "xarray>=0.12.0",
        "shapely",
    ],
    python_requires=">=3.6",
)
