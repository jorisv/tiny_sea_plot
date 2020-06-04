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
    url="https://github.com/jorisv/tiny_sea_plot",
    packages=["tinyseaplot"],
    entry_points={"console_scripts": ["tiny_sea_plot=tinyseaplot.__main__:main",]},
    install_requires=[
        "pytinysea>=0.2.0",
        "bokeh>=2.0.0",
        "cfgrib",
        "xarray>=0.12.0",
        "shapely",
    ],
    python_requires=">=3.6",
    zip_safe=False,
    include_package_data=True,
)
