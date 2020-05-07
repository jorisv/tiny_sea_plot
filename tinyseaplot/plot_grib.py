import typing
import argparse
from pathlib import Path

import numpy as np

import xarray as xr

from shapely import affinity
from shapely.geometry import LineString

from bokeh.models import ColumnDataSource, Slider, Button
from bokeh.plotting import figure
from bokeh.layouts import column, row
from bokeh.tile_providers import CARTODBPOSITRON, get_provider

# import pytinysea
from pytinysea import (
    latitude_t,
    longitude_t,
    time_t,
    velocity_t,
    radian_t,
    LatitudeLinearSpace,
    LongitudeLinearSpace,
    TimeLinearSpace,
    WorldMap,
    WorldMapGrid,
    WorldMapData,
    WorldMapGridBuilder,
    TimeWorldMap,
    TimeWorldMapBuilder,
)


def check_path(str_path: str) -> Path:
    path = Path(str_path)
    if not path.exists():
        raise argparse.ArgumentTypeError(f"{str_path} doesn't exist")
    return path


def create_map_space(
    ds: xr.core.dataset.Dataset,
) -> typing.Tuple[TimeLinearSpace, LatitudeLinearSpace, LongitudeLinearSpace]:
    """Extract time, latitude and longitude space from the grib file"""

    u_wind_table = ds.u10

    nr_lat = u_wind_table.GRIB_Ny
    lat_0 = u_wind_table.GRIB_latitudeOfFirstGridPointInDegrees
    lat_step = u_wind_table.GRIB_jDirectionIncrementInDegrees
    nr_lon = u_wind_table.GRIB_Nx
    lon_0 = u_wind_table.GRIB_longitudeOfFirstGridPointInDegrees
    lon_step = u_wind_table.GRIB_iDirectionIncrementInDegrees

    nr_time = len(ds.step)
    time_0 = float(ds.step[0]) / 1e9
    time_step = float(ds.step[1] - ds.step[0]) / 1e9

    lat_linear_space = LatitudeLinearSpace(
        latitude_t(lat_0), latitude_t(lat_step), nr_lat
    )
    lon_linear_space = LongitudeLinearSpace(
        longitude_t(lon_0), longitude_t(lon_step), nr_lon
    )

    time_linear_space = TimeLinearSpace(time_t(time_0), time_t(time_step), nr_time)

    return time_linear_space, lat_linear_space, lon_linear_space


def build_layer(
    u_layer: xr.core.dataarray.DataArray,
    v_layer: xr.core.dataarray.DataArray,
    lat_space: LatitudeLinearSpace,
    lon_space: LongitudeLinearSpace,
    world_map_grid_builder: WorldMapGridBuilder,
) -> WorldMap:
    """Build one world map from a specific grib layer"""

    # convert uv wind to wind bearing (0 degree north, clockwise)
    wind_bearing = ((np.pi / 2.0) - np.arctan2(v_layer, u_layer)) + np.pi
    wind_velocity = np.sqrt(v_layer ** 2 + u_layer ** 2)

    for i in range(lat_space.nr_points()):
        for j in range(lon_space.nr_points()):
            world_map_grid_builder[i, j] = WorldMapData(
                radian_t(wind_bearing[i, j]), velocity_t(wind_velocity[i, j])
            )
    return WorldMap(world_map_grid_builder.build())


def build_time_world_map(
    u: xr.core.dataset.Dataset,
    v: xr.core.dataset.Dataset,
    time_space: TimeLinearSpace,
    lat_space: LatitudeLinearSpace,
    lon_space: LongitudeLinearSpace,
) -> TimeWorldMap:
    """Build all world map from all grib layer"""

    world_map_grid_builder = WorldMapGridBuilder(lat_space, lon_space)
    time_world_map_builder = TimeWorldMapBuilder(time_space)
    for i in range(time_space.nr_points()):
        time_world_map_builder.add(
            build_layer(u[i], v[i], lat_space, lon_space, world_map_grid_builder,)
        )

    return time_world_map_builder.build()


def lat_to_web_mercator(lat):
    """Convert latitude to web mercator"""
    k = 6378137
    return np.log(np.tan((90 + lat) * np.pi / 360.0)) * k


def lon_to_web_mercator(lon):
    """Convert longitude to web mercator"""
    k = 6378137
    return lon * (k * np.pi / 180.0)


def world_map_to_data_source(
    world_map: WorldMapGrid, discret_step: float
) -> typing.Tuple[ColumnDataSource, ColumnDataSource]:
    """Turn WorldMapGrid into a wind intensity heatmap and wind direction"""

    patches = []

    lat_space = world_map.x_space()
    lon_space = world_map.y_space()

    lat_range = np.concatenate(
        (
            np.arange(lat_space.start().t, lat_space.stop().t, discret_step),
            [lat_space.stop().t],
        )
    )
    lon_range = np.concatenate(
        (
            np.arange(lon_space.start().t, lon_space.stop().t, discret_step),
            [lon_space.stop().t],
        )
    )

    img = np.empty((len(lon_range), len(lat_range)))
    wind_poly = LineString(
        [
            [-discret_step * 0.1, -discret_step * 0.02,],
            [discret_step * 0.1, 0.0],
            [-discret_step * 0.1, discret_step * 0.02,],
        ]
    )

    for j, lat in enumerate(lat_range):
        for i, lon in enumerate(lon_range):
            data: WorldMapData = world_map.interpolated(
                latitude_t(lat), longitude_t(lon)
            )
            angle = -(data.wind_bearing.t + (np.pi / 2.0))
            poly = affinity.rotate(
                affinity.translate(wind_poly, lon, lat), angle, use_radians=True
            )

            patches.append(poly.coords)
            img[i, j] = data.wind_velocity.t

    xs = [list(map(lambda x: lon_to_web_mercator(x[0]), p)) for p in patches]
    ys = [list(map(lambda x: lat_to_web_mercator(x[1]), p)) for p in patches]

    return (
        ColumnDataSource(data={"wind_intensity": [np.fliplr(np.flipud(img))],}),
        ColumnDataSource(data={"wind_xs": xs, "wind_ys": ys,}),
    )


class BokehApp(object):
    def __init__(self, time_world_map: TimeWorldMap):
        self.time_world_map = time_world_map

    def setup(self, doc):
        world_map_t0: WorldMapGrid = self.time_world_map[time_t(0)].world_grid()
        lat_space = world_map_t0.x_space()
        lon_space = world_map_t0.y_space()

        tile_provider = get_provider(CARTODBPOSITRON)

        p = figure(
            x_range=(
                lon_to_web_mercator(lon_space.start().t),
                lon_to_web_mercator(lon_space.stop().t),
            ),
            y_range=(
                lat_to_web_mercator(lat_space.start().t),
                lat_to_web_mercator(lat_space.stop().t),
            ),
            x_axis_type="mercator",
            y_axis_type="mercator",
            width=1280,
        )
        p.add_tile(tile_provider)

        img_data_source_t0, wind_data_source_t0 = world_map_to_data_source(
            world_map_t0, 0.1
        )

        p.image(
            image="wind_intensity",
            x=p.x_range.start,
            y=p.y_range.start,
            dw=(p.x_range.end - p.x_range.start),
            dh=(p.y_range.end - p.y_range.start),
            global_alpha=0.6,
            palette="Turbo256",
            source=img_data_source_t0,
        )
        p.patches(
            xs="wind_xs", ys="wind_ys", source=wind_data_source_t0,
        )

        self.start_end_source = ColumnDataSource(
            data={
                "x": [p.x_range.start, p.x_range.end],
                "y": [p.y_range.start, p.y_range.end],
            }
        )
        p.cross(
            x="x",
            y="y",
            color="red",
            line_width=2,
            size=lon_to_web_mercator(0.0002),
            source=self.start_end_source,
        )

        start_x_slider = Slider(
            start=lon_space.start().t,
            end=lon_space.stop().t,
            value=lon_space.start().t,
            step=0.001,
            title="Start X",
        )
        start_x_slider.on_change(
            "value",
            lambda attr, new, old: self.start_end_update(
                lon_to_web_mercator, ("x", 0), attr, new, old
            ),
        )

        start_y_slider = Slider(
            start=lat_space.start().t,
            end=lat_space.stop().t,
            value=lat_space.start().t,
            step=0.001,
            title="Start Y",
        )
        start_y_slider.on_change(
            "value",
            lambda attr, new, old: self.start_end_update(
                lat_to_web_mercator, ("y", 0), attr, new, old
            ),
        )

        end_x_slider = Slider(
            start=lon_space.start().t,
            end=lon_space.stop().t,
            value=lon_space.stop().t,
            step=0.001,
            title="End X",
        )
        end_x_slider.on_change(
            "value",
            lambda attr, new, old: self.start_end_update(
                lon_to_web_mercator, ("x", 1), attr, new, old
            ),
        )

        end_y_slider = Slider(
            start=lat_space.start().t,
            end=lat_space.stop().t,
            value=lat_space.stop().t,
            step=0.001,
            title="End Y",
        )
        end_y_slider.on_change(
            "value",
            lambda attr, new, old: self.start_end_update(
                lat_to_web_mercator, ("y", 1), attr, new, old
            ),
        )

        compute_button = Button(label="Compute")
        compute_button.on_click(self.compute)

        doc.add_root(
            row(
                [
                    p,
                    column(
                        [
                            start_x_slider,
                            start_y_slider,
                            end_x_slider,
                            end_y_slider,
                            compute_button,
                        ]
                    ),
                ]
            )
        )

    def compute(self, new):
        print(new)

    def start_end_update(self, tr, pos, attr, new, old):
        self.start_end_source.patch({pos[0]: [(pos[1], tr(new))]})


def main_func():
    parser = argparse.ArgumentParser(description="Plot grib wind data")
    parser.add_argument(
        "grib_file", type=check_path, help="Grib file path",
    )
    args = parser.parse_args()

    ds = xr.open_dataset(str(args.grib_file), engine="cfgrib")

    time_linear_space, lat_linear_space, lon_linear_space = create_map_space(ds)
    time_world_map = build_time_world_map(
        ds.u10, ds.v10, time_linear_space, lat_linear_space, lon_linear_space
    )

    bokeh_app = BokehApp(time_world_map)

    from bokeh.server.server import Server

    server = Server({"/": bokeh_app.setup}, num_procs=1)
    server.start()
    server.io_loop.add_callback(server.show, "/")
    server.io_loop.start()
    server.io_loop.start()
