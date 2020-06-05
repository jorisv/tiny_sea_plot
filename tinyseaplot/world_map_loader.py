import typing
from pathlib import Path

import numpy as np

import xarray as xr

from shapely import affinity
from shapely.geometry import LineString

from bokeh.models import ColumnDataSource

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

from tinyseaplot.transform import lat_to_web_mercator, lon_to_web_mercator


def create_map_space(
    ds: xr.core.dataset.Dataset,
) -> typing.Tuple[TimeLinearSpace, LatitudeLinearSpace, LongitudeLinearSpace]:
    """Extract time, latitude and longitude space from the grib file"""

    u_wind_table = ds.u10

    nr_lat = u_wind_table.GRIB_Ny
    lat_0 = np.deg2rad(u_wind_table.GRIB_latitudeOfFirstGridPointInDegrees)
    lat_step = np.deg2rad(u_wind_table.GRIB_jDirectionIncrementInDegrees)
    nr_lon = u_wind_table.GRIB_Nx
    lon_0 = np.deg2rad(u_wind_table.GRIB_longitudeOfFirstGridPointInDegrees)
    lon_step = np.deg2rad(u_wind_table.GRIB_iDirectionIncrementInDegrees)

    nr_time = len(ds.step)
    time_0 = float(ds.step[0] + ds.time) / 1e9
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


def build_time_world_map_from_grib(grib_file: Path) -> TimeWorldMap:
    ds = xr.open_dataset(str(grib_file), engine="cfgrib")

    time_linear_space, lat_linear_space, lon_linear_space = create_map_space(ds)
    time_world_map = build_time_world_map(
        ds.u10, ds.v10, time_linear_space, lat_linear_space, lon_linear_space
    )
    return time_world_map


def world_map_to_data_source(
    world_map: WorldMapGrid, discret_step: float
) -> typing.Tuple[ColumnDataSource, ColumnDataSource]:
    """Turn WorldMapGrid into a wind intensity heatmap and wind direction"""

    patches = []

    discret_step_rad = np.deg2rad(discret_step)

    lat_space = world_map.x_space()
    lon_space = world_map.y_space()

    lat_range = np.concatenate(
        (
            np.arange(lat_space.start().t, lat_space.stop().t, discret_step_rad),
            [lat_space.stop().t],
        )
    )
    lon_range = np.concatenate(
        (
            np.arange(lon_space.start().t, lon_space.stop().t, discret_step_rad),
            [lon_space.stop().t],
        )
    )

    img = np.empty((len(lon_range), len(lat_range)))
    wind_poly = LineString(
        [
            [-discret_step_rad * 0.1, -discret_step_rad * 0.02,],
            [discret_step_rad * 0.1, 0.0],
            [-discret_step_rad * 0.1, discret_step_rad * 0.02,],
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

    xs = [
        list(map(lambda x: lon_to_web_mercator(np.rad2deg(x[0])), p)) for p in patches
    ]
    ys = [
        list(map(lambda x: lat_to_web_mercator(np.rad2deg(x[1])), p)) for p in patches
    ]

    return (
        ColumnDataSource(data={"wind_intensity": [np.fliplr(np.flipud(img))],}),
        ColumnDataSource(data={"wind_xs": xs, "wind_ys": ys,}),
    )
