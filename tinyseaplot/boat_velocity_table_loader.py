import csv
import typing
import itertools
from pathlib import Path

import numpy as np

from pytinysea import (
    velocity_t,
    radian_t,
    BoatVelocityTableBuilder,
    BoatVelocityTable,
    VelocityLinearSpace,
)

KNOT_TO_MS = 0.51444


def extract_line(
    line: typing.List[str],
) -> typing.Tuple[radian_t, typing.List[velocity_t]]:
    relative_wind = radian_t(np.deg2rad(float(line[0])))
    # Add 0 velocity for 0 wind and 0 velocity for max wind
    velocity_list = itertools.chain(
        [0.0], map(lambda x: KNOT_TO_MS * float(x), line[1:]), [0.0]
    )
    velocity_t_list = list(map(velocity_t, velocity_list))
    return relative_wind, velocity_t_list


def build_boat_velocity_table_from_csv(csv_path: Path) -> BoatVelocityTable:

    with csv_path.open("r") as table_fp:
        content = list(csv.reader(table_fp, delimiter=";"))

    wind_stop = float(content[0][-1]) * KNOT_TO_MS
    nr_wind = len(content[0][1:]) + 2  # add 0 wind and max wind

    wind_space = VelocityLinearSpace.from_bound(
        velocity_t(0.0), velocity_t(wind_stop), nr_wind
    )
    builder = BoatVelocityTableBuilder(wind_space)
    # jump front wind and rear wind
    for line in content[2:-1]:
        builder.add_symetric(*extract_line(line))

    builder.add(*extract_line(content[-1]))

    return builder.build()
