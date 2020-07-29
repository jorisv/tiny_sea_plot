import typing
from datetime import datetime, timezone

import numpy as np

import gpxpy

from pytinysea.gsp import State

ELEVATION = 0


def gpx_export(state_list: typing.List[State]) -> gpxpy.gpx.GPX:
    gpx = gpxpy.gpx.GPX()

    # Create track
    gpx_track = gpxpy.gpx.GPXTrack()
    gpx.tracks.append(gpx_track)

    # Create first segment in our GPX track:
    gpx_segment = gpxpy.gpx.GPXTrackSegment()
    gpx_track.segments.append(gpx_segment)

    # Create points:
    for s in state_list:
        lat, lon = s.position().to_lat_lon()
        gpx_segment.points.append(
            gpxpy.gpx.GPXTrackPoint(
                np.rad2deg(lat.t),
                np.rad2deg(lon.t),
                elevation=ELEVATION,
                time=datetime.fromtimestamp(s.time().t, timezone.utc),
            )
        )

    return gpx
