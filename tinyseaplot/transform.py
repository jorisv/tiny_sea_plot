import numpy as np


def lat_to_web_mercator(lat):
    """Convert latitude to web mercator"""
    k = 6378137
    return np.log(np.tan((90 + lat) * np.pi / 360.0)) * k


def lon_to_web_mercator(lon):
    """Convert longitude to web mercator"""
    k = 6378137
    return lon * (k * np.pi / 180.0)
