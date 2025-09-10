"""
Geometry Utilities for Accident Data Analysis
---------------------------------------------

Provides functions for:
1. Safe computation of alpha shapes.
2. Converting dataframes to numpy point arrays.
3. Fixing invalid geometries.
4. Checking if a geometry contains all points.

Dependencies:
- numpy, pandas, shapely, geopandas, alphashape, scipy, tqdm
"""

from typing import Iterable, Tuple, Dict, List

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from shapely.geometry import Point, MultiPoint, Polygon, MultiPolygon, LineString
from shapely.ops import unary_union, polygonize
from shapely.validation import make_valid

import geopandas
import geodatasets
import contextily as cx
import alphashape
from scipy.spatial import Delaunay, KDTree
from tqdm import tqdm

# -----------------------------
# SAFE ALPHASHAPE COMPUTATION
# -----------------------------
def alphashape_safe(points: np.ndarray, alpha: float):
    """
    Compute the alpha shape of a set of points safely.
    
    Returns None if computation fails or there are fewer than 3 points.
    
    Args:
        points (np.ndarray): Nx2 array of point coordinates [lon, lat].
        alpha (float): Alpha parameter for the alphashape.
    
    Returns:
        shapely.geometry.Polygon or None: Fixed geometry or None on failure.
    """
    if len(points) < 3:
        return None
    try:
        geom = alphashape.alphashape(points, alpha)
        return fix_geom(geom)
    except Exception:
        return None

# -----------------------------
# CONVERT DATAFRAME TO POINTS ARRAY
# -----------------------------
def to_points_array(df_region: pd.DataFrame) -> np.ndarray:
    """
    Convert a DataFrame of accidents to an Nx2 array of [Longitude, Latitude].
    
    Cleans non-numeric or missing values and filters out invalid coordinates.
    
    Args:
        df_region (pd.DataFrame): DataFrame with 'Longitude' and 'Latitude' columns.
    
    Returns:
        np.ndarray: Nx2 array of points.
    """
    df_region["Longitude"] = pd.to_numeric(df_region["Longitude"], errors="coerce")
    df_region["Latitude"] = pd.to_numeric(df_region["Latitude"], errors="coerce")
    
    pts = df_region[["Longitude", "Latitude"]].dropna().values
    # Remove obviously invalid coordinates (non-finite)
    pts = pts[np.isfinite(pts).all(axis=1)]
    return pts

# -----------------------------
# FIX INVALID GEOMETRY
# -----------------------------
def fix_geom(geom):
    """
    Attempt to fix invalid geometries robustly.
    
    Tries `make_valid` first, then fallback to `buffer(0)`.
    
    Args:
        geom (shapely.geometry.BaseGeometry): Input geometry.
    
    Returns:
        shapely.geometry.BaseGeometry or None: Fixed geometry or None.
    """
    if geom is None:
        return None
    try:
        if not geom.is_valid:
            geom = make_valid(geom)
    except Exception:
        geom = geom.buffer(0)
    return geom

# -----------------------------
# CHECK IF GEOMETRY CONTAINS ALL POINTS
# -----------------------------
def contains_all_points(geom, pts: np.ndarray, tolerance: float = 0.0) -> bool:
    """
    Check if a geometry contains all points (optionally with a small buffer).
    
    Uses bounding box prefilter for efficiency and precise shapely `contains` test.
    
    Args:
        geom (shapely.geometry.BaseGeometry): Geometry to check.
        pts (np.ndarray): Nx2 array of points [lon, lat].
        tolerance (float): Optional buffer to avoid precision issues.
    
    Returns:
        bool: True if all points are contained, False otherwise.
    """
    if geom is None or geom.is_empty:
        return False
    
    g = fix_geom(geom)
    if g is None or g.is_empty:
        return False
    
    if tolerance != 0:
        g = g.buffer(tolerance)
    
    # Fast bounding box check
    minx, miny, maxx, maxy = g.bounds
    mask = (pts[:,0] >= minx) & (pts[:,0] <= maxx) & (pts[:,1] >= miny) & (pts[:,1] <= maxy)
    if not mask.all():
        return False  # Any point outside bounding box fails fast
    
    # Precise check using shapely
    for x, y in pts:
        if not g.contains(Point(x, y)):
            return False
    
    return True
