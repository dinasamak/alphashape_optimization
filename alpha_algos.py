"""
Alpha Shape and Concave Hull Utilities
--------------------------------------

This module provides functions to:
1. Compute optimized alpha parameters for alpha shapes.
2. Generate concave hulls using a KNN-inspired Delaunay triangulation.
3. Provide a fast alternative to standard alpha-shape optimization.

Dependencies:
- numpy, pandas, shapely, alphashape, scipy, tqdm, matplotlib
"""

import os
import time
import math
import warnings
from typing import Iterable, Tuple, Dict, List

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from shapely.geometry import Point, MultiPoint, Polygon, MultiPolygon, LineString
from shapely.ops import unary_union, polygonize
from shapely.validation import make_valid

# Avoid Shapely 2.x plotting helper for portability
import alphashape
from scipy.spatial import Delaunay, KDTree
from tqdm import tqdm

from utils import *

# -----------------------------
# OPTIMIZED ALPHA SELECTION
# -----------------------------
def optimized_alpha(points: np.ndarray) -> int:
    """
    Compute an integer alpha parameter for alphashape using a two-pass method:
    1) Coarse search using predefined alpha values.
    2) Fine search via binary search in the selected interval.

    Constraints:
    - Polygon should include all cluster points.
    - Avoid multipolygons with holes.

    Args:
        points (np.ndarray): Array of points as (lon, lat) coordinates.

    Returns:
        int: Optimized alpha value.
    """
    PREDEFINED_ALPHAS = [0, 5, 10, 18, 25, 50, 100, 300, 500, 1000, 3000, 5000]

    if len(points) < 3:
        return 0  # Not enough points for a polygon

    best_alpha = None
    idx = None

    # 1) Coarse pass
    for i in range(1, len(PREDEFINED_ALPHAS)):
        a = PREDEFINED_ALPHAS[i]
        geom = alphashape_safe(points, a)
        if geom is None:
            continue

        # Accept only single polygons
        valid_single = isinstance(geom, Polygon)
        if not valid_single:
            best_alpha, idx = PREDEFINED_ALPHAS[i - 1], i - 1
            break

    if best_alpha is None:
        return 0  # Fallback to convex hull

    # 2) Fine pass: binary search within interval
    lo = PREDEFINED_ALPHAS[idx]
    hi = PREDEFINED_ALPHAS[idx + 1]

    while hi - lo > 1:
        mid = (hi + lo) // 2
        geom = alphashape_safe(points, mid)
        if geom is None:
            lo = mid
            continue

        valid_single = isinstance(geom, Polygon)
        if valid_single:
            best_alpha = mid
            hi = mid
            break
        else:
            hi = mid
    return int(best_alpha)

# -----------------------------
# CONCAVE HULL VIA KNN-DELAUNAY
# -----------------------------
def concave_hull_knn(points: np.ndarray, k: int = 3, scale: float = 1.8) -> Polygon:
    """
    Approximate a concave hull using a KNN-inspired Delaunay triangulation.

    Steps:
    1. Build KDTree to compute k-nearest neighbor distances.
    2. Compute median k-distance, scale it for threshold.
    3. Build Delaunay triangulation.
    4. Keep triangles where all edges <= threshold.
    5. Polygonize kept triangle edges and union polygons.

    Args:
        points (np.ndarray): Array of points as (lon, lat).
        k (int): Number of neighbors for distance computation.
        scale (float): Scale factor for distance threshold.

    Returns:
        Polygon or None: Largest polygon generated, or None if insufficient data.
    """
    n = len(points)
    if n < 3:
        return None

    tree = KDTree(points)
    dists, _ = tree.query(points, k=k + 1)
    dk = dists[:, -1]
    thresh = float(np.median(dk) * scale)

    try:
        tri = Delaunay(points)
    except Exception:
        return None

    edges = set()
    for tri_idx in tri.simplices:
        p0, p1, p2 = points[tri_idx[0]], points[tri_idx[1]], points[tri_idx[2]]
        e01 = np.linalg.norm(p0 - p1)
        e12 = np.linalg.norm(p1 - p2)
        e20 = np.linalg.norm(p2 - p0)
        if e01 <= thresh and e12 <= thresh and e20 <= thresh:
            edges.add(tuple(sorted((tri_idx[0], tri_idx[1]))))
            edges.add(tuple(sorted((tri_idx[1], tri_idx[2]))))
            edges.add(tuple(sorted((tri_idx[2], tri_idx[0]))))

    if not edges:
        return None

    # Polygonize edges
    lines = [LineString([points[i], points[j]]) for i, j in edges]
    mls_union = unary_union(lines)
    polys = list(polygonize(mls_union))
    if not polys:
        return None

    geom = unary_union(polys)
    geom = fix_geom(geom)

    if isinstance(geom, MultiPolygon):
        geom = max(geom.geoms, key=lambda g: g.area)

    return geom

# -----------------------------
# SLOW ALPHA OPTIMIZATION (REFERENCE)
# -----------------------------
def optimizealpha_code_from_alpha_shape(
    points,
    max_iterations: int = 10000,
    lower: float = 0.,
    upper: float = 500
) -> float:
    """
    Slow alpha optimization routine from the alphashape package.

    Args:
        points: Iterable of points or GeoDataFrame['geometry'].
        max_iterations (int): Maximum number of iterations.
        lower (float): Lower bound for alpha.
        upper (float): Upper bound for alpha.

    Returns:
        float: Optimized alpha parameter.
    """
    # Convert to shapely multipoint if GeoDataFrame
    if isinstance(points, gpd.GeoDataFrame):
        points = points['geometry']

    counter = 0
    while (upper - lower) > np.finfo(float).eps * 2:
        test_alpha = (upper + lower) * 0.5

        if alphashape_safe(points, test_alpha):
            lower = test_alpha
        else:
            upper = test_alpha

        counter += 1
        if counter > max_iterations:
            warnings.warn(
                'Maximum allowed iterations reached while optimizing alpha parameter'
            )
            lower = 0.
            break

    return lower
