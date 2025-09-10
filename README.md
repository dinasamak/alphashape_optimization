
# Optimization of 2D Surface Detection and Edge Reconstruction using Alpha Shape


The project improves upon existing approaches by optimizing alpha values efficiently and accurately.


## Alpha Shape Optimization


### Algorithm
To compute the alpha shape:
1. Generate the Delaunay triangulation of the point set.
2. For each triangle, compute its area and edge lengths.
3. If the circumradius ≤ 1/α, include the triangle edges in the alpha shape.
4. Merge triangles to form the final polygon.


### Alpha Value Optimization
- Large alpha → may miss points
- Small alpha → non-exact representation
- Alpha = 0 → convex hull


**Optimized approach:**
- Precompute an array of candidate alpha values (e.g., `[5000, 3000, 1000, …, 0]`).
- Identify sub-optimal alpha, then refine using binary search.
- Ensures all points are included, avoiding multi-polygons or empty regions.
- Significantly reduces computation time compared to standard implementations.


## Case Study - Clusters Representation


### Dataset
- **Source:** UK road accidents (2014)
- **Size:** 146,322 accidents across 207 regions
- **Key Features:** Latitude, Longitude, Local Authority ID
- **Link:** [UK Accidents Dataset](https://www.kaggle.com/silicon99/dft-accident-data)


Optimized alpha shapes were used to generate polygon clusters representing accident locations.


### Comparative Study
- **Convex Hull:** Fast but inaccurate for exact boundaries.
- **Original Alpha Shape:** Accurate but slow (hours for large datasets).
- **Optimized Alpha Shape:** Accurate and extremely fast (seconds instead of hours).
- **Concave Hull (KNN-based):** Alternative approach; slower and less consistent.


**Processing Time Comparison:**


| Algorithm | Time (s) |
|-----------|-----------|
| Original Alpha Shape | 5529.9 |
| Optimized Alpha Shape | 0.48 |
| Convex Hull | 0.7 |


The optimized algorithm drastically reduces processing time while maintaining accuracy.


## Conclusion
- Polygons are essential for representing 2D spatial clusters.
- Convex hulls are quick but often inaccurate.
- Optimized alpha shapes provide accurate cluster representation with dramatic improvements in computational efficiency.


## References
1. Edelsbrunner, H., Kirkpatrick, D.G., Seidel, R.: *On the shape of a set of points in the plane*. IEEE Trans. IT 29, 551–559 (1983)
2. Duckham, M., et al.: *Efficient generation of simple polygons for characterizing point sets*. Pattern Recognition 41, 3224–3236 (2008)
3. [AlphaShape Python Library](https://pypi.org/project/alphashape/)
4. [UK Accidents Dataset](https://github.com/joaofig/uk-accidents)
*(See full references in the project PDF)*


## Usage
1. Clone the repository:
```bash
git clone <your-repo-url>
cd <repo-directory>

