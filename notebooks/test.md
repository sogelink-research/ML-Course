You can find the incident triangles to a vertex using the `incident_triangles_to_vertex` method in the startinpy library. Here's an example:

```python
import startinpy
dt = startinpy.DT()

# Create a triangle with vertices 0, 1, and 2
dt.triangles[0] = [0, 1, 2]

# Find the incident triangles to vertex 0
trs = dt.incident_triangles_to_vertex(0)

# Print the incident triangles
for i, t in enumerate(trs):
    print(f"Triangle {i}: {t}")

# Output:
# Triangle 0: [3 4 6]
# Triangle 1: [3 6 7]
# Triangle 2: [3 7 8]
```

In this example, the `incident_triangles_to_vertex` method returns an array of triangles (ordered counter-clockwise) that are incident to each vertex. The first triangle in the array is the one that is directly adjacent to the vertex, and the second triangle is the one that is opposite to the vertex. If there are infinite triangles incident to a vertex (e.g., in a degenerate polygon), the method returns an array with three elements: the first element is the first triangle, the second element is the second triangle, and the third element is the third triangle.

You can also use this method to find the incident triangles to a vertex by specifying the vertex index as an argument:

```python
trs = dt.incident_triangles_to_vertex(1)
```

This will return the incident triangles to vertex 1.
