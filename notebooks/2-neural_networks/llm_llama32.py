import torch
from transformers import pipeline

model_id = "meta-llama/Llama-3.2-1B-Instruct"
pipe = pipeline(
    "text-generation",
    model=model_id,
    torch_dtype=torch.bfloat16,
    device_map="auto",
)
messages = [
    # {
    #     "role": "system",
    #     "content": "You are a pirate chatbot who always responds in pirate speak!",
    # },
    {
        "role": "system",
        "content": """Below is an extract of the API documentation of startinpy, a python library for Delaunay triangulation.

# API 

startinpy does not have specific classes and/or objects for points, vertices, and triangles.
[NumPy arrays](https://numpy.org/doc/stable/reference/arrays.html) of floats and integers are instead used.

A **Point** is an array of 3 floats (x-coordinate, y-coordinate, z-coordinate):

```python
>>> import startinpy
>>> dt = startinpy.DT()
>>> dt.insert_one_pt([11.3, 22.2, 4.7])
>>> dt.points[1]
array([11.3, 22.2, 4.7])
```

A **Vertex** is an integer, it is the index in the array of points ({func}`startinpy.DT.points`, which is 0-based).

A **Triangle** is an array of 3 integers, the values of the indices of the 3 vertices (ordered counter-clockwise) in the array of points ({func}`startinpy.DT.points`, which is 0-based).

```python
>>> dt.triangles[2]
array([1, 3, 2], dtype=uint64)
>>> #-- z-coordinate of 3rd vertex of the same triangle
>>> dt.points[dt.triangles[2][2]][2]
3.3
```

## incident_triangles_to_vertex(vi)
Return the triangles incident to vertex vi. Infinite triangles are also returned. Exception thrown if vertex index doesn’t exist in the DT or if it has been removed.

### Parameters:
vi – the vertex index

### Returns:
an array of triangles (ordered counter-clockwise)

```python
trs = dt.incident_triangles_to_vertex(3)
for i, t in enumerate(trs):
    print(i, t)    
0 [3 4 6]
1 [3 6 7]
2 [3 7 8]
3 [3 8 2]
4 [3 2 9]
5 [3 9 4]
```""",
    },
    {
        "role": "user",
        "content": "How can I find the incident triangles to a vertex using startinpy?",
    },
]
outputs = pipe(
    messages,
    max_new_tokens=512,
)
print(outputs[0]["generated_text"][-1]["content"])
