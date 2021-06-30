# an example of how to generate a GHH function
First, one should make sure the required libraries are installed(PyTorch, numpy, ...)

To begin with, we first import the GHH class and the grid methods.
```python
from grid_method import *
from model import GHH
```

For the case where the input is 2D, we construct the input grid.
```python
data, l = generate_grid(x_range, y_range, stepsize)
data.requires_grad_(True)
```

Then, we can generate the GHH based on this input.
```python
ghh = GHH(data, 2, 3)  # 2 is the input dimension, 3 is the value of K in the GHH representation formula
```
