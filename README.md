# som
A simple self-organizing map implementation in Python.

Self-organizing maps are also called Kohonen maps and were invented by Teuvo Kohonen.(1) They are an unsupervised machine 
learning technique to efficiently create spatially organized internal representations of various types of data. For 
example, SOMs are well-suited for the visualization of high-dimensional data. 

This is a simple implementation of SOMs in Python. This SOM has periodic boundary conditions and therefore can be
imagined as a "donut".

### Usage
Download the file `som.py` and place it somewhere in your PYTHONPATH.

Then you can import and use the `SOM` class as follows: 

``` python
import numpy as np
from som import SOM

data = np.random.random((1000, 36)  # generate some random data with 36 features

som = SOM(10, 10)  # initialize the SOM
som.fit(data, 2000)  # fit the SOM for 2000 epochs

targets = 500 * [0] + 500 * [1]  # create some dummy target values
# now visualize the learned representation with the class labels
som.plot_point_map(data, targets, ['class 1', 'class 2'], filename='som.png')
```

### References:
(1) Kohonen, T. Self-Organized Formation of Topologically Correct Feature Maps. Biol. Cybern. 1982, 43 (1), 59–69.