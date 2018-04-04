# som
A simple self-organizing map implementation in Python.

Self-organizing maps are also called Kohonen maps and were invented by Teuvo Kohonen.(1) They are an unsupervised machine 
learning technique to efficiently create spatially organized internal representations of various types of data. For 
example, SOMs are well-suited for the visualization of high-dimensional data. 

This is a simple implementation of SOMs in Python. This SOM has periodic boundary conditions and therefore can be
imagined as a "donut". The implementation uses `numpy`.

### Usage
Download the file `som.py` and place it somewhere in your PYTHONPATH.

Then you can import and use the `SOM` class as follows: 

``` python
import numpy as np
from som import SOM

# generate some random data with 36 features
data1 = np.random.normal(loc=-.25, scale=0.5, size=(500, 36))
data2 = np.random.normal(loc=.25, scale=0.5, size=(500, 36))
data = np.vstack((data1, data2))

som = SOM(10, 10)  # initialize the SOM
som.fit(data, 2000)  # fit the SOM for 2000 epochs

targets = 500 * [0] + 500 * [1]  # create some dummy target values

# now visualize the learned representation with the class labels
som.plot_point_map(data, targets, ['class 1', 'class 2'], filename='som.png')
som.plot_class_density(data, targets, 0, filename='class_0.png')
```

The same way you can handle your own data.

The `SOM` class has the following methods:
- `winner(vector)`: compute the winner neuron closest to a given data point in `vector` (Euclidean distance)
- `cycle(vector)`: perform one iteration in adapting the SOM towards the chosen data point in `vector`
- `fit(data, epochs, batch_size=1)`: train the SOM on the given `data` for several `epochs`
- `transform(data)`: transform given `data` in to the SOM space
- `distance_map()`: get a map of every neuron and its distances to all neighbors
- `winner_map(data)`: get the number of times, a certain neuron in the trained SOM is winner for the given `data`
- `som_error(data)`: calculates the overall error as the average difference between the winning neurons and the `data`
- `plot_point_map(data, targets, targetnames, filename=None, colors=None, markers=None, density=True)`: visualize the som with all data as points around the neurons
- `plot_density_map(data, filename=None, internal=False)`: visualize the data density in different areas of the SOM.
- `plot_class_density(data, targets, t, colormap='Oranges', filename=None)`: plot a density map only for the given class


### References:
(1) Kohonen, T. Self-Organized Formation of Topologically Correct Feature Maps. Biol. Cybern. 1982, 43 (1), 59–69.

This work was partially inspired by [ramalina's som implementation](https://github.com/ramarlina/som "ramarlina's som github repo") and [JustGlowing's minisom](https://github.com/JustGlowing/minisom "JustGlowing's minisom github repo").
