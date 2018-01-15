import numpy as np
from modlamp.sequences import Helices, Random, AMPngrams
from modlamp.descriptors import PeptideDescriptor
from som import SOM

libnum = 1000
h = Helices(libnum)
r = Random(libnum)
n = AMPngrams(libnum, n_min=4)

h.generate_sequences()
r.generate_sequences(proba='AMP')
n.generate_sequences()

d = PeptideDescriptor(np.hstack((h.sequences, r.sequences, n.sequences)), 'pepcats')
d.calculate_crosscorr(7)

som = SOM(20, 20)
som.fit(d.descriptor, 2000)

targets = libnum*[0] + libnum*[1] + libnum*[2]
names = ['Helices', 'Random', 'nGrams']
som.plot_point_map(d.descriptor, targets, names, filename="peptidesom.png")
som.plot_density_map(d.descriptor, filename="density.png")
