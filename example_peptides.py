#! /usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
from modlamp.sequences import Helices, Random, AMPngrams
from modlamp.descriptors import PeptideDescriptor
from modlamp.datasets import load_AMPvsTM
from som import SOM

libnum = 3000
h = Helices(libnum)
r = Random(libnum)
n = AMPngrams(libnum, n_min=4)

h.generate_sequences()
r.generate_sequences(proba='AMP')
n.generate_sequences()

d = PeptideDescriptor(np.hstack((h.sequences, r.sequences, n.sequences)), 'pepcats')
d.calculate_crosscorr(7)

som = SOM(20, 20)
som.fit(d.descriptor, 4000)

dataset = load_AMPvsTM()
d2 = PeptideDescriptor(dataset.sequences, 'pepcats')
d2.calculate_crosscorr(7)

targets = libnum*[0] + libnum*[1] + libnum*[2] + 206*[3] + 206*[4]
names = ['Helices', 'Random', 'nGrams', 'TM', 'AMP']
som.plot_point_map(np.vstack((d.descriptor, d2.descriptor)), targets, names, filename="peptidesom.png")
som.plot_density_map(np.vstack((d.descriptor, d2.descriptor)), filename="density.png")
