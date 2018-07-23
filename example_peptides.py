#! /usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
from modlamp.sequences import Helices, Random, AMPngrams
from modlamp.descriptors import PeptideDescriptor
from modlamp.datasets import load_AMPvsTM
from som import SOM

# generate some virtual peptide sequences
libnum = 1000  # 1000 sequences per sublibrary
h = Helices(seqnum=libnum)
r = Random(seqnum=libnum)
n = AMPngrams(seqnum=libnum, n_min=4)
h.generate_sequences()
r.generate_sequences(proba='AMP')
n.generate_sequences()

# calculate molecular descirptors for the peptides
d = PeptideDescriptor(seqs=np.hstack((h.sequences, r.sequences, n.sequences)), scalename='pepcats')
d.calculate_crosscorr(window=7)

# train a som on the descriptors and print / plot the training error
som = SOM(x=12, y=12)
som.fit(data=d.descriptor, epochs=100000, decay='hill')
print("Fit error: %.4f" % som.error)
som.plot_error_history(filename="som_error.png")

# load known antimicrobial peptides (AMPs) and transmembrane sequences
dataset = load_AMPvsTM()
d2 = PeptideDescriptor(dataset.sequences, 'pepcats')
d2.calculate_crosscorr(7)
targets = np.array(libnum*[0] + libnum*[1] + libnum*[2] + 206*[3])
names = ['Helices', 'Random', 'nGrams', 'AMP']

# plot som maps with location of AMPs
som.plot_point_map(np.vstack((d.descriptor, d2.descriptor[206:])), targets, names, filename="peptidesom.png")
som.plot_density_map(np.vstack((d.descriptor, d2.descriptor)), filename="density.png")
som.plot_distance_map(colormap='Reds', filename="distances.png")

colormaps = ['Oranges', 'Purples', 'Greens', 'Reds']
for i, c in enumerate(set(targets)):
    som.plot_class_density(np.vstack((d.descriptor, d2.descriptor)), targets, c, names, colormap=colormaps[i],
                           filename='class%i.png' % c)

# get neighboring peptides (AMPs / TMs) for a sequence of interest
my_d = PeptideDescriptor(seqs='GLFDIVKKVVGALLAG', scalename='pepcats')
my_d.calculate_crosscorr(window=7)
som.get_neighbors(datapoint=my_d.descriptor, data=d2.descriptor, labels=dataset.sequences, d=0)
