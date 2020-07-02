import pickle
from multiprocessing import cpu_count, Process, Queue

import matplotlib.patches as mptchs
import matplotlib.pyplot as plt
import numpy as np
from scipy.spatial.distance import cdist
from sklearn.decomposition import PCA


def man_dist_pbc(m, vector, shape=(10, 10)):
    """ Manhattan distance calculation of coordinates with periodic boundary condition

    :param m: {numpy.ndarray} array / matrix
    :param vector: {numpy.ndarray} array / vector
    :param shape: {tuple} shape of the SOM
    :return: {numpy.ndarray} Manhattan distance for v to m
    """
    dims = np.array(shape)
    delta = np.abs(m - vector)
    delta = np.where(delta > 0.5 * dims, np.abs(delta - dims), delta)
    return np.sum(delta, axis=len(m.shape) - 1)


class SOM(object):
    def __init__(self, x, y, alpha_start=0.6, seed=42):
        """ Initialize the SOM object with a given map size
        
        :param x: {int} width of the map
        :param y: {int} height of the map
        :param alpha_start: {float} initial alpha at training start
        :param seed: {int} random seed to use
        """
        np.random.seed(seed)
        self.x = x
        self.y = y
        self.shape = (x, y)
        self.sigma = x / 2.
        self.alpha_start = alpha_start
        self.alphas = None
        self.sigmas = None
        self.epoch = 0
        self.interval = int()
        self.map = np.array([])
        self.indxmap = np.stack(np.unravel_index(np.arange(x * y, dtype=int).reshape(x, y), (x, y)), 2)
        self.distmap = np.zeros((self.x, self.y))
        self.winner_indices = np.array([])
        self.pca = None  # attribute to save potential PCA to for saving and later reloading
        self.inizialized = False
        self.error = 0.  # reconstruction error
        self.history = list()  # reconstruction error training history

    def initialize(self, data, how='pca'):
        """ Initialize the SOM neurons

        :param data: {numpy.ndarray} data to use for initialization
        :param how: {str} how to initialize the map, available: 'pca' (via 4 first eigenvalues) or 'random' (via random
            values normally distributed like data)
        :return: initialized map in self.map
        """
        self.map = np.random.normal(np.mean(data), np.std(data), size=(self.x, self.y, len(data[0])))
        if how == 'pca':
            eivalues = PCA(4).fit_transform(data.T).T
            for i in range(4):
                self.map[np.random.randint(0, self.x), np.random.randint(0, self.y)] = eivalues[i]

        self.inizialized = True

    def winner(self, vector):
        """ Compute the winner neuron closest to the vector (Euclidean distance)

        :param vector: {numpy.ndarray} vector of current data point(s)
        :return: indices of winning neuron
        """
        indx = np.argmin(np.sum((self.map - vector) ** 2, axis=2))
        return np.array([int(indx / self.x), indx % self.y])

    def cycle(self, vector):
        """ Perform one iteration in adapting the SOM towards a chosen data point

        :param vector: {numpy.ndarray} current data point
        """
        w = self.winner(vector)
        # get Manhattan distance (with PBC) of every neuron in the map to the winner
        dists = man_dist_pbc(self.indxmap, w, self.shape)
        # smooth the distances with the current sigma
        h = np.exp(-(dists / self.sigmas[self.epoch]) ** 2).reshape(self.x, self.y, 1)
        # update neuron weights
        self.map -= h * self.alphas[self.epoch] * (self.map - vector)

        print("Epoch %i;    Neuron [%i, %i];    \tSigma: %.4f;    alpha: %.4f" %
              (self.epoch, w[0], w[1], self.sigmas[self.epoch], self.alphas[self.epoch]))
        self.epoch = self.epoch + 1

    def fit(self, data, epochs=0, save_e=False, interval=1000, decay='hill'):
        """ Train the SOM on the given data for several iterations

        :param data: {numpy.ndarray} data to train on
        :param epochs: {int} number of iterations to train; if 0, epochs=len(data) and every data point is used once
        :param save_e: {bool} whether to save the error history
        :param interval: {int} interval of epochs to use for saving training errors
        :param decay: {str} type of decay for alpha and sigma. Choose from 'hill' (Hill function) and 'linear', with
            'hill' having the form ``y = 1 / (1 + (x / 0.5) **4)``
        """
        self.interval = interval
        if not self.inizialized:
            self.initialize(data)
        if not epochs:
            epochs = len(data)
            indx = np.random.choice(np.arange(len(data)), epochs, replace=False)
        else:
            indx = np.random.choice(np.arange(len(data)), epochs)

        # get alpha and sigma decays for given number of epochs or for hill decay
        if decay == 'hill':
            epoch_list = np.linspace(0, 1, epochs)
            self.alphas = self.alpha_start / (1 + (epoch_list / 0.5) ** 4)
            self.sigmas = self.sigma / (1 + (epoch_list / 0.5) ** 4)
        else:
            self.alphas = np.linspace(self.alpha_start, 0.05, epochs)
            self.sigmas = np.linspace(self.sigma, 1, epochs)

        if save_e:  # save the error to history every "interval" epochs
            for i in range(epochs):
                self.cycle(data[indx[i]])
                if i % interval == 0:
                    self.history.append(self.som_error(data))
        else:
            for i in range(epochs):
                self.cycle(data[indx[i]])
        self.error = self.som_error(data)

    def transform(self, data):
        """ Transform data in to the SOM space

        :param data: {numpy.ndarray} data to be transformed
        :return: transformed data in the SOM space
        """
        m = self.map.reshape((self.x * self.y, self.map.shape[-1]))
        dotprod = np.dot(np.exp(data), np.exp(m.T)) / np.sum(np.exp(m), axis=1)
        return (dotprod / (np.exp(np.max(dotprod)) + 1e-8)).reshape(data.shape[0], self.x, self.y)

    def distance_map(self, metric='euclidean'):
        """ Get the distance map of the neuron weights. Every cell is the normalised sum of all distances between
        the neuron and all other neurons.

        :param metric: {str} distance metric to be used (see ``scipy.spatial.distance.cdist``)
        :return: normalized sum of distances for every neuron to its neighbors
        """
        dists = np.zeros((self.x, self.y))
        for x in range(self.x):
            for y in range(self.y):
                d = cdist(self.map[x, y].reshape((1, -1)), self.map.reshape((-1, self.map.shape[-1])), metric=metric)
                dists[x, y] = np.mean(d)
        self.distmap = dists / float(np.max(dists))

    def winner_map(self, data):
        """ Get the number of times, a certain neuron in the trained SOM is winner for the given data.

        :param data: {numpy.ndarray} data to compute the winner neurons on
        :return: {numpy.ndarray} map with winner counts at corresponding neuron location
        """
        wm = np.zeros(self.shape, dtype=int)
        for d in data:
            [x, y] = self.winner(d)
            wm[x, y] += 1
        return wm

    def _one_winner_neuron(self, data, q):
        """Private function to be used for parallel winner neuron computation

        :param data: {numpy.ndarray} data matrix to compute the winner neurons on
        :param q: {multiprocessing.Queue} queue
        :return: {list} winner neuron cooridnates for every datapoint
        """
        q.put(np.array([self.winner(d) for d in data], dtype='int'))

    def winner_neurons(self, data):
        """ For every datapoint, get the winner neuron coordinates.

        :param data: {numpy.ndarray} data to compute the winner neurons on
        :return: {numpy.ndarray} winner neuron coordinates for every datapoint
        """
        print("Calculating neuron indices for all data points...")
        queue = Queue()
        n = cpu_count() - 1
        for d in np.array_split(np.array(data), n):
            p = Process(target=self._one_winner_neuron, args=(d, queue,))
            p.start()
        rslt = []
        for _ in range(n):
            rslt.extend(queue.get(10))
        self.winner_indices = np.array(rslt, dtype='int').reshape((len(data), 2))

    def _one_error(self, data, q):
        """Private function to be used for parallel error calculation

        :param data: {numpy.ndarray} data matrix to calculate SOM error for
        :param q: {multiprocessing.Queue} queue
        :return: {list} list of SOM errors
        """
        errs = list()
        for d in data:
            w = self.winner(d)
            dist = self.map[w[0], w[1]] - d
            errs.append(np.sqrt(np.dot(dist, dist.T)))
        q.put(errs)

    def som_error(self, data):
        """ Calculates the overall error as the average difference between the winning neurons and the data points

        :param data: {numpy.ndarray}
        :return: normalized error
        """
        queue = Queue()
        for d in np.array_split(np.array(data), cpu_count()):
            p = Process(target=self._one_error, args=(d, queue,))
            p.start()
        rslt = []
        for _ in range(cpu_count()):
            rslt.extend(queue.get(50))
        return float(sum(rslt) / float(len(data)))

    def get_neighbors(self, datapoint, data, labels, d=0):
        """ return the neighboring data instances and their labels for a given data point of interest

        :param datapoint: {numpy.ndarray} descriptor vector of the data point of interest to check for neighbors
        :param data: {numpy.ndarray} reference data to compare ``datapoint`` to
        :param labels: {numpy.ndarray} array of labels describing the target classes for every data point in ``data``
        :param d: {int} length of Manhattan distance to explore the neighborhood (0: only same neuron as data point)
        :return: {numpy.ndarray} found neighbors (labels)
        """
        if not len(self.winner_indices):
            self.winner_neurons(data)
        labels = np.array(labels)
        w = self.winner(datapoint)
        print("Winner neuron of given data point: [%i, %i]" % (w[0], w[1]))
        dists = np.array([man_dist_pbc(winner, w, self.shape) for winner in self.winner_indices]).flatten()
        return labels[np.where(dists <= d)[0]]

    def plot_point_map(self, data, targets, targetnames, filename=None, colors=None, markers=None, example_dict=None,
                       density=True, activities=None):
        """ Visualize the som with all data as points around the neurons

        :param data: {numpy.ndarray} data to visualize with the SOM
        :param targets: {list/array} array of target classes (0 to len(targetnames)) corresponding to data
        :param targetnames: {list/array} names describing the target classes given in targets
        :param filename: {str} optional, if given, the plot is saved to this location
        :param colors: {list/array} optional, if given, different classes are colored in these colors
        :param markers: {list/array} optional, if given, different classes are visualized with these markers
        :param example_dict: {dict} dictionary containing names of examples as keys and corresponding descriptor values
            as values. These examples will be mapped onto the density map and marked
        :param density: {bool} whether to plot the density map with winner neuron counts in the background
        :param activities: {list/array} list of activities (e.g. IC50 values) to use for coloring the points
            accordingly; high values will appear in blue, low values in green
        :return: plot shown or saved if a filename is given
        """
        if not markers:
            markers = ['o'] * len(targetnames)
        if not colors:
            colors = ['#EDB233', '#90C3EC', '#C02942', '#79BD9A', '#774F38', 'gray', 'black']
        if activities:
            heatmap = plt.get_cmap('coolwarm').reversed()
            colors = [heatmap(a / max(activities)) for a in activities]
        if density:
            fig, ax = self.plot_density_map(data, internal=True)
        else:
            fig, ax = plt.subplots(figsize=self.shape)

        for cnt, xx in enumerate(data):
            if activities:
                c = colors[cnt]
            else:
                c = colors[targets[cnt]]
            w = self.winner(xx)
            ax.plot(w[1] + .5 + 0.1 * np.random.randn(1), w[0] + .5 + 0.1 * np.random.randn(1),
                    markers[targets[cnt]], color=c, markersize=12)

        ax.set_aspect('equal')
        ax.set_xlim([0, self.x])
        ax.set_ylim([0, self.y])
        plt.xticks(np.arange(.5, self.x + .5), range(self.x))
        plt.yticks(np.arange(.5, self.y + .5), range(self.y))
        ax.grid(which='both')

        if not activities:
            patches = [mptchs.Patch(color=colors[i], label=targetnames[i]) for i in range(len(targetnames))]
            legend = plt.legend(handles=patches, bbox_to_anchor=(0., 1.02, 1., .102), loc=3, ncol=len(targetnames),
                                mode="expand", borderaxespad=0.1)
            legend.get_frame().set_facecolor('#e5e5e5')

        if example_dict:
            for k, v in example_dict.items():
                w = self.winner(v)
                x = w[1] + 0.5 + np.random.normal(0, 0.15)
                y = w[0] + 0.5 + np.random.normal(0, 0.15)
                plt.plot(x, y, marker='*', color='#FDBC1C', markersize=24)
                plt.annotate(k, xy=(x + 0.5, y - 0.18), textcoords='data', fontsize=18, fontweight='bold')

        if filename:
            plt.savefig(filename)
            plt.close()
            print("Point map plot done!")
        else:
            plt.show()

    def plot_density_map(self, data, colormap='Oranges', filename=None, example_dict=None, internal=False):
        """ Visualize the data density in different areas of the SOM.

        :param data: {numpy.ndarray} data to visualize the SOM density (number of times a neuron was winner)
        :param colormap: {str} colormap to use, select from matplolib sequential colormaps
        :param filename: {str} optional, if given, the plot is saved to this location
        :param example_dict: {dict} dictionary containing names of examples as keys and corresponding descriptor values
            as values. These examples will be mapped onto the density map and marked
        :param internal: {bool} if True, the current plot will stay open to be used for other plot functions
        :return: plot shown or saved if a filename is given
        """
        wm = self.winner_map(data)
        fig, ax = plt.subplots(figsize=self.shape)
        plt.pcolormesh(wm, cmap=colormap, edgecolors=None)
        plt.colorbar()
        plt.xticks(np.arange(.5, self.x + .5), range(self.x))
        plt.yticks(np.arange(.5, self.y + .5), range(self.y))
        ax.set_aspect('equal')

        if example_dict:
            for k, v in example_dict.items():
                w = self.winner(v)
                x = w[1] + 0.5 + np.random.normal(0, 0.15)
                y = w[0] + 0.5 + np.random.normal(0, 0.15)
                plt.plot(x, y, marker='*', color='#FDBC1C', markersize=24)
                plt.annotate(k, xy=(x + 0.5, y - 0.18), textcoords='data', fontsize=18, fontweight='bold')

        if not internal:
            if filename:
                plt.savefig(filename)
                plt.close()
                print("Density map plot done!")
            else:
                plt.show()
        else:
            return fig, ax

    def plot_class_density(self, data, targets, t=1, name='actives', colormap='Oranges', example_dict=None,
                           filename=None):
        """ Plot a density map only for the given class

        :param data: {numpy.ndarray} data to visualize the SOM density (number of times a neuron was winner)
        :param targets: {list/array} array of target classes (0 to len(targetnames)) corresponding to data
        :param t: {int} target class to plot the density map for
        :param name: {str} target name corresponding to target given in t
        :param colormap: {str} colormap to use, select from matplolib sequential colormaps
        :param example_dict: {dict} dictionary containing names of examples as keys and corresponding descriptor values
            as values. These examples will be mapped onto the density map and marked
        :param filename: {str} optional, if given, the plot is saved to this location
        :return: plot shown or saved if a filename is given
        """
        targets = np.array(targets)
        t_data = data[np.where(targets == t)[0]]
        wm = self.winner_map(t_data)
        fig, ax = plt.subplots(figsize=self.shape)
        plt.pcolormesh(wm, cmap=colormap, edgecolors=None)
        plt.colorbar()
        plt.xticks(np.arange(.5, self.x + .5), range(self.x))
        plt.yticks(np.arange(.5, self.y + .5), range(self.y))
        plt.title(name, fontweight='bold', fontsize=28)
        ax.set_aspect('equal')
        plt.text(0.1, -1., "%i Datapoints" % len(t_data), fontsize=20, fontweight='bold')

        if example_dict:
            for k, v in example_dict.items():
                w = self.winner(v)
                x = w[1] + 0.5 + np.random.normal(0, 0.15)
                y = w[0] + 0.5 + np.random.normal(0, 0.15)
                plt.plot(x, y, marker='*', color='#FDBC1C', markersize=24)
                plt.annotate(k, xy=(x + 0.5, y - 0.18), textcoords='data', fontsize=18, fontweight='bold')

        if filename:
            plt.savefig(filename)
            plt.close()
            print("Class density plot done!")
        else:
            plt.show()

    def plot_distance_map(self, colormap='Oranges', filename=None):
        """ Plot the distance map after training.

        :param colormap: {str} colormap to use, select from matplolib sequential colormaps
        :param filename: {str} optional, if given, the plot is saved to this location
        :return: plot shown or saved if a filename is given
        """
        if np.mean(self.distmap) == 0.:
            self.distance_map()
        fig, ax = plt.subplots(figsize=self.shape)
        plt.pcolormesh(self.distmap, cmap=colormap, edgecolors=None)
        plt.colorbar()
        plt.xticks(np.arange(.5, self.x + .5), range(self.x))
        plt.yticks(np.arange(.5, self.y + .5), range(self.y))
        plt.title("Distance Map", fontweight='bold', fontsize=28)
        ax.set_aspect('equal')
        if filename:
            plt.savefig(filename)
            plt.close()
            print("Distance map plot done!")
        else:
            plt.show()

    def plot_error_history(self, color='orange', filename=None):
        """ plot the training reconstruction error history that was recorded during the fit

        :param color: {str} color of the line
        :param filename: {str} optional, if given, the plot is saved to this location
        :return: plot shown or saved if a filename is given
        """
        if not len(self.history):
            raise LookupError("No error history was found! Is the SOM already trained?")
        fig, ax = plt.subplots()
        ax.plot(range(0, self.epoch, self.interval), self.history, '-o', c=color)
        ax.set_title('SOM Error History', fontweight='bold')
        ax.set_xlabel('Epoch', fontweight='bold')
        ax.set_ylabel('Error', fontweight='bold')
        if filename:
            plt.savefig(filename)
            plt.close()
            print("Error history plot done!")
        else:
            plt.show()

    def save(self, filename):
        """ Save the SOM instance to a pickle file.

        :param filename: {str} filename (best to end with .p)
        :return: saved instance in file with name ``filename``
        """
        f = open(filename, 'wb')
        pickle.dump(self.__dict__, f, 2)
        f.close()

    def load(self, filename):
        """ Save the SOM instance to a pickle file.

        :param filename: {str} filename (best to end with .p)
        :return: updated instance with data from ``filename``
        """
        f = open(filename, 'rb')
        tmp_dict = pickle.load(f)
        f.close()
        self.__dict__.update(tmp_dict)
