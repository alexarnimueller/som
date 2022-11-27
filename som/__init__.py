import pickle
from multiprocessing import Process, Queue, cpu_count
from typing import Union

import matplotlib.patches as mptchs
import matplotlib.pyplot as plt
import numpy as np
from scipy.spatial.distance import cdist
from sklearn.decomposition import PCA

__author__ = "Alex Müller"
__docformat__ = "restructuredtext en"


def man_dist_pbc(m: np.ndarray, vector: np.ndarray, shape: tuple = (10, 10)) -> np.ndarray:
    """Manhattan distance calculation of coordinates with periodic boundary condition

    :param m: array / matrix (reference)
    :type m: np.ndarray
    :param vector: array / vector (target)
    :type vector: np.ndarray
    :param shape: shape of the full SOM
    :type shape: tuple, optional
    :return: Manhattan distance for v to m
    :rtype: np.ndarray
    """
    dims = np.array(shape)
    delta = np.abs(m - vector)
    delta = np.where(delta > 0.5 * dims, np.abs(delta - dims), delta)
    return np.sum(delta, axis=len(m.shape) - 1)


class SOM(object):
    """
    Class implementing a self-organizing map with periodic boundary conditions. It has the following methods:
    """

    def __init__(self, x: int, y: int, alpha_start: float = 0.6, sigma_start: float = None, seed: int = None):
        """Initialize the SOM object with a given map size and training conditions

        :param x: width of the map
        :type x: int
        :param y: height of the map
        :type y: int
        :param alpha_start: initial alpha (learning rate) at training start
        :type alpha_start: float
        :param sigma_start: initial sigma (restraint / neighborhood function) at training start; if `None`: x / 2
        :type sigma_start: float
        :param seed: random seed to use
        :type seed: int
        """
        np.random.seed(seed)
        self.x = x
        self.y = y
        self.shape = (x, y)
        if sigma_start:
            self.sigma = sigma_start
        else:
            self.sigma = x / 2.0
        self.alpha_start = alpha_start
        self.alphas = None
        self.sigmas = None
        self.epoch = 0
        self.interval = 0
        self.map = np.array([])
        self.indxmap = np.stack(np.unravel_index(np.arange(x * y, dtype=int).reshape(x, y), (x, y)), 2)
        self.distmap = np.zeros((self.x, self.y))
        self.winner_indices = np.array([])
        self.pca = None  # attribute to save potential PCA for saving and later reloading
        self.inizialized = False
        self.error = 0.0  # reconstruction error
        self.history = []  # reconstruction error training history

    def initialize(self, data: np.ndarray, how: str = "pca"):
        """Initialize the SOM neurons

        :param data: data to use for initialization
        :type data: numpy.ndarray
        :param how: how to initialize the map, available: `pca` (via 4 first eigenvalues) or `random` (via random
            values normally distributed in the shape of `data`)
        :type how: str
        :return: initialized map in :py:attr:`SOM.map`
        """
        self.map = np.random.normal(np.mean(data), np.std(data), size=(self.x, self.y, len(data[0])))
        if how == "pca":
            eivalues = PCA(4).fit_transform(data.T).T
            for i in range(4):
                self.map[np.random.randint(0, self.x), np.random.randint(0, self.y)] = eivalues[i]

        self.inizialized = True

    def winner(self, vector: np.ndarray) -> np.ndarray:
        """Compute the winner neuron closest to the vector (Euclidean distance)

        :param vector: vector of current data point(s)
        :type vector: np.ndarray
        :return: indices of winning neuron
        :rtype: np.ndarray
        """
        indx = np.argmin(np.sum((self.map - vector) ** 2, axis=2))
        return np.array([indx // self.y, indx % self.y])

    def cycle(self, vector: np.ndarray, verbose: bool = True):
        """Perform one iteration in adapting the SOM towards a chosen data point

        :param vector: current data point
        :type vector: np.ndarray
        :param verbose: verbosity control
        :type verbose: bool
        """
        w = self.winner(vector)
        # get Manhattan distance (with PBC) of every neuron in the map to the winner
        dists = man_dist_pbc(self.indxmap, w, self.shape)
        # smooth the distances with the current sigma
        h = np.exp(-((dists / self.sigmas[self.epoch]) ** 2)).reshape(self.x, self.y, 1)
        # update neuron weights
        self.map -= h * self.alphas[self.epoch] * (self.map - vector)

        if verbose:
            print(
                "Epoch %i;    Neuron [%i, %i];    \tSigma: %.4f;    alpha: %.4f"
                % (self.epoch, w[0], w[1], self.sigmas[self.epoch], self.alphas[self.epoch])
            )
        self.epoch = self.epoch + 1

    def fit(
        self,
        data: np.ndarray,
        epochs: int = 0,
        save_e: bool = False,
        interval: int = 1000,
        decay: str = "hill",
        verbose: bool = True,
    ):
        """Train the SOM on the given data for several iterations

        :param data: data to train on
        :type data: np.ndarray
        :param epochs: number of iterations to train; if 0, epochs=len(data) and every data point is used once
        :type epochs: int, optional
        :param save_e: whether to save the error history
        :type save_e: bool, optional
        :param interval: interval of epochs to use for saving training errors
        :type interval: int, optional
        :param decay: type of decay for alpha and sigma. Choose from 'hill' (Hill function) and 'linear', with
            'hill' having the form ``y = 1 / (1 + (x / 0.5) **4)``
        :type decay: str, optional
        :param verbose: verbosity control
        :type verbose: bool
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
        if decay == "hill":
            epoch_list = np.linspace(0, 1, epochs)
            self.alphas = self.alpha_start / (1 + (epoch_list / 0.5) ** 4)
            self.sigmas = self.sigma / (1 + (epoch_list / 0.5) ** 4)
        else:
            self.alphas = np.linspace(self.alpha_start, 0.05, epochs)
            self.sigmas = np.linspace(self.sigma, 1, epochs)

        if save_e:  # save the error to history every "interval" epochs
            for i in range(epochs):
                self.cycle(data[indx[i]], verbose=verbose)
                if i % interval == 0:
                    self.history.append(self.som_error(data))
            self.error = self.som_error(data)
        else:
            for i in range(epochs):
                self.cycle(data[indx[i]], verbose=verbose)

    def transform(self, data: np.ndarray) -> np.ndarray:
        """Transform data in to the SOM space

        :param data: data to be transformed
        :type data: np.ndarray
        :return: transformed data in the SOM space
        :rtype: np.ndarray
        """
        m = self.map.reshape((self.x * self.y, self.map.shape[-1]))
        dotprod = np.exp(data).dot(np.exp(m.T)) / np.exp(m).sum(axis=1)
        return (dotprod / (np.exp(dotprod.max()) + 1e-8)).reshape(data.shape[0], self.x, self.y)

    def distance_map(self, metric: str = "euclidean"):
        """Get the distance map of the neuron weights. Every cell is the normalised average of all distances between
        the neuron and all other neurons.

        :param metric: distance metric to be used (see ``scipy.spatial.distance.cdist``)
        :type metric: str
        :return: normalized sum of distances for every neuron to its neighbors, stored in :py:attr:`SOM.distmap`
        """
        dists = np.zeros((self.x, self.y))
        for x in range(self.x):
            for y in range(self.y):
                d = cdist(self.map[x, y].reshape((1, -1)), self.map.reshape((-1, self.map.shape[-1])), metric=metric)
                dists[x, y] = np.mean(d)
        self.distmap = dists / dists.max()

    def winner_map(self, data: np.ndarray) -> np.ndarray:
        """Get the number of times, a certain neuron in the trained SOM is the winner for the given data.

        :param data: data to compute the winner neurons on
        :type data: np.ndarray
        :return: map with winner counts at corresponding neuron location
        :rtype: np.ndarray
        """
        wm = np.zeros(self.shape, dtype=int)
        for d in data:
            [x, y] = self.winner(d)
            wm[x, y] += 1
        return wm

    def _one_winner_neuron(self, data: np.ndarray, q: Queue):
        """Private function to be used for parallel winner neuron computation

        :param data: data matrix to compute the winner neurons on
        :type data: np.ndarray
        :param q: queue
        :type q: multiprocessing.Queue
        :return: winner neuron cooridnates for every datapoint (see :py:method:`SOM.winner_neurons`)
        """
        q.put(np.array([self.winner(d) for d in data], dtype="int"))

    def winner_neurons(self, data: np.ndarray) -> np.ndarray:
        """For every datapoint, get the winner neuron coordinates.

        :param data: data to compute the winner neurons on
        :type data: np.ndarray
        :return: winner neuron coordinates for every datapoint
        :rtype: np.ndarray
        """
        print("Calculating neuron indices for all data points...")
        queue = Queue()
        n = cpu_count() - 1
        for d in np.array_split(np.array(data), n):
            p = Process(
                target=self._one_winner_neuron,
                args=(
                    d,
                    queue,
                ),
            )
            p.start()
        rslt = []
        for _ in range(n):
            rslt.extend(queue.get(timeout=10))
        self.winner_indices = np.array(rslt, dtype="int").reshape((len(data), 2))
        return self.winner_indices

    def _one_error(self, data: np.ndarray, q: Queue):
        """Private function to be used for parallel error calculation

        :param data: data matrix to calculate SOM error for
        :type data: np.ndarray
        :param q: queue
        :type q: multiprocessing.Queue
        :return: list of SOM errors (see :py:method:`SOM.som_error`)
        """
        errs = []
        for d in data:
            [x, y] = self.winner(d)
            dist = self.map[x, y] - d
            errs.append(np.sqrt(dist.dot(dist.T)))
        q.put(errs)

    def som_error(self, data: np.ndarray) -> float:
        """Calculates the overall error as the average difference between the winning neurons and the data points

        :param data: data to calculate the overall error for
        :type data: np.ndarray
        :return: normalized error
        :rtype: float
        """
        queue = Queue()
        for d in np.array_split(np.array(data), cpu_count()):
            p = Process(
                target=self._one_error,
                args=(
                    d,
                    queue,
                ),
            )
            p.start()
        rslt = []
        for _ in range(cpu_count()):
            rslt.extend(queue.get(timeout=50))
        return float(sum(rslt) / float(len(data)))

    def get_neighbors(self, datapoint: np.ndarray, data: np.ndarray, labels: np.ndarray, d: int = 0) -> np.ndarray:
        """return the labels of the neighboring data instances at distance `d` for a given data point of interest

        :param datapoint: descriptor vector of the data point of interest to check for neighbors
        :type datapoint: np.ndarray
        :param data: reference data to compare `datapoint` to
        :type data: np.ndarray
        :param labels: array of labels describing the target classes for every data point in `data`
        :type labels: np.ndarray
        :param d: length of Manhattan distance to explore the neighborhood (0: same neuron as data point)
        :type d: int
        :return: found neighbors (labels)
        :rtype: np.ndarray
        """
        if not len(self.winner_indices):
            _ = self.winner_neurons(data)
        labels = np.array(labels)
        w = self.winner(datapoint)
        print("Winner neuron of given data point: [%i, %i]" % (w[0], w[1]))
        dists = np.array([man_dist_pbc(winner, w, self.shape) for winner in self.winner_indices]).flatten()
        return labels[np.where(dists <= d)[0]]

    def plot_point_map(
        self,
        data: np.ndarray,
        targets: Union[list, np.ndarray],
        targetnames: Union[list, np.ndarray],
        filename: Union[str, None] = None,
        colors: Union[list, np.ndarray, None] = None,
        markers: Union[list, np.ndarray, None] = None,
        colormap: str = "gray",
        example_dict: Union[dict, None] = None,
        density: bool = True,
        activities: Union[list, np.ndarray, None] = None,
    ):
        """Visualize the som with all data as points around the neurons

        :param data: data to visualize with the SOM
        :type data: np.ndarray
        :param targets: array of target classes (0 to len(targetnames)) corresponding to data
        :type targets: list, np.ndarray
        :param targetnames: names describing the target classes given in targets
        :type targetnames: list, np.ndarray
        :param filename: if provided, the plot is saved to this location
        :type filename: str, optional
        :param colors: if provided, different classes are colored in these colors
        :type colors: list, np.ndarray, None; optional
        :param markers: if provided, different classes are visualized with these markers
        :type markers: list, np.ndarray, None; optional
        :param colormap: colormap to use, select from matplolib sequential colormaps
        :type colormap: str
        :param example_dict: dictionary containing names of examples as keys and corresponding descriptor values
            as values. These examples will be mapped onto the density map and marked
        :type example_dict: dict
        :param density: whether to plot the density map with winner neuron counts in the background
        :type density: bool
        :param activities: list of activities (e.g. IC50 values) to use for coloring the points
            accordingly; high values will appear in blue, low values in green
        :type activities: list, np.ndarray, None; optional
        :return: plot shown or saved if a filename is given
        """
        if not markers:
            markers = ["o"] * len(targetnames)
        if not colors:
            colors = ["#EDB233", "#90C3EC", "#C02942", "#79BD9A", "#774F38", "gray", "black"]
        if activities:
            heatmap = plt.get_cmap("coolwarm").reversed()
            colors = [heatmap(a / max(activities)) for a in activities]
        if density:
            fig, ax = self.plot_density_map(data, colormap=colormap, internal=True)
        else:
            fig, ax = plt.subplots(figsize=(self.y, self.x))

        for cnt, xx in enumerate(data):
            if activities:
                c = colors[cnt]
            else:
                c = colors[targets[cnt]]
            w = self.winner(xx)
            ax.plot(
                w[1] + 0.5 + 0.1 * np.random.randn(1),
                w[0] + 0.5 + 0.1 * np.random.randn(1),
                markers[targets[cnt]],
                color=c,
                markersize=12,
            )

        ax.set_aspect("equal")
        ax.set_xlim([0, self.y])
        ax.set_ylim([0, self.x])
        plt.xticks(np.arange(0.5, self.y + 0.5), range(self.y))
        plt.yticks(np.arange(0.5, self.x + 0.5), range(self.x))
        plt.xlabel("y")
        plt.ylabel("x")
        ax.grid(which="both")

        if not activities:
            patches = [mptchs.Patch(color=colors[i], label=targetnames[i]) for i in range(len(targetnames))]
            legend = plt.legend(
                handles=patches,
                bbox_to_anchor=(0.0, 1.02, 1.0, 0.102),
                loc=3,
                ncol=len(targetnames),
                mode="expand",
                borderaxespad=0.1,
            )
            legend.get_frame().set_facecolor("#e5e5e5")

        if example_dict:
            for k, v in example_dict.items():
                w = self.winner(v)
                x = w[1] + 0.5 + np.random.normal(0, 0.15)
                y = w[0] + 0.5 + np.random.normal(0, 0.15)
                plt.plot(x, y, marker="*", color="#FDBC1C", markersize=24)
                plt.annotate(k, xy=(x + 0.5, y - 0.18), textcoords="data", fontsize=18, fontweight="bold")

        if filename:
            plt.savefig(filename)
            plt.close()
            print("Point map plot done!")
        else:
            plt.show()

    def plot_density_map(
        self,
        data: np.ndarray,
        colormap: str = "gray",
        filename: Union[str, None] = None,
        example_dict: Union[dict, None] = None,
        internal: bool = False,
    ):
        """Visualize the data density in different areas of the SOM.

        :param data: data to visualize the SOM density (number of times a neuron was winner)
        :type data: np.ndarray
        :param colormap: colormap to use, select from matplolib sequential colormaps
        :type colormap: str
        :param filename: optional, if given, the plot is saved to this location
        :type filename: str
        :param example_dict: dictionary containing names of examples as keys and corresponding descriptor values
            as values. These examples will be mapped onto the density map and marked
        :type example_dict: dict
        :param internal: if True, the current plot will stay open to be used for other plot functions
        :type internal: bool
        :return: plot shown or saved if a filename is given
        """
        wm = self.winner_map(data)
        fig, ax = plt.subplots(figsize=(self.y, self.x))
        plt.pcolormesh(wm, cmap=colormap, edgecolors=None)
        plt.colorbar()
        plt.xticks(np.arange(0.5, self.y + 0.5), range(self.y))
        plt.yticks(np.arange(0.5, self.x + 0.5), range(self.x))
        plt.xlabel("y")
        plt.ylabel("x")
        ax.set_aspect("equal")

        if example_dict:
            for k, v in example_dict.items():
                w = self.winner(v)
                x = w[1] + 0.5 + np.random.normal(0, 0.15)
                y = w[0] + 0.5 + np.random.normal(0, 0.15)
                plt.plot(x, y, marker="*", color="#FDBC1C", markersize=24)
                plt.annotate(k, xy=(x + 0.5, y - 0.18), textcoords="data", fontsize=18, fontweight="bold")

        if not internal:
            plt.title("Density Map")
            if filename:
                plt.savefig(filename)
                plt.close()
                print("Density map plot done!")
            else:
                plt.show()
        else:
            return fig, ax

    def plot_class_density(
        self,
        data: np.ndarray,
        targets: Union[list, np.ndarray],
        t: int = 1,
        name: str = "actives",
        colormap: str = "gray",
        example_dict: Union[dict, None] = None,
        filename: Union[str, None] = None,
    ):
        """Plot a density map only for the given class

        :param data: data to visualize the SOM density (number of times a neuron was winner)
        :type data: np.ndarray
        :param targets: array of target classes (0 to len(targetnames)) corresponding to data
        :type targets: list, np.ndarray
        :param t: target class to plot the density map for
        :type t: int
        :param name: target name corresponding to target given in t
        :type name: str
        :param colormap: colormap to use, select from matplolib sequential colormaps
        :type colormap: str
        :param example_dict: dictionary containing names of examples as keys and corresponding descriptor values
            as values. These examples will be mapped onto the density map and marked
        :type example_dict: dict
        :param filename: optional, if given, the plot is saved to this location
        :type filename: str
        :return: plot shown or saved if a filename is given
        """
        targets = np.array(targets)
        t_data = data[np.where(targets == t)[0]]
        wm = self.winner_map(t_data)
        fig, ax = plt.subplots(figsize=(self.y, self.x))
        plt.pcolormesh(wm, cmap=colormap, edgecolors=None)
        plt.colorbar()
        plt.xticks(np.arange(0.5, self.y + 0.5), range(self.y))
        plt.yticks(np.arange(0.5, self.x + 0.5), range(self.x))
        plt.title(name, fontweight="bold", fontsize=28)
        plt.xlabel("y")
        plt.ylabel("x")
        ax.set_aspect("equal")
        plt.text(0.1, -1.0, "%i Datapoints" % len(t_data), fontsize=20, fontweight="bold")

        if example_dict:
            for k, v in example_dict.items():
                w = self.winner(v)
                x = w[1] + 0.5 + np.random.normal(0, 0.15)
                y = w[0] + 0.5 + np.random.normal(0, 0.15)
                plt.plot(x, y, marker="*", color="#FDBC1C", markersize=24)
                plt.annotate(k, xy=(x + 0.5, y - 0.18), textcoords="data", fontsize=18, fontweight="bold")

        if filename:
            plt.savefig(filename)
            plt.close()
            print("Class density plot done!")
        else:
            plt.show()

    def plot_distance_map(self, colormap: str = "gray", filename: Union[str, None] = None):
        """Plot the distance map after training.

        :param colormap: colormap to use, select from matplolib sequential colormaps
        :type colormap: str
        :param filename: optional, if given, the plot is saved to this location
        :type filename: str
        :return: plot shown or saved if a filename is given
        """
        if np.mean(self.distmap) == 0.0:
            self.distance_map()
        fig, ax = plt.subplots(figsize=(self.y, self.x))
        plt.pcolormesh(self.distmap, cmap=colormap, edgecolors=None)
        plt.colorbar()
        plt.xticks(np.arange(0.5, self.y + 0.5), range(self.y))
        plt.yticks(np.arange(0.5, self.x + 0.5), range(self.x))
        plt.title("Distance Map", fontweight="bold", fontsize=28)
        plt.xlabel("y")
        plt.ylabel("x")
        ax.set_aspect("equal")
        if filename:
            plt.savefig(filename)
            plt.close()
            print("Distance map plot done!")
        else:
            plt.show()

    def plot_error_history(self, color: str = "orange", filename: Union[str, None] = None):
        """plot the training reconstruction error history that was recorded during the fit

        :param color: color of the line
        :type color: str
        :param filename: optional, if given, the plot is saved to this location
        :type filename: str
        :return: plot shown or saved if a filename is given
        """
        if not len(self.history):
            raise LookupError("No error history was found! Is the SOM already trained?")
        fig, ax = plt.subplots()
        ax.plot(range(0, self.epoch, self.interval), self.history, "-o", c=color)
        ax.set_title("SOM Error History", fontweight="bold")
        ax.set_xlabel("Epoch", fontweight="bold")
        ax.set_ylabel("Error", fontweight="bold")
        if filename:
            plt.savefig(filename)
            plt.close()
            print("Error history plot done!")
        else:
            plt.show()

    def save(self, filename: str):
        """Save the SOM instance to a pickle file.

        :param filename: filename (best to end with .p)
        :type filename: str
        :return: saved instance in file with name `filename`
        """
        f = open(filename, "wb")
        pickle.dump(self.__dict__, f, 2)
        f.close()

    def load(self, filename: str):
        """Load a SOM instance from a pickle file.

        :param filename: filename (best to end with .p)
        :type filename: str
        :return: updated instance with data from `filename`
        """
        f = open(filename, "rb")
        tmp_dict = pickle.load(f)
        f.close()
        self.__dict__.update(tmp_dict)
