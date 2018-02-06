import numpy as np
import matplotlib.pyplot as plt
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
    def __init__(self, x, y, alpha=0.6, alpha_final=0.1, seed=42):
        """ Initialize the SOM object with a given map size
        
        :param x: {int} width of the map
        :param y: {int} height of the map
        :param alpha: {float} initial alpha at training start
        :param alpha_final: {float} final alpha to reach at last training epoch
        :param seed: {int} random seed to use
        """
        np.random.seed(seed)
        self.x = x
        self.y = y
        self.shape = (x, y)
        self.sigma = x / 2.
        self.alpha = alpha
        self.alpha_final = alpha_final
        self.alpha_decay = float()
        self.sigma_decay = float()
        self.epoch = 0
        self.map = np.array([])
        self.indxmap = np.stack(np.unravel_index(np.arange(x * y, dtype=int).reshape(x, y), (x, y)), 2)
        self.distmap = np.array([])
        self.inizialized = False

    def winner(self, vector):
        """ Compute the winner neuron closest to the vector (Euclidean distance)
        
        :param vector: {numpy.ndarray} vector of current data point(s)
        :return: indices of winning neuron
        """
        delta = np.abs(self.map - vector)
        dists = np.sum(delta ** 2, axis=2)
        indx = np.argmin(dists)
        return np.array([indx / self.shape[0], indx % self.shape[1]])

    def cycle(self, vector):
        """ Perform one iteration in adapting the SOM towards the chosen data point
        
        :param vector: {numpy.ndarray} current data point
        """
        w = self.winner(vector)
        # get Manhattan distance (with PBC) of every neuron in the map to the winner
        dists = man_dist_pbc(self.indxmap, w, self.shape)

        # smooth the distances with the current sigma
        h = np.exp(-(dists / self.sigma) ** 2).reshape(self.x, self.y, 1)

        # update neuron weights
        self.map -= h * self.alpha * (self.map - vector)

        print("Epoch %i;  Neuron [%i, %i];  \tSigma: %.4f;  alpha: %.4f" % 
              (self.epoch, w[0], w[1], self.sigma, self.alpha))
        
        # update alpha, sigma and epoch
        self.alpha = self.alpha * self.alpha_decay
        self.sigma *= self.sigma_decay
        self.epoch = self.epoch + 1

    def fit(self, data, epochs, batch_size=1):
        """ Train the SOM on the given data for several iterations

        :param data: {numpy.ndarray} data to train on
        :param epochs: {int} number of iterations to train
        :param batch_size: {int} number of data points to consider per iteration
        """
        if not self.inizialized:
            # initialize map
            self.map = np.zeros((self.x, self.y, len(data[0])))
            eivalues = PCA(4).fit_transform(data.T).T
            for i in range(4):
                self.map[np.random.randint(0, self.x), np.random.randint(0, self.y)] = eivalues[i]

        # get decays for given epochs
        self.alpha_decay = (self.alpha_final / self.alpha) ** (1.0 / epochs)
        self.sigma_decay = (np.sqrt(self.x) / (4. * self.sigma)) ** (1.0 / epochs)

        samples = np.arange(len(data))
        for i in range(epochs):
            indx = np.random.choice(samples, batch_size)
            self.cycle(data[indx])

    def transform(self, data):
        """ Transform data in to the SOM space

        :param data: {numpy.ndarray} data to be transformed
        :return: transformed data in the SOM space
        """
        m = self.map.reshape((self.x * self.y, self.map.shape[-1]))
        dotprod = np.dot(np.exp(data), np.exp(m.T)) / np.sum(np.exp(m), axis=1)
        return (dotprod / (np.exp(np.max(dotprod)) + 1e-8)).reshape(data.shape[0], self.x, self.y)

    def distance_map(self):
        """ Get the distance map of the neuron weights. Every cell is the normalised sum of all distances between
        the neuron and its neighbors.

        :return: normalized sum of distances for every neuron to its neighbors
        """
        # TODO: make function working
        return None

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

    def som_error(self, data):
        """ Calculates the overall error as the average difference between the winning neurons and the data points

        :param data: {numpy.ndarray}
        :return: normalized error
        """
        e = float()
        for d in data:
            [x, y] = self.winner(d)
            dist = self.map[x, y] - d
            e += np.sqrt(np.dot(dist, dist.T))
        return e / len(data)

    def plot_point_map(self, data, targets, targetnames, filename=None, colors=None, markers=None, density=True):
        """ Visualize the som with all data as points around the neurons

        :param data: {numpy.ndarray} data to visualize with the SOM
        :param targets: {list/array} array of target classes (0 to len(targetnames)) corresponding to data
        :param targetnames: {list/array} names describing the target classes given in targets
        :param filename: {str} optional, if given, the plot is saved to this location
        :param colors: {list/array} optional, if given, different classes are colored in these colors
        :param markers: {list/array} optional, if given, different classes are visualized with these markers
        :param density: {bool} whether to plot the density map with winner neuron counts in the background
        :return: plot shown or saved if a filename is given
        """
        print("\nPlotting...")
        if not markers:
            markers = ['o', 'o', 'o', 'o', 'o']
        if not colors:
            colors = ['#ffa100', '#e900ff', '#00ffe1', '#ff0008', '#00ff19']
        if density:
            fig, ax = self.plot_density_map(data, internal=True)
        else:
            fig, ax = plt.subplots(figsize=self.shape)
        for cnt, xx in enumerate(data):
            w = self.winner(xx)
            ax.plot(w[0] + .5 + 0.15 * np.random.randn(1), w[1] + .5 + 0.15 * np.random.randn(1),
                    markers[targets[cnt]], color=colors[targets[cnt]], markersize=4, label=targetnames[targets[cnt]])

        ax.set_aspect('equal')
        ax.set_xlim([0, self.x])
        ax.set_ylim([0, self.y])
        ax.set_xticks(np.arange(self.x))
        ax.set_yticks(np.arange(self.y))
        ax.grid(which='both')
        legend = plt.legend(bbox_to_anchor=(0., 1.02, 1., .102), ncol=len(targetnames), loc=3, mode="expand",
                            borderaxespad=0.1)
        legend.get_frame().set_facecolor('#e5e5e5')

        if filename:
            plt.savefig(filename)
            plt.close()
            print("Done!")
        else:
            plt.show()

    def plot_density_map(self, data, filename=None, internal=False):
        """ Visualize the data density in different areas of the SOM.

        :param data: {numpy.ndarray} data to visualize the SOM density (number of times a neuron was winner)
        :param filename: {str} optional, if given, the plot is saved to this location
        :param internal: {bool} if True, the current plot will stay open to be used for other plot functions
        :return: plot shown or saved if a filename is given
        """
        wm = self.winner_map(data)
        fig, ax = plt.subplots(figsize=self.shape)
        plt.gray()
        plt.pcolormesh(wm.T)
        plt.colorbar()
        plt.xticks(np.arange(self.x))
        plt.yticks(np.arange(self.y))
        ax.set_aspect('equal')
        if not internal:
            if filename:
                plt.savefig(filename)
                plt.close()
                print("Done!")
            else:
                plt.show()
        else:
            return fig, ax

    def plot_class_density(self, data, targets, t, colormap='Oranges', filename=None):
        """ Plot a density map only for the given class

        :param data: {numpy.ndarray} data to visualize the SOM density (number of times a neuron was winner)
        :param targets: {list/array} array of target classes (0 to len(targetnames)) corresponding to data
        :param t: {int} target class to plot the density map for
        :param colormap: {str} colormap to use, select from matplolib sequential colormaps
        :param filename: {str} optional, if given, the plot is saved to this location
        :return: plot shown or saved if a filename is given
        """
        t_data = data[np.where(targets == t)[0]]
        wm = self.winner_map(t_data)
        fig, ax = plt.subplots(figsize=self.shape)
        plt.gray()
        plt.pcolormesh(wm.T, cmap=colormap)
        plt.colorbar()
        plt.xticks(np.arange(self.x))
        plt.yticks(np.arange(self.y))
        plt.title('Class %s' % t, fontweight='bold', fontsize=20)
        ax.set_aspect('equal')
        if filename:
            plt.savefig(filename)
            plt.close()
            print("Done!")
        else:
            plt.show()
