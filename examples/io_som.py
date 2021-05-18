#! /usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Alex Müller     2021-05-18      Created
"""

import logging
import os
import sys
import time
from argparse import ArgumentParser

import numpy as np
import pandas as pd
from som import SOM

logger = logging.getLogger(__name__)
__version__ = '1.0'
__author__ = 'Alex Müller'


def main(in_file, out_file, x, y, epochs, ref=None, test=False, verbose=0):
    if test:
        df = pd.DataFrame(in_file, columns=range(in_file.shape[1]))
    else:
        df = pd.read_table(in_file, sep='\t', low_memory=True, index_col=0)

    s = df.shape[0]
    df.dropna(axis=0, how='any', inplace=True)
    sn = df.shape[0]
    if s != sn:
        logger.warning('%d rows dropped due to missing values' % (s - sn))

    s = df.shape[1]
    df = df.select_dtypes(include=[np.number])
    sn = df.shape[1]
    if s != sn:
        logger.warning('%d columns dropped due to non-numeric data type' % (s - sn))

    basedir = os.path.dirname(os.path.abspath(__file__))
    som = SOM(x, y)
    if ref == 'IRCI':
        som = som.load('/SOM.pkl')
        embedding = som.winner_neurons(df.values)
    else:
        som.fit(df.values, epochs, verbose=verbose)
        embedding = som.winner_neurons(df.values)
        if ref == 'Create':
            som.save(basedir + '/SOM.pkl')

    emb_df = pd.DataFrame({'ID': df.index})
    emb_df['X'] = embedding[:, 1]
    emb_df['Y'] = embedding[:, 0]
    if test:
        return emb_df
    else:
        emb_df.to_csv(out_file, index=False, sep='\t')


if __name__ == "__main__":
    description = "Self-Organizing Map\n\n"
    description += "%s [options] -i infile -o outfile\n\n" % os.path.split(__file__)[1]
    description += "%s: version %s - created by %s\n" % (os.path.split(__file__)[1], __version__, __author__)

    parser = ArgumentParser(description=description)
    parser.add_argument('-i', '--infile', dest='file_in', metavar='FILE',
                        help='Specify the input file (TAB format with ID in column 1)', action='store', default="-")
    parser.add_argument('-o', '--outfile', dest='file_out', metavar='FILE',
                        help='Specify the output file (default is STDOUT).', action='store', default="-")
    parser.add_argument('-x', '--x', dest='x', action='store', type=int, default=10,
                        help='Size of the SOM in x-coordinate')
    parser.add_argument('-y', '--y', dest='y', action='store', type=int, default=10,
                        help='Size of the SOM in y-coordinate')
    parser.add_argument('-e', '--epochs', dest='epochs', action='store', type=int, default=1000,
                        help='Number of epochs to train.')
    parser.add_argument('-r', '--ref', dest='ref', choices=['Create', 'IRCI', 'None'], default='None',
                        help='Use or create a reference PCA / UMAP model. If `None`, a new one is trained (not saved).')
    parser.add_argument('-v', '--verbose', dest='verbose', const=1, default=0, type=int, nargs="?",
                        help="increase verbosity: 0=warnings, 1=info, 2=debug. No number means info. Default is 0.")
    parser.add_argument('-s', '--test', dest='test', type=bool, default=False, action='store', help='Use for testing.')
    args = parser.parse_args()

    if args.test:
        import matplotlib.pyplot as plt
        from sklearn.datasets import make_blobs

        X = make_blobs(n_features=512, cluster_std=3.)
        T = main(X[0], None, args.x, args.y, args.epochs, ref=args.ref, test=True)
        plt.scatter(T['X'], T['Y'], c=X[1])
        plt.title('SOM test plot')
        plt.savefig('SOM_test.png')
    else:
        infile = sys.stdin if args.file_in == '-' else args.file_in
        outfile = sys.stdout if args.file_out == '-' else args.file_out

        # Start Time Monitoring
        timestart = time.time()

        # Initialisation...
        level = logging.WARNING
        if args.verbose == 1:
            level = logging.INFO
        elif args.verbose == 2:
            level = logging.DEBUG
        logging.basicConfig(level=level, format="%(asctime)s %(module)s %(levelname)-7s %(message)s",
                            datefmt="%Y/%b/%d %H:%M:%S")

        try:
            main(infile, outfile, args.x, args.y, args.epochs, args.ref)
        except Exception as err:
            logger.warning('Error occurred: %s' % str(err))

        if args.verbose:
            timetotal = time.time() - timestart
            logger.info("%s completed!" % os.path.split(__file__)[1])
            logger.info('Total wall time in seconds was: %f' % timetotal)

        # Close properly
        logging.shutdown()
        sys.stdin.close()
        sys.stdout.close()
