
import os
import sys
import numpy as np
import h5py
import argparse
import scipy.sparse
from sklearn.neighbors import KDTree
import multiprocessing as multiproc
from functools import partial
import glog as logger
from copy import deepcopy
import errno
import gdown #https://github.com/wkentaro/gdown
import pyxis
import torch


def edges2A(edges, n_nodes, mode='P', sparse_mat_type=scipy.sparse.csr_matrix):
    '''
    note: assume no (i,i)-like edge
    edges: <2xE>
    '''
    edges = np.array(edges).astype(int)

    data_D = np.zeros(n_nodes, dtype=np.float32)
    for d in range(n_nodes):
        data_D[ d ] = len(np.where(edges[0] == d)[0])   # compute the number of node which pick node_i as their neighbor

    if mode.upper() == 'M':  # 'M' means max pooling, which use the same graph matrix as the adjacency matrix
        data = np.ones(edges[0].shape[0], dtype=np.int32)
    elif mode.upper() == 'P':
        data = 1. / data_D[ edges[0] ]
    else:
        raise NotImplementedError("edges2A with unknown mode=" + mode)

    return sparse_mat_type((data, edges), shape=(n_nodes, n_nodes))

def knn_search(data, knn=16, metric="euclidean", symmetric=True):
    """
    Args:
      data: Nx3
      knn: default=16
    """
    assert(knn>0)
    n_data_i = data.shape[0]
    kdt = KDTree(data, leaf_size=30, metric=metric)

    nbs = kdt.query(data, k=knn+1, return_distance=True)    # nbs[0]:NN distance,N*17. nbs[1]:NN index,N*17
    cov = np.zeros((n_data_i,9), dtype=np.float32)
    adjdict = dict()
    # wadj = np.zeros((n_data_i, n_data_i), dtype=np.float32)
    for i in range(n_data_i):
        # nbsd = nbs[0][i]
        nbsi = nbs[1][i]    #index i, N*17 YW comment
        cov[i] = np.cov(data[nbsi[1:]].T).reshape(-1) #compute local covariance matrix
        for j in range(knn):
            if symmetric:
                adjdict[(i, nbsi[j+1])] = 1
                adjdict[(nbsi[j+1], i)] = 1
                # wadj[i, nbsi[j + 1]] = 1.0 / nbsd[j + 1]
                # wadj[nbsi[j + 1], i] = 1.0 / nbsd[j + 1]
            else:
                adjdict[(i, nbsi[j+1])] = 1
                # wadj[i, nbsi[j + 1]] = 1.0 / nbsd[j + 1]
    edges = np.array(list(adjdict.keys()), dtype=int).T
    return edges, nbs[0], cov #, wadj

def build_graph_core(ith_datai, args):
    try:
        #ith, xyi = ith_datai #xyi: 2048x3
        xyi = ith_datai  # xyi: 2048x3
        n_data_i = xyi.shape[0]
        edges, nbsd, cov = knn_search(xyi, knn=args.knn, metric=args.metric)
        ith_graph = edges2A(edges, n_data_i, args.mode, sparse_mat_type=scipy.sparse.csr_matrix)
        nbsd=np.asarray(nbsd)[:, 1:]
        nbsd=np.reshape(nbsd, -1)

        #if ith % 500 == 0:
            #logger.info('{} processed: {}'.format(args.flag, ith))

        #return ith, ith_graph, nbsd, cov
        return ith_graph, nbsd, cov
    except KeyboardInterrupt:
        exit(-1)

def build_graph(points, args):      # points: batch, num of points, 3

    points = points.numpy()
    batch_graph = []
    Cov = torch.zeros(points.shape[0], args.num_points, 9)

    pool = multiproc.Pool(2)
    pool_func = partial(build_graph_core, args=args)
    rets = pool.map(pool_func, points)
    pool.close()
    count = 0
    for ret in rets:
        ith_graph, _, cov = ret
        batch_graph.append(ith_graph)
        Cov[count,:,:] = torch.from_numpy(cov)
        count = count+1
    del rets

    return batch_graph, Cov




if __name__ == '__main__':

    #   test YW
    parser = argparse.ArgumentParser(sys.argv[0])

    parser.add_argument('--pts_mn40', type=int, default=2048,
                        help="number of points per modelNet40 object")
    parser.add_argument('--pts_shapenet_part', type=int, default=2048,
                        help="number of points per shapenet_part object")
    parser.add_argument('--pts_shapenet', type=int, default=2048,
                        help="number of points per shapenet object")
    parser.add_argument('-md', '--mode', type=str, default="M",
                        help="mode used to compute graphs: M, P")
    parser.add_argument('-m', '--metric', type=str, default='euclidean',
                        help="metric for distance calculation (manhattan/euclidean)")
    parser.add_argument('--no-shuffle', dest='shuffle', action='store_false', default=True,
                        help="whether to shuffle data (1) or not (0) before saving")
    parser.add_argument('--regenerate', dest='regenerate', action='store_true', default=False,
                        help='regenerate from raw pointnet data or not (default: False)')

    args = parser.parse_args(sys.argv[1:])
    args.script_folder = os.path.dirname(os.path.abspath(__file__))

    args.knn = 16
    args.mode = 'M'
    args.batchSize = 10
    args.num_points = 2048
    args.metric = 'euclidean'
    sim_data = torch.rand(10,2048, 3)
    #ith, ith_graph, nbsd, cov = build_graph_core((0, sim_data), args)
    #ith_graph, nbsd, cov = build_graph_core(sim_data, args)

    batch_graph, Cov = build_graph(sim_data, args)

    print('done')
