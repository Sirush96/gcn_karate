import networkx as nx
import scipy.sparse as sp
from scipy.sparse import coo_matrix
import numpy as np
import torch
from sklearn.preprocessing import StandardScaler

def returnEmbeddings():
    # load graph from networkx library

    G = nx.karate_club_graph()

    # retrieve the labels for each node
    labels = np.asarray([G.nodes[i]['club'] != 'Mr. Hi' for i in G.nodes]).astype(np.int64)

    # create edge index from
    adj = sp.coo_matrix(nx.to_scipy_sparse_array(G)) # I`ve changed the function to this, so that I can use the new version of networkx
    row = torch.from_numpy(adj.row.astype(np.int64)).to(torch.long)
    col = torch.from_numpy(adj.col.astype(np.int64)).to(torch.long)
    edge_index = torch.stack([row, col], dim=0)

    # using degree as embedding
    first_degree = G.degree()
    dict_first_degree = dict(first_degree)
    values_dict_first_degree = dict_first_degree.values()
    list_values_dict_first_degree = list(values_dict_first_degree)
    np_array_list_values_dict_first_degree = np.array(list_values_dict_first_degree)
    embeddings = np.array(list(dict(G.degree()).values()))

    # normalizing degree values
    scale = StandardScaler()
    embeddings_reshaped = embeddings.reshape(-1, 1)
    embeddings = scale.fit_transform(embeddings_reshaped)

    # let us also create the adjacency matrix
    adj_t = nx.to_numpy_array(G)

    return G, labels, edge_index, embeddings, adj_t


def acc_operator(pred_y, y):
    """Calculate accuracy."""
    return ((pred_y == y).sum() / len(y)).item()
