import torch
import pandas as pd
from torch_geometric.data import InMemoryDataset, Data
from sklearn.model_selection import train_test_split
from tqdm import tqdm
from utils import returnEmbeddings
from sklearn.preprocessing import LabelEncoder

# custom dataset
class KarateDataset(InMemoryDataset): # to handle multiple graphs, each Data object corresponds to a different graph


    def __init__(self, transform=None):
        super(KarateDataset, self).__init__('.', transform, None , None) #This calls the __init__ method of the InMemoryDataset class, which this class inherits from

        # # This calls the returnEmbeddings function which returns five variables.
        # G - the graph, labels - node labels, adj_t - transposed adjacency matrix, edge_index is a tensor that contains the edge connectivity information of the graph
        # These variables are then unpacked into the corresponding variables using tuple unpacking.
        G, labels, edge_index, embeddings, adj_t = returnEmbeddings()

        # generate data for the dataset
        # Data is a PyTorch Geometric class that represents a single graph
        data = Data(edge_index=edge_index)
        data.num_nodes = G.number_of_nodes() # specifies the number of nodes in the graph

        # embeddings
        # assigns the node embeddings to the x attribute of the Data object
        data.x = torch.from_numpy(embeddings).type(torch.float32)

        # labels
        # creates a PyTorch tensor y from the labels array, which contains the class labels of the nodes
        # The tensor is cast to long data type to represent integer labels
        y = torch.from_numpy(labels).type(torch.long)
        data.y = y.clone().detach() #creates a copy of the y tensor, and the ensures that the copy is independent of the original tensor and does not track gradients.

        data.num_classes = 2 #sets the number of classes in the dataset

        # specify the ratios of the dataset to use for training, validation, and testing
        train_ratio = 0.75
        validation_ratio = 0.15
        test_ratio = 0.10

        # splitting the data into train, validation and test
        X_train, X_test, y_train, y_test = train_test_split(pd.Series(list(G.nodes())), # the list of nodes in the graph as a pandas series
                                                            pd.Series(labels), # the corresponding labels for each node in the graph as a pandas series
                                                            test_size= 1 - train_ratio,
                                                            random_state=42)

        X_train, X_val, y_train, y_val = train_test_split(pd.Series(list(G.nodes())),
                                                            pd.Series(labels),
                                                            test_size=test_ratio/(test_ratio + validation_ratio),
                                                            random_state=42)

        n_nodes = G.number_of_nodes()


        # create train and test masks for data
        train_mask = torch.zeros(n_nodes, dtype=torch.bool) # create a boolean tensor of shape (n_nodes)
        test_mask = torch.zeros(n_nodes, dtype=torch.bool)
        val_mask = torch.zeros(n_nodes, dtype=torch.bool)
        train_mask[X_train.index] = True # set the indices of the nodes in each subset to True, so that when we feed the graph into the model, the model knows which nodes belong to which subset
        test_mask[X_test.index] = True
        val_mask[X_val.index] = True
        data['train_mask'] = train_mask # adds the train_mask tensor to the data dictionary with key train_mask
        data['test_mask'] = test_mask
        data['val_mask'] = val_mask

        self.data, self.slices = self.collate([data]) # returns that Data object and a corresponding slices object

    def _download(self):
        return

    def _process(self):
        return

    def __repr__(self): # returns a string that contains the name of the class
        return '{}()'.format(self.__class__.__name__)


class YooChooseBinaryDataset(InMemoryDataset):
    def __init__(self, root, transform=None, pre_transform=None):
        super(YooChooseBinaryDataset, self).__init__(root, transform, pre_transform)
        self.data, self.slices = torch.load(self.processed_paths[0])

    @property
    def raw_file_names(self):
        return []

    @property
    def processed_file_names(self):
        return ['../input/yoochoose_click_binary_1M_sess.dataset']

    def download(self):
        pass

    def process(self):
        data_list = []

        # process by session_id
        grouped = df.groupby('session_id')
        for session_id, group in tqdm(grouped):
            sess_item_id = LabelEncoder().fit_transform(group.item_id)
            group = group.reset_index(drop=True)
            group['sess_item_id'] = sess_item_id
            node_features = group.loc[group.session_id == session_id, ['sess_item_id', 'item_id']].sort_values(
                'sess_item_id').item_id.drop_duplicates().values

            node_features = torch.LongTensor(node_features).unsqueeze(1)
            target_nodes = group.sess_item_id.values[1:]
            source_nodes = group.sess_item_id.values[:-1]

            edge_index = torch.tensor([source_nodes,
                                       target_nodes], dtype=torch.long)
            x = node_features

            y = torch.FloatTensor([group.label.values[0]])

            data = Data(x=x, edge_index=edge_index, y=y)
            data_list.append(data)

        data, slices = self.collate(data_list)
        torch.save((data, slices), self.processed_paths[0])