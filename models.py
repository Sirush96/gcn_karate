import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, SAGEConv, GATv2Conv
import torch

# GCN model with 2 layers, changing hidden number of layers to 32
class GCN(nn.Module): # transformation to each node's features and aggregates information from its neighboring nodes to produce a new feature representation.
    def __init__(self, data):
        super(GCN, self).__init__()

        self.data = data # stores the input data as a class attribute.

        self.conv1 = GCNConv(self.data.num_features, 32)  # initializes the first graph convolutional layer with GCNConv module, which takes in the number of input features (i.e., the number of node features in the graph) and outputs 32 features
        self.conv2 = GCNConv(32, int(self.data.num_classes)) # the second graph convolutional layer, which takes in 32 features as input and outputs a number of classes equal to the number of classes in the dataset

    def forward(self):
        x, edge_index = self.data.x, self.data.edge_index # extracts the node feature matrix x and the edge index matrix edge_index from the input data
        x = F.relu(self.conv1(x, edge_index)) # applies the first graph convolutional layer to the input node feature matrix x and the edge index matrix edge_index, followed by a ReLU activation function
        x = F.dropout(x, training=self.training) # applies dropout regularization to the output of the first graph convolutional layer during training, in order to prevent overfitting
        x = self.conv2(x, edge_index) # second convolutional layer
        return F.log_softmax(x, dim=1) # applies a log-softmax function to the output of the second graph convolutional layer and returns it
    # The dim=1 argument specifies that the softmax should be applied across the class dimension of the output matrix. This output can be used to compute the loss function and make predictions during training and evaluation


class GraphSAGE(nn.Module): # instead of just computing the mean of the neighboring nodes' features, it learns a more complex function to aggregate the information. The model then applies a non-linear activation function, specifically elu, to the transformed features before outputting the final prediction
    def __init__(self, data, dropout=0.2):
        super(GraphSAGE, self).__init__()

        self.data = data
        in_dim = self.data.num_features # stores the number of input features (i.e., the number of node features in the graph) as a local variable
        hidden_dim = 16 # sets the size of the hidden dimension to 16
        out_dim = self.data.num_classes

        self.dropout = dropout
        self.conv1 = SAGEConv(in_dim, hidden_dim) # initializes the first GraphSAGE convolutional layer with SAGEConv module, which takes in the number of input features (i.e., the number of node features in the graph) and outputs hidden_dim features
        self.conv2 = SAGEConv(hidden_dim, hidden_dim)
        self.conv3 = SAGEConv(hidden_dim, out_dim)

    def forward(self):
        x = self.conv1(self.data.x, self.data.edge_index) # This line applies the first graph convolution layer to the node feature matrix self.data.x and the graph edge index self.data.edge_index. The output is assigned to the variable x
        x = F.elu(x) # applies the Exponential Linear Unit (ELU) activation function to the output x of the first convolution layer
        x = F.dropout(x, p=self.dropout) # applies dropout regularization to the output x with the dropout probability

        x = self.conv2(x, self.data.edge_index)
        x = F.elu(x)
        x = F.dropout(x, p=self.dropout)

        x = self.conv3(x, self.data.edge_index)
        x = F.elu(x)
        x = F.dropout(x, p=self.dropout)
        return torch.log_softmax(x, dim=-1)

class GAT(nn.Module): # uses a different approach called attention mechanism to weight each neighbor's contribution to the node's new feature representation
    def __init__(self, data, heads=8):
        super(GAT, self).__init__()

        self.data = data
        in_dim = self.data.num_features
        hidden_dim = 16
        out_dim = self.data.num_classes

        self.gat1 = GATv2Conv(in_dim, hidden_dim, heads=heads) # uses heads attention heads to compute node embeddings
        self.gat2 = GATv2Conv(hidden_dim * heads, out_dim, heads=1) # takes as input the output of the first layer, which has a dimension of hidden_dim * heads, and produces the final output of dimension out_dim. It uses a single attention head
        self.optimizer = torch.optim.Adam(self.parameters(),
                                          lr=0.005,
                                          weight_decay=5e-4) # weight_decay is added to the original loss to produce the final loss value to prevent overfitting

    def forward(self):

        h = F.dropout(self.data.x, p=0.6, training=self.training) # passed through a dropout layer with a probability of 0.6
        h = self.gat1(h, self.data.edge_index)
        h = F.elu(h)
        h = F.dropout(h, p=0.6, training=self.training)
        h = self.gat2(h, self.data.edge_index)
        return F.log_softmax(h, dim=1)
