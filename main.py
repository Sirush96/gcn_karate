from networkx import algorithms, from_pandas_edgelist, to_pandas_adjacency, from_pandas_adjacency, to_pandas_edgelist, \
    from_numpy_array, to_numpy_array


import torch
from datasets.datasets import KarateDataset
import models
import utils
#from models.models import Net, GraphSAGE
#from models.GraphSAGE_basic import GraphSAGE
#import torch.nn.functional as F

import argparse

class Main():
    def __init__(self, train_config, debug=False):
        self.train_config = train_config

        torch.manual_seed(42)
        self.optimizer_name = "Adam"
        self.lr = 1e-1
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        self.dataset = KarateDataset()
        self.data = self.dataset[0]

        self.epochs = self.train_config['epochs']
        self.model_name = self.train_config['model']
        self.model = getattr(models, self.model_name)(self.data).to(self.device)
        self.optimizer = getattr(torch.optim, self.optimizer_name)(self.model.parameters(), lr=self.lr)

    def train(self):
        loss_operator = torch.nn.NLLLoss()
        self.model.train()
        for epoch in range(1, self.epochs):
            self.optimizer.zero_grad()
            loss = loss_operator(self.model()[self.data.train_mask], self.data.y[self.data.train_mask])
            acc = utils.acc_operator(self.model()[self.data.train_mask].argmax(dim=1), self.data.y[self.data.train_mask])
            loss.backward()
            self.optimizer.step()

            # Validation
            val_loss = loss_operator(self.model()[self.data.val_mask], self.data.y[self.data.val_mask])
            val_acc = utils.acc_operator(self.model()[self.data.val_mask].argmax(dim=1), self.data.y[self.data.val_mask])

            # Print metrics every 10 epochs
            if(epoch % 10 == 0):
                print(f'Epoch {epoch:>3} | Train Loss: {loss:.3f} | Train Acc: '
                      f'{acc*100:>6.2f}% | Val Loss: {val_loss:.2f} | '
                      f'Val Acc: {val_acc*100:.2f}%')

    @torch.no_grad()
    def test(self):
        self.model.eval()
        logits = self.model() # performs a forward pass of the model on the input data, which generates predictions for each node in the graph
        mask1 = self.data['train_mask']
        pred1 = logits[mask1].max(1)[1] # predictions for train nodes, logits[mask1] selects the predictions for the training nodes, max(1) returns the maximum value and [1] selects the indices, which represent the predicted classes
        acc1 = pred1.eq(self.data.y[mask1]).sum().item() / mask1.sum().item() # pred1.eq(self.data.y[mask1]) compares the predicted classes with the true labels
        mask = self.data['test_mask']
        pred = logits[mask].max(1)[1]
        acc = pred.eq(self.data.y[mask]).sum().item() / mask.sum().item()
        return acc1, acc

    def run(self):
        self.train()
        train_acc, test_acc = self.test()
        print('#' * 70)
        print('Train Accuracy: %s' % train_acc)
        print('Test Accuracy: %s' % test_acc)
        print('#' * 70)


if __name__ == "__main__": # condition is used to ensure that the code inside the block is only executed when the script is run as the main program and not imported as a module

    parser = argparse.ArgumentParser()

    parser.add_argument('-epochs', help='train epoch', type = int, default=800)
    parser.add_argument('-model', help='training model', type = str, default='GraphSAGE')

    args = parser.parse_args()

    train_config = {
        'epochs': args.epochs,
        'model': args.model
    }


    main = Main(train_config, debug=False)
    main.run()