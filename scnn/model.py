#!/usr/bin/env python3

import torch
import torch.nn as nn
import torch.nn.functional
import torch.utils.data as data
import numpy as np
import sys

import sys
sys.path.append('.')
import scnn.scnn
import scnn.chebyshev


import torch
import torch.nn as nn

class MySCNN(nn.Module):
    def __init__(self, colors=1):
        super(MySCNN, self).__init__()

        assert(colors > 0)
        self.colors = colors

        num_filters = 10  # 30
        variance = 0.01  # 0.001

        # Degree 0 convolutions.
        self.C0_1 = scnn.scnn.SimplicialConvolution(5, self.colors, num_filters*self.colors, variance=variance)
        self.C0_2 = scnn.scnn.SimplicialConvolution(5, num_filters*self.colors, num_filters*self.colors, variance=variance)
        self.C0_3 = scnn.scnn.SimplicialConvolution(5, num_filters*self.colors, self.colors, variance=variance)

        # Degree 1 convolutions.
        self.C1_1 = scnn.scnn.SimplicialConvolution(5, self.colors, num_filters*self.colors, variance=variance)
        self.C1_2 = scnn.scnn.SimplicialConvolution(5, num_filters*self.colors, num_filters*self.colors, variance=variance)
        self.C1_3 = scnn.scnn.SimplicialConvolution(5, num_filters*self.colors, self.colors, variance=variance)

        # Degree 2 convolutions.
        self.C2_1 = scnn.scnn.SimplicialConvolution(5, self.colors, num_filters*self.colors, variance=variance)
        self.C2_2 = scnn.scnn.SimplicialConvolution(5, num_filters*self.colors, num_filters*self.colors, variance=variance)
        self.C2_3 = scnn.scnn.SimplicialConvolution(5, num_filters*self.colors, self.colors, variance=variance)

        # Binary classification layer
        self.fc = nn.Linear(3 * self.colors, 2)  # Concatenated out0_3, out1_3, out2_3
        self.sigmoid = nn.Sigmoid()

    def forward(self, Ls, Ds, adDs, xs):
        assert(len(xs) == 3)  # The three degrees are fed together as a list.

        out0_1 = self.C0_1(Ls[0], xs[0])
        out1_1 = self.C1_1(Ls[1], xs[1])
        out2_1 = self.C2_1(Ls[2], xs[2])

        out0_2 = self.C0_2(Ls[0], nn.LeakyReLU()(out0_1))
        out1_2 = self.C1_2(Ls[1], nn.LeakyReLU()(out1_1))
        out2_2 = self.C2_2(Ls[2], nn.LeakyReLU()(out2_1))

        out0_3 = self.C0_3(Ls[0], nn.LeakyReLU()(out0_2))
        out1_3 = self.C1_3(Ls[1], nn.LeakyReLU()(out1_2))
        out2_3 = self.C2_3(Ls[2], nn.LeakyReLU()(out2_2))

        # Concatenate the three outputs
        concatenated_out = torch.cat([out0_3, out1_3, out2_3], dim=1)

        # Pass through the classifier
        logits = self.fc(concatenated_out)
        probs = self.sigmoid(logits)

        return probs  # Return the binary classification probabilities


def main():
    torch.manual_seed(1337)
    np.random.seed(1337)


    prefix = sys.argv[1] ##input

    logdir = sys.argv[2] ##output
    starting_node=sys.argv[3]
    percentage_missing_values=sys.argv[4]
    cuda = False

    topdim = 2


    laplacians = np.load('scnn/bounds_and_laps/laplacians.npz'.format(prefix, starting_node),allow_pickle=True)
    boundaries = np.load('scnn/bounds_and_laps/boundaries.npz'.format(prefix,starting_node),allow_pickle=True)
    data = np.load('scnn/bounds_and_laps/labels.npy'.format(prefix),allow_pickle=True)
    # Assuming arrays are stored as arr_0, arr_1, ..., extract them
    labels = []
    for i in range(10000):  # Modify this if you have more/less arrays
        labels.append(data[f'arr_{i}'])

    # Convert binary labels [0, 1] -> 0 (quarks), [1, 0] -> 1 (gluons)
    formatted_labels = [1 if label[0] == 1 else 0 for label in labels]

    # Convert to a torch tensor
    labels_tensor = torch.tensor(formatted_labels, dtype=torch.int)


    Ls =[scnn.scnn.coo2tensor(scnn.chebyshev.normalize(laplacians[i],half_interval=True)) for i in range(topdim+1)] #####scnn.chebyshev.normalize ?
    Ds=[scnn.scnn.coo2tensor(boundaries[i].transpose()) for i in range(topdim+1)]
    adDs=[scnn.scnn.coo2tensor(boundaries[i]) for i in range(topdim+1)]


    criterion = nn.BCELoss()  # Binary Cross Entropy loss

    num_epochs = 10

    network = MySCNN(colors = 1)


    learning_rate = 0.001
    optimizer = torch.optim.Adam(network.parameters(), lr=learning_rate)
    criterion = nn.L1Loss(reduction="sum")
    #criterion = nn.MSELoss(reduction="sum"
    for i in range(num_epochs):
        optimizer.zero_grad()

        # Get predictions from the network
        probs = network(Ls, Ds, adDs, xs)

        # Compute loss between predictions and true labels (binary_labels)
        loss = criterion(probs, labels_tensor)

        # Backpropagation and optimization step
        loss.backward()
        optimizer.step()

        # Print loss every few iterations
        if i % 10 == 0:
            print(f"Epoch {i}/{num_epochs}, Loss: {loss.item()}")



if __name__ == "__main__":
    main()