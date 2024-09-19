#!/usr/bin/env python3

import torch
import torch.nn as nn
import torch.nn.functional
import torch.utils.data as data
import numpy as np
import sys

import sys
sys.path.append('.')
import scnn
import chebyshev


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
        self.C0_1 = scnn.SimplicialConvolution(5, C_in = 4, C_out = 16, variance=variance)
        self.C0_2 = scnn.SimplicialConvolution(5, C_in = 16, C_out = 16, variance=variance)
        self.C0_3 = scnn.SimplicialConvolution(5, C_in = 16, C_out = 4, variance=variance)

        # Degree 1 convolutions.
        self.C1_1 = scnn.SimplicialConvolution(5, C_in = 4, C_out = 16, variance=variance)
        self.C1_2 = scnn.SimplicialConvolution(5, C_in = 16, C_out = 16, variance=variance)
        self.C1_3 = scnn.SimplicialConvolution(5, C_in = 16, C_out = 4, variance=variance)

        # Degree 2 convolutions.
        self.C2_1 = scnn.SimplicialConvolution(5, C_in = 4, C_out = 16, variance=variance)
        self.C2_2 = scnn.SimplicialConvolution(5, C_in = 16, C_out = 16, variance=variance)
        self.C2_3 = scnn.SimplicialConvolution(5, C_in = 16, C_out = 4, variance=variance)

        # Binary classification layer
        self.fc = nn.Linear(3 * 4, 2)  # Concatenated out0_3, out1_3, out2_3
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

        xs_update = [out0_3, out1_3, out2_3]
        out0_3_agg = torch.sum(out0_3, dim=2, keepdim=True)
        out1_3_agg = torch.sum(out1_3, dim=2, keepdim=True)
        out2_3_agg = torch.sum(out2_3, dim=2, keepdim=True)

        concatenated_out = torch.cat([out0_3_agg, out1_3_agg, out2_3_agg], dim=1)
        # print(concatenated_out.shape)
        concatenated_out = torch.flatten(concatenated_out, start_dim=1)
        # print(concatenated_out.shape)
        # input()
        # Pass through the classifier
        logits = self.fc(concatenated_out)
        probs = self.sigmoid(logits)
        # print(probs.shape)
        # print(probs)
        # input()
        return logits, xs_update  # Return the binary classification probabilities


def main():
    torch.manual_seed(1337)
    np.random.seed(1337)


#   prefix = sys.argv[1] ##input

#    logdir = sys.argv[2] ##output
#    starting_node=sys.argv[3]
#    percentage_missing_values=sys.argv[4]
#    cuda = False



    all_lapl = []
    all_bounds = []
    labels = []
    laplacians = np.load('bounds_and_laps/laplacians.npz',allow_pickle=True)
    boundaries = np.load('bounds_and_laps/boundaries.npz',allow_pickle=True)
    data = np.load('bounds_and_laps/labels.npz',allow_pickle=True)


    for i in range(len(laplacians)):
        all_lapl.append(laplacians[f'arr_{i}'])
        all_bounds.append(boundaries[f'arr_{i}'])
        labels.append(data[f'arr_{i}'])
    del laplacians, boundaries, data

    # print(len(all_bounds[511]))
    # print(len(all_lapl[511]))
    # input()
    # print(all_lapl[511][0].shape)
    # print(all_lapl[511][1].shape)
    # print(all_lapl[511][2].shape)
    #
    # print(all_bounds[511][0].shape)
    # print(all_bounds[511][1].shape)
    # # print(all_bounds[511][2].shape)
    # input()

    # Convert binary labels [0, 1] -> 0 (quarks), [1, 0] -> 1 (gluons)
    formatted_labels = [1 if label[0] == 1 else 0 for label in labels]

    # Convert to a torch tensor
    # labels_tensor = torch.tensor(formatted_labels, dtype=torch.int)
    # print(labels_tensor.shape)
    # input()
    batch_size = 1
    xs = []
    Ls_all = []
    Ds_all = []
    adDs_all = []
    for i in range(len(all_lapl)):
        lap = all_lapl[i]
        boundary = all_bounds[i]
        # print(i)
        num_nodes = all_lapl[i][0].shape[0] # Number of nodes (0-simplices)
        # print(num_nodes)
        if len(all_bounds[i]) == 2:
            topdim = 2
            num_edges = all_bounds[i][0].shape[1]  # Number of edges (1-simplices)
            # print(num_edges)
            num_faces = all_bounds[i][1].shape[1]  # Number of faces (2-simplices)

            xs_temp = [
                torch.rand((batch_size, 4, num_nodes)),  # Degree 0 input (nodes)
                torch.rand((batch_size, 4, num_edges)),  # Degree 1 input (edges)
                torch.rand((batch_size, 4, num_faces))  # Degree 2 input (faces)
            ]

            Ls = [scnn.coo2tensor(lap[k]) for k in
                  range(topdim + 1)]  #####scnn.chebyshev.normalize ?
            Ds = [scnn.coo2tensor(boundary[k].transpose()) for k in range(topdim)]
            adDs = [scnn.coo2tensor(boundary[k]) for k in range(topdim)]

        elif len(all_bounds[i]) == 1:
            topdim = 1
            num_edges = all_bounds[i][0].shape[1]
            num_faces = 1 # No faces exist in the chosen filtration
            xs_temp = [
                torch.rand((batch_size, 4, num_nodes)),
                torch.rand((batch_size, 4, num_edges)),
                torch.zeros((batch_size, 4, num_faces))
            ]
            Ls = [scnn.coo2tensor(scnn.chebyshev.normalize(lap[k], half_interval=True)) for k in
                  range(topdim + 1)]  #####scnn.chebyshev.normalize ?
            Ds = [scnn.coo2tensor(boundary[k].transpose()) for k in range(topdim)]
            adDs = [scnn.coo2tensor(boundary[k]) for k in range(topdim)]


        xs.append(xs_temp)
        Ls_all.append(Ls)
        Ds_all.append(Ds)
        adDs_all.append(adDs)




    criterion = nn.BCELoss()  # Binary Cross Entropy loss

    num_epochs = 10

    network = MySCNN(colors = 1)


    learning_rate = 0.001
    optimizer = torch.optim.Adam(network.parameters(), lr=learning_rate)
    criterion = nn.BCELoss()
    #criterion = nn.MSELoss(reduction="sum"
    for i in range(num_epochs):
        for j in range(len(xs)):
            print(j)
            laplacians = all_lapl[j]
            boundaries = all_bounds[j]
            Ls = Ls_all[j]
            Ds = Ds_all[j]
            adDs = adDs_all[j]

            xs_i = xs[j]
            # print(xs_i[0].shape)
            # print(xs_i[1].shape)
            # print(xs_i[2].shape)
            # input()
            labels_grnd = torch.tensor(labels[j], dtype = torch.float)
            labels_grnd = labels_grnd.unsqueeze(0)
            # print(labels_grnd.size())
            # input()
            optimizer.zero_grad()

            # Ls = [scnn.coo2tensor(scnn.chebyshev.normalize(laplacians[k], half_interval=True)) for k in
            #       range(topdim + 1)]  #####scnn.chebyshev.normalize ?
            # Ds = [scnn.coo2tensor(boundaries[k].transpose()) for k in range(topdim)]
            # adDs = [scnn.coo2tensor(boundaries[k]) for k in range(topdim)]

            # Get predictions from the network
            probs, xs_temp = network(Ls, Ds, adDs, xs_i)
            # print(xs[j][0].shape)
            # print(xs[j][1].shape)
            # print(xs[j][2].shape)
            # input()
            # Compute loss between predictions and true labels (binary_labels)
            loss = criterion(probs, labels_grnd)

            # Backpropagation and optimization step
            loss.backward(retain_graph=True)
            for param in network.parameters():
                if param.grad is not None:
                    print(param.grad.abs().mean())
            optimizer.step()
            # updated_xs = []
            # updated_xs.append(xs_temp)

        # After the inner loop, update xs outside the gradient calculation
        # xs = updated_xs
        # del updated_xs
        # Print loss every few iterations
        if i % 1 == 0:
            print(f"Epoch {i}/{num_epochs}, Loss: {loss.item()}")



if __name__ == "__main__":
    main()