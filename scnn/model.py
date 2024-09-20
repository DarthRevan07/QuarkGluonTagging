import os

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import scnn
import chebyshev
import logging

class MySCNN(nn.Module):
    def __init__(self, colors=1):
        super(MySCNN, self).__init__()

        assert(colors > 0)
        self.colors = colors

        num_filters = 10  # 30
        variance = 0.01  # 0.001

        # Degree 0 convolutions.
        self.C0_1 = scnn.SimplicialConvolution(5, C_in=4, C_out=16, variance=variance)
        self.C0_2 = scnn.SimplicialConvolution(5, C_in=16, C_out=16, variance=variance)
        self.C0_3 = scnn.SimplicialConvolution(5, C_in=16, C_out=4, variance=variance)

        # Degree 1 convolutions.
        self.C1_1 = scnn.SimplicialConvolution(5, C_in=4, C_out=16, variance=variance)
        self.C1_2 = scnn.SimplicialConvolution(5, C_in=16, C_out=16, variance=variance)
        self.C1_3 = scnn.SimplicialConvolution(5, C_in=16, C_out=4, variance=variance)

        # Degree 2 convolutions.
        self.C2_1 = scnn.SimplicialConvolution(5, C_in=4, C_out=16, variance=variance)
        self.C2_2 = scnn.SimplicialConvolution(5, C_in=16, C_out=16, variance=variance)
        self.C2_3 = scnn.SimplicialConvolution(5, C_in=16, C_out=4, variance=variance)

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
        out0_3_agg = torch.mean(out0_3, dim=2, keepdim=True)
        out1_3_agg = torch.mean(out1_3, dim=2, keepdim=True)
        out2_3_agg = torch.mean(out2_3, dim=2, keepdim=True)

        concatenated_out = torch.cat([out0_3_agg, out1_3_agg, out2_3_agg], dim=1)
        concatenated_out = torch.flatten(concatenated_out, start_dim=1)

        logits = self.fc(concatenated_out)
        probs = self.sigmoid(logits)
        return probs, xs_update  # Return the binary classification probabilities


def load_data(lapl_file, boundary_file, label_file):
    laplacians = np.load(lapl_file, allow_pickle=True)
    boundaries = np.load(boundary_file, allow_pickle=True)
    labels = np.load(label_file, allow_pickle=True)

    all_lapl = [laplacians[f'arr_{i}'] for i in range(len(laplacians))]
    all_bounds = [boundaries[f'arr_{i}'] for i in range(len(boundaries))]
    all_labels = [labels[f'arr_{i}'] for i in range(len(labels))]
    # formatted_labels = [1 if label[0] == 1 else 0 for label in all_labels]

    del laplacians, boundaries, labels
    return all_lapl, all_bounds, all_labels


def prepare_inputs(all_lapl, all_bounds, batch_size):
    xs, Ls_all, Ds_all, adDs_all = [], [], [], []

    for i in range(len(all_lapl)):
        lap = all_lapl[i]
        boundary = all_bounds[i]
        num_nodes = all_lapl[i][0].shape[0]  # Number of nodes (0-simplices)

        if len(all_bounds[i]) == 2:
            topdim = 2
            num_edges = all_bounds[i][0].shape[1]  # Number of edges (1-simplices)
            num_faces = all_bounds[i][1].shape[1]  # Number of faces (2-simplices)

            xs_temp = [
                torch.rand((batch_size, 4, num_nodes)),  # Degree 0 input (nodes)
                torch.rand((batch_size, 4, num_edges)),  # Degree 1 input (edges)
                torch.rand((batch_size, 4, num_faces))   # Degree 2 input (faces)
            ]
            # print(lap.shape)
            # input()
            Ls = [scnn.coo2tensor(chebyshev.normalize(lap[k])) for k in range(topdim + 1)]
            Ds = [scnn.coo2tensor(boundary[k].transpose()) for k in range(topdim)]
            adDs = [scnn.coo2tensor(boundary[k]) for k in range(topdim)]

        elif len(all_bounds[i]) == 1:
            topdim = 1
            num_edges = all_bounds[i][0].shape[1]
            num_faces = 1  # No faces exist in the chosen filtration

            xs_temp = [
                torch.rand((batch_size, 4, num_nodes)),
                torch.rand((batch_size, 4, num_edges)),
                torch.zeros((batch_size, 4, num_faces))
            ]

            Ls = [scnn.coo2tensor(scnn.chebyshev.normalize(lap[k])) for k in range(topdim + 1)]
            Ds = [scnn.coo2tensor(boundary[k].transpose()) for k in range(topdim)]
            adDs = [scnn.coo2tensor(boundary[k]) for k in range(topdim)]

        xs.append(xs_temp)
        Ls_all.append(Ls)
        Ds_all.append(Ds)
        adDs_all.append(adDs)

    return xs, Ls_all, Ds_all, adDs_all


def compute_accuracy(probs, labels):
    preds = (probs > 0.5).float()
    correct = (preds == labels).float().sum()
    return correct / len(labels)


def main():
    torch.manual_seed(1337)
    np.random.seed(1337)
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

    batch_size = 1
    num_epochs = 10
    learning_rate = 0.001

    os.chdir(os.path.abspath('bounds_and_laps/'))
    # Load training data
    train_laplacians, train_boundaries, train_labels = load_data('train_laplacians.npz', 'train_boundaries.npz', 'train_labels.npz')
    train_xs, train_Ls, train_Ds, train_adDs = prepare_inputs(train_laplacians, train_boundaries, batch_size)

    # Load test data
    test_laplacians, test_boundaries, test_labels = load_data('test_laplacians.npz', 'test_boundaries.npz', 'test_labels.npz')
    test_xs, test_Ls, test_Ds, test_adDs = prepare_inputs(test_laplacians, test_boundaries, batch_size)

    network = MySCNN(colors=1)
    optimizer = torch.optim.Adam(network.parameters(), lr=learning_rate)
    criterion = nn.BCELoss()

    stats = {'train_acc': [], 'test_acc': []}

    for epoch in range(num_epochs):
        train_loss = 0.0
        train_correct = 0.0
        test_loss = 0.0
        test_correct = 0.0
        network.train()

        # Training loop
        for j in range(len(train_xs)):
            optimizer.zero_grad()
            labels_grnd = torch.tensor(train_labels[j], dtype=torch.float).unsqueeze(0)
            # print(labels_grnd.shape)
            # labels_grnd = labels_grnd.unsqueeze(0)
            # print(labels_grnd.shape)
            # input()
            probs, _ = network(train_Ls[j], train_Ds[j], train_adDs[j], train_xs[j])

            loss = criterion(probs, labels_grnd)
            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            train_correct += compute_accuracy(probs, labels_grnd).item()

            if j % 100 == 0 :
                logging.info(f"Processed event - {j}")

        # Evaluation on test set
        network.eval()
        with torch.no_grad():
            for j in range(len(test_xs)):
                labels_test = torch.tensor(test_labels[j], dtype=torch.float).unsqueeze(0)
                probs, _ = network(test_Ls[j], test_Ds[j], test_adDs[j], test_xs[j])

                loss = criterion(probs, labels_test)
                test_loss += loss.item()
                test_correct += compute_accuracy(probs, labels_test).item()

        # Print epoch statistics
        print(f'Epoch {epoch + 1}/{num_epochs}, Train Loss: {train_loss / len(train_xs):.4f}, '
              f'Train Accuracy: {train_correct / len(train_xs):.4f}, Test Loss: {test_loss / len(test_xs):.4f}, '
              f'Test Accuracy: {test_correct / len(test_xs):.4f}')



if __name__ == "__main__":
    main()