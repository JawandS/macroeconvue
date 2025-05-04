"""
Federated Learning experiment: predict inflation from per-household transaction data.
Each household is treated as a client with its own local dataset.
Uses FedAvg over all clients with a simple neural network.
"""
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from copy import deepcopy

# --- Model Definition ---
class InflationPredictor(nn.Module):
    """
    Simple feed-forward network to predict inflation from transaction features.
    """
    def __init__(self, input_dim, hidden_dim=16):
        super(InflationPredictor, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )

    def forward(self, x):
        return self.net(x)

# --- Federated Learning Utilities ---

def local_train(model, data, targets, epochs=1, lr=0.01):
    """
    Perform local training on a client's data.
    Returns the updated state_dict.
    """
    model = deepcopy(model)
    model.train()
    optimizer = optim.SGD(model.parameters(), lr=lr)
    loss_fn = nn.MSELoss()
    for _ in range(epochs):
        optimizer.zero_grad()
        preds = model(data).squeeze()
        loss = loss_fn(preds, targets)
        loss.backward()
        optimizer.step()
    return model.state_dict()


def federated_average(state_dicts):
    """
    Averages a list of state_dicts (parameter dicts) from clients.
    """
    avg_dict = deepcopy(state_dicts[0])
    for key in avg_dict.keys():
        for sd in state_dicts[1:]:
            avg_dict[key] += sd[key]
        avg_dict[key] = avg_dict[key] / len(state_dicts)
    return avg_dict

# --- Main Experiment ---
if __name__ == "__main__":
    # Load simulation data
    df = pd.read_csv('simulation_transactions.csv')
    # Identify feature columns and target
    feature_cols = [c for c in df.columns if c.startswith('hh_')]
    target_col = 'inflation'

    # Prepare client datasets
    client_ids = set(col.split('_')[1] for col in feature_cols)
    clients = {}
    for hid in client_ids:
        # select all categories for this household
        cols = [c for c in feature_cols if c.startswith(f'hh_{hid}_')]
        X = torch.tensor(df[cols].values, dtype=torch.float32)
        y = torch.tensor(df[target_col].values, dtype=torch.float32)
        clients[hid] = (X, y)

    # Initialize global model
    input_dim = len([c for c in feature_cols if c.startswith(f'hh_{list(client_ids)[0]}_')])
    global_model = InflationPredictor(input_dim=input_dim)

    # Federated training settings
    rounds = 10
    local_epochs = 1
    lr = 0.01

    for r in range(rounds):
        client_states = []
        for hid, (X, y) in clients.items():
            sd = local_train(global_model, X, y, epochs=local_epochs, lr=lr)
            client_states.append(sd)
        # Aggregate
        avg_state = federated_average(client_states)
        global_model.load_state_dict(avg_state)
        # Optionally evaluate on aggregated data
        global_model.eval()
        with torch.no_grad():
            all_X = torch.cat([c[0] for c in clients.values()], dim=0)
            all_y = torch.cat([c[1] for c in clients.values()], dim=0)
            preds = global_model(all_X).squeeze()
            mse = nn.MSELoss()(preds, all_y).item()
        print(f"Round {r+1}/{rounds} — Global MSE: {mse:.6f}")

    print("Federated learning experiment complete.")
