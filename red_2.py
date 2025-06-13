import torch
import numpy as np
from torch import nn
from torch.utils.data import TensorDataset, DataLoader
from tqdm import tqdm


num_pairs = 30000
X_pairs = np.random.randint(0, 100, size=(num_pairs, 2)).astype('float32')
y_pairs = (X_pairs[:, 0] > X_pairs[:, 1]).astype('float32').reshape(-1, 1)

# Convertir a tensores
X = torch.from_numpy(X_pairs)
y = torch.from_numpy(y_pairs)

# DataLoader
dataset = TensorDataset(X, y)
loader  = DataLoader(dataset, batch_size=64, shuffle=True)


class ComparatorNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(2, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.net(x)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model  = ComparatorNet().to(device)
opt    = torch.optim.Adam(model.parameters(), lr=1e-3)
crit   = nn.BCELoss()


for epoch in range(10):
    model.train()
    total_loss = 0.0
    loop = tqdm(loader, desc=f"Epoch {epoch+1}/10")
    for xb, yb in loop:
        xb, yb = xb.to(device), yb.to(device)
        pred   = model(xb)
        loss   = crit(pred, yb)
        opt.zero_grad()
        loss.backward()
        opt.step()
        total_loss += loss.item()
        loop.set_postfix(loss=total_loss / (loop.n + 1e-8))
    print(f"→ Época {epoch+1} finalizada — loss promedio: {total_loss/len(loader):.4f}")


def nn_bubble_sort(arr, model):
    a = arr.copy().astype('float32')
    n = len(a)
    model.eval()
    with torch.no_grad():
        for i in range(n):
            for j in range(n - i - 1):
                pair = torch.tensor([[a[j], a[j+1]]], dtype=torch.float32).to(device)
                swap_prob = model(pair).item()
                if swap_prob > 0.5:
                    a[j], a[j+1] = a[j+1], a[j]
    return a.astype(int)


entrada = input("Ingresa la lista de números separados por comas: ")
try:
    test_list = [int(x.strip()) for x in entrada.split(",")]
    sorted_pred = nn_bubble_sort(np.array(test_list), model)
    print("Entrada:", test_list)
    print("NN-BubbleSort:", sorted_pred.tolist())
except ValueError:
    print("Entrada inválida. Asegúrate de ingresar solo números separados por comas.")
