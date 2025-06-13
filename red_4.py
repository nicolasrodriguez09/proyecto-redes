import random
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

# Dataset para colas
class QueueDataset(Dataset):
    def __init__(self, num_samples=10000, max_seq_len=10, max_val=9):
        self.samples = []
        self.max_val = max_val
        self.max_seq_len = max_seq_len + 1
        self.pad_token = max_val + 2
        self.desencolar_token = 0
        for _ in range(num_samples):
            n_encolar = random.randint(1, max_seq_len)
            operations = []
            queue = []
            for _ in range(n_encolar):
                v = random.randint(0, max_val)
                operations.append(self.encolar_token(v))
                queue.append(v)
            operations.append(self.desencolar_token)
            label = queue.pop(0) if queue else -1
            seq = operations + [self.pad_token] * (self.max_seq_len - len(operations))
            self.samples.append((torch.tensor(seq, dtype=torch.long), label))

    def encolar_token(self, value):
        return value + 1  

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        seq, label = self.samples[idx]
        return seq, label

# Modelo RNN
class QueueRNN(nn.Module):
    def __init__(self, vocab_size, embed_dim=32, hidden_dim=64, num_classes=10):
        super(QueueRNN, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.lstm = nn.LSTM(embed_dim, hidden_dim, batch_first=True)
        self.fc = nn.Linear(hidden_dim, num_classes)

    def forward(self, x):
        emb = self.embedding(x)
        _, (h_n, _) = self.lstm(emb)
        out = self.fc(h_n[-1])
        return out

# tokenizar
def ops_to_sequence(ops, max_seq_len, pad_token):
    tokens = []
    for op in ops:
        if op[0] == 'encolar':
            tokens.append(op[1] + 1)
        elif op[0] == 'desencolar':
            tokens.append(0)
    return tokens + [pad_token] * (max_seq_len - len(tokens))

if __name__ == '__main__':
    num_samples = 20000
    max_seq_len = 10
    max_val = 9
    batch_size = 128
    epochs = 10
    lr = 0.001

    dataset = QueueDataset(num_samples, max_seq_len, max_val)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    vocab_size = max_val + 3
    model = QueueRNN(vocab_size, embed_dim=32, hidden_dim=64, num_classes=max_val+1)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    for epoch in range(1, epochs+1):
        total_loss = 0.0
        loop = tqdm(loader, desc=f"Epoch {epoch}/{epochs}")
        for seqs, labels in loop:
            logits = model(seqs)
            loss = criterion(logits, labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            loop.set_postfix(loss=total_loss / (loop.n + 1e-8))
        print(f"→ Época {epoch} finalizada — loss promedio: {total_loss/len(loader):.4f}")

    # Entrada por consola
    entrada = input("\nIngresa operaciones tipo encolar y desencolar: ")
    try:
        ops_raw = [op.strip() for op in entrada.split(",")]
        ops = []
        for op in ops_raw:
            if op.lower().startswith("encolar"):
                _, val = op.split()
                ops.append(("encolar", int(val)))
            elif op.lower().startswith("desencolar"):
                ops.append(("desencolar",))
        if not any(o[0] == "desencolar" for o in ops):
            raise ValueError("Debe haber al menos una operación 'desencolar'.")

        seq_tokens = ops_to_sequence(ops, dataset.max_seq_len, dataset.pad_token)
        input_seq = torch.tensor([seq_tokens], dtype=torch.long)
        with torch.no_grad():
            pred = torch.argmax(model(input_seq), dim=1).item()
        ops_str = ' -> '.join(f"{op[0]}({op[1]})" if op[0]=='encolar' else 'desencolar()' for op in ops)
        print(f"\nOperaciones: {ops_str}")
        print(f"Valor predicho al hacer desencolar(): {pred}")
    except Exception as e:
        print("Error en la entrada:", e)
