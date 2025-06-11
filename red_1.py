import torch
import pandas as pd
import numpy as np
from transformers import DistilBertTokenizerFast, DistilBertModel
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
import torch.optim as optim
from tqdm import trange, tqdm
import random
import sys


NUM_EPOCHS = 10
BATCH_SIZE = 16
LEARNING_RATE = 2e-5

#Cargar el dataset desde CSV

csv_path = "dataset_complejidad_1.csv"
df = pd.read_csv(csv_path)

#Tokenizar codigos con distil
tokenizer = DistilBertTokenizerFast.from_pretrained('distilbert-base-uncased')

labels = ['O(1)', 'O(log n)', 'O(n)', 'O(n log n)', 'O(n^2)', 'O(n^3)', 'O(2^n)', 'O(n!)', 'O(1.6^n)']
label2id = {l: i for i, l in enumerate(labels)}

class CodeDataset(Dataset):
    def __init__(self, df):
        
        self.encodings = tokenizer(df['code'].tolist(), truncation=True, padding=True)
        self.labels_O = [label2id[o] for o in df['O']]
        self.labels_Omega = [label2id[o] for o in df['Omega']]
        self.labels_Theta = [label2id[o] for o in df['Theta']]

    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        item['labels_O'] = torch.tensor(self.labels_O[idx])
        item['labels_Omega'] = torch.tensor(self.labels_Omega[idx])
        item['labels_Theta'] = torch.tensor(self.labels_Theta[idx])
        return item

    def __len__(self):
        return len(self.labels_O)

# Dividir en entrenamiento y validación
train_df, val_df = train_test_split(df, test_size=0.2, random_state=42)
train_dataset = CodeDataset(train_df)
val_dataset = CodeDataset(val_df)

# 
class MultiTaskModel(nn.Module):
    def __init__(self, num_labels):
        super().__init__()
        self.bert = DistilBertModel.from_pretrained('distilbert-base-uncased')
        self.dropout = nn.Dropout(0.3)
        self.classifier_O = nn.Linear(768, num_labels)
        self.classifier_Omega = nn.Linear(768, num_labels)
        self.classifier_Theta = nn.Linear(768, num_labels)

    def forward(self, input_ids, attention_mask):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        pooled = self.dropout(outputs.last_hidden_state[:, 0])  # tomar [CLS]
        return {
            'O': self.classifier_O(pooled),
            'Omega': self.classifier_Omega(pooled),
            'Theta': self.classifier_Theta(pooled)
        }

# proceso para hacer el entrenamiento con gpu o cpu
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Usando dispositivo: {device}")
model = MultiTaskModel(num_labels=len(labels)).to(device)

#DataLoaders y optimizador/pérdida
train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
val_loader   = DataLoader(val_dataset,   batch_size=BATCH_SIZE)

optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE)
loss_fn = nn.CrossEntropyLoss()

# entrenamiento
for epoch in trange(NUM_EPOCHS, desc="Épocas"):
    model.train()
    total_loss = 0.0

    # Barra interna para batches
    with tqdm(train_loader, desc=f" Epoch {epoch+1}/{NUM_EPOCHS}", leave=False) as tloader:
        for batch in tloader:
            input_ids      = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels_O       = batch['labels_O'].to(device)
            labels_Omega   = batch['labels_Omega'].to(device)
            labels_Theta   = batch['labels_Theta'].to(device)

            optimizer.zero_grad()
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            loss = (
                loss_fn(outputs['O'], labels_O) +
                loss_fn(outputs['Omega'], labels_Omega) +
                loss_fn(outputs['Theta'], labels_Theta)
            ) / 3.0
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            tloader.set_postfix(loss=total_loss / (tloader.n + 1e-8))

    avg_loss = total_loss / len(train_loader)
    print(f"→ Época {epoch+1} finalizada — loss promedio: {avg_loss:.4f}")

print("Entrenamiento finalizado.\n")

#Función de inferencia
def predecir_complejidad(codigo: str):
    model.eval()
    with torch.no_grad():
        inputs = tokenizer(codigo, return_tensors="pt", truncation=True, padding=True)
        inputs = {k: v.to(device) for k, v in inputs.items()}
        outputs = model(input_ids=inputs['input_ids'], attention_mask=inputs['attention_mask'])
        pred_O     = labels[torch.argmax(outputs['O'], dim=1).item()]
        pred_Omega = labels[torch.argmax(outputs['Omega'], dim=1).item()]
        pred_Theta = labels[torch.argmax(outputs['Theta'], dim=1).item()]
    return {'O': pred_O, 'Ω': pred_Omega, 'Θ': pred_Theta}


print("Ahora puedes ingresar tu propio fragmento de código para predecir su complejidad.")
print("Escribe tu código línea a línea. Cuando termines, ingresa una línea vacía (ENTER).\n")

lineas_usuario = []
while True:
    try:
        linea = input()
    except EOFError:
     
        break
    if linea.strip() == "":
        break
    lineas_usuario.append(linea)

codigo_usuario = "\n".join(lineas_usuario).strip()
if len(codigo_usuario) == 0:
    print("No ingresaste ningún código. Saliendo.")
    sys.exit(0)


print("\n== Tu código ingresado ==\n")
print(codigo_usuario)
print("\n== Predicción de complejidad ==\n")
pred = predecir_complejidad(codigo_usuario)
print(f"Big-O: {pred['O']}")
print(f"Omega: {pred['Ω']}")
print(f"Theta: {pred['Θ']}")
