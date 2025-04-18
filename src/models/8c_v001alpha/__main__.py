# 8c-v0-beta/__main__.py - Przykład treningu XOR

"""
Tworzy dane wejściowe i wyjściowe dla XOR

Buduje models: Linear → ReLU → Linear

Używa Trainer z MSELoss i SGD

Trenuje przez 1000 epok

Drukuje stratę co 100 epok

Na końcu pokazuje predykcje 🔍

"""

import pickle
import matplotlib.pyplot as plt
from .core.tensor import Tensor
from nn.layers import Sequential, Linear, ReLU_
from train.trainer import Trainer

# XOR dane treningowe
X_data = [
    [0.0, 0.0],
    [0.0, 1.0],
    [1.0, 0.0],
    [1.0, 1.0]
]

y_data = [
    [0.0],
    [1.0],
    [1.0],
    [0.0]
]

X = Tensor(X_data, requires_grad=False)
y = Tensor(y_data, requires_grad=False)

# Definiujemy models
model = Sequential(
    Linear(2, 4),
    ReLU_(),
    Linear(4, 1)
)

trainer = Trainer(model, lr=0.1)
loss_history = []

# Trening
for epoch in range(1000):
    loss = trainer.train_step(X, y)
    loss_history.append(loss.data[0])
    if epoch % 100 == 0:
        print(f"Epoch {epoch:4d} | Loss: {loss.data}")

# Zapis modelu
with open("model_8c_xor.pkl", "wb") as f:
    pickle.dump(model, f)

# Odczyt modelu
with open("model_8c_xor.pkl", "rb") as f:
    loaded_model = pickle.load(f)

# Predykcje z załadowanego modelu
print("\nPredictions (after reload):")
out = loaded_model(X)
print(out)

# Testowanie na nowych danych
print("\nCustom test:")
test_data = Tensor([[0.5, 0.5], [1.0, 1.0]], requires_grad=False)
print(loaded_model(test_data))

# Wykres straty
plt.plot(loss_history)
plt.title("Training Loss (XOR)")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.grid(True)
plt.savefig("xor_loss_curve.png")
plt.show()