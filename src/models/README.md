# 8c-v0-beta

**8c-v0-beta** to minimalistyczny, własnoręcznie zbudowany framework do uczenia maszynowego i obliczeń tensorowych. Jest lekki, przejrzysty i w 100% pozbawiony zewnętrznych zależności (oprócz matplotlib dla wizualizacji). Stworzony w duchu edukacyjnym i optymalizacyjnym – jako podstawa do budowy własnych modeli ML.

---

## 🚀 Funkcjonalności

- ✅ Własna klasa `Array` (odpowiednik NumPy)
- ✅ Klasa `Tensor` z autograd (automatyczne różniczkowanie)
- ✅ Graf obliczeń (`Node`, `Op`, np. Add, Mul, MatMul, Pow, ReLU)
- ✅ Warstwy `Linear`, `ReLU`, `Sequential`
- ✅ Funkcja straty `MSELoss`
- ✅ Optymalizator `SGD`
- ✅ Trener z pętlą uczącą `Trainer`
- ✅ Przykład modelu uczącego się XOR + zapis modelu + wykres

---

## 📂 Struktura projektu

```
8c-v0-beta/
├── core/           # Array, Tensor
│   ├── array.py
│   └── tensor.py
│
├── graph/          # Graf obliczeń + operacje
│   └── node.py
│
├── nn/             # Warstwy (Dense, ReLU)
│   └── layers.py
│
├── optim/          # Optymalizatory
│   └── sgd.py
│
├── train/          # Trener, funkcje straty
│   ├── losses.py
│   └── trainer.py
│
├── main.py         # Przykład modelu XOR
└── xor_loss_curve.png  # Wygenerowany wykres straty
```

---

## 🧠 Jak działa `main.py`

1. Tworzy dane XOR (`X`, `y`)
2. Buduje model: `Linear(2→4) → ReLU → Linear(4→1)`
3. Trenuje model przez 1000 epok
4. Zapisuje model do pliku `.pkl`
5. Wczytuje go z pliku i wykonuje predykcje
6. Testuje model na nowych danych
7. Generuje wykres straty `xor_loss_curve.png`

---

## 📦 Szybki start

```bash
pip install matplotlib
python __main__.py
```

---

## ✅ Przykład działania
```
Epoch    0 | Loss: [0.3812]
Epoch  100 | Loss: [0.1123]
...
Epoch  900 | Loss: [0.0217]

Predictions (after reload):
Array(...)

Custom test:
Array(...)
```

---

## 🔧 Planowane rozszerzenia

- [ ] CrossEntropyLoss
- [ ] Softmax
- [ ] Obsługa batching
- [ ] Conv2D + MaxPool
- [ ] Eksport do ONNX lub JSON

---

