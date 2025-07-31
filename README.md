# Machine Learning Example

<img width="1334" height="409" alt="imagem" src="https://github.com/user-attachments/assets/1ef7a9b1-89bf-4210-91a9-b0e59331d28f" />

If you want to test the example above, run the following emulated data:

```
np.random.seed(42)
n = 150

# Matrix X
X = pd.DataFrame({
    'reagent 1': np.random.uniform(50, 150, n),
    'reagent 2': np.random.uniform(100, 200, n),
    'reagent 3': np.random.uniform(1, 5, n),
    'catalyst': np.random.uniform(0.1, 1.0, n)
})

# Vector y
y = pd.DataFrame({
    'melt point': 0.02 * X['reagent 1'] + 0.01 * X['reagent 2'] - 0.4 * X['reagent 3'] + 2.5 * X['catalyst'] + np.random.normal(0, 0.2, n),
    'viscosity': 0.05 * X['reagent 1'] + 0.01 * X['reagent 2'] + 0.5 * X['catalyst'] + np.random.normal(0, 1.4, n),
    'solubility': 45 - 0.08 * X['reagent 3'] - 0.018 * X['reagent 1'] + np.random.normal(0, 0.3, n)
})
```
