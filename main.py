import pandas as pd
import numpy as np
import os
from feature import preparar_datos_definitivo

def sigmoide(x):
    return 1 / (1 + np.exp(-x))

# Carga y Procesamiento
carpeta = os.path.dirname(__file__)
df_games = pd.read_csv(os.path.join(carpeta, 'archive', 'games.csv'))
df_ranking = pd.read_csv(os.path.join(carpeta, 'archive', 'ranking.csv'))

# Ahora recibimos los 3 elementos
X, y, df_fuerzas_finales = preparar_datos_definitivo(df_games, df_ranking)

# División
split = int(len(X) * 0.8)
X_train, X_test = X.iloc[:split], X.iloc[split:]
y_train, y_test = y.iloc[:split], y.iloc[split:]

# Entrenamiento
pesos = np.zeros(X_train.shape[1])
tasa = 0.01

for _ in range(1000):
    pred = sigmoide(np.dot(X_train, pesos))
    pesos -= tasa * np.dot(X_train.T, (pred - y_train)) / len(y_train)

# Exportación de Cerebro y Estado
np.save('modelo_nba.npy', pesos)
df_fuerzas_finales.to_csv('fuerzas_finales.csv', index=False)

# Evaluación
pred_final = (sigmoide(np.dot(X_test, pesos)) > 0.5).astype(int)
print(f"Precisión Final: {np.mean(pred_final == y_test):.2%}")
print("Archivos 'modelo_nba.npy' y 'fuerzas_finales.csv' generados con éxito.")