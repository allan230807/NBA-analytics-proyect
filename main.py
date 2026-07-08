import pandas as pd
import numpy as np
import os
from feature import preparar_datos_definitivo

# 1. Función matemática para la Regresión Logística
def sigmoide(x):
    return 1 / (1 + np.exp(-x))

# 2. Configuración de rutas dinámicas a los archivos CRUDOS
carpeta_proyecto = os.path.dirname(__file__)
ruta_games = os.path.join(carpeta_proyecto, 'archive', 'games.csv')
ruta_ranking = os.path.join(carpeta_proyecto, 'archive', 'ranking.csv')

# 3. Carga directa sin pasar por el script de limpieza roto
df_games_raw = pd.read_csv(ruta_games)
df_ranking_raw = pd.read_csv(ruta_ranking)

# 4. Saneamiento inmediato de las columnas críticas antes de procesar
# Renombramos las columnas del juego crudo para que coincidan con lo que espera feature.py
df_games_raw = df_games_raw.rename(columns={
    'HOME_TEAM_ID': 'TEAM_ABR_home', 
    'VISITOR_TEAM_ID': 'TEAM_ABR_away'
})

# Nos aseguramos de eliminar duplicados reales del juego aquí mismo
df_games_raw = df_games_raw.drop_duplicates(subset=['GAME_ID']).copy() if 'GAME_ID' in df_games_raw.columns else df_games_raw.drop_duplicates(subset=['GAME_DATE_EST', 'TEAM_ABR_home']).copy()

# 5. Procesamiento
X, y = preparar_datos_definitivo(df_games_raw, df_ranking_raw)

# --- DIAGNÓSTICO ESTRICTO ---
print(f"Total de partidos evaluados: {len(X)}")
print(f"Victorias locales reales (Distribución del Target): {y.mean():.2%}")
# ----------------------------

# 6. División manual de datos (80% entrenamiento, 20% prueba)
split = int(len(X) * 0.8)
X_train, X_test = X.iloc[:split], X.iloc[split:]
y_train, y_test = y.iloc[:split], y.iloc[split:]

# 7. Entrenamiento del modelo (Descenso de Gradiente)
pesos = np.zeros(X_train.shape[1])
tasa_aprendizaje = 0.01

for _ in range(1000):
    prediccion = sigmoide(np.dot(X_train, pesos))
    error = prediccion - y_train
    gradiente = np.dot(X_train.T, error) / len(y_train)
    pesos -= tasa_aprendizaje * gradiente

# 8. Evaluación
predicciones_finales = (sigmoide(np.dot(X_test, pesos)) > 0.5).astype(int)
precision = np.mean(predicciones_finales == y_test)

print(f"Precisión: {precision:.2%}")
print(f"Pesos de las variables: {pesos}")