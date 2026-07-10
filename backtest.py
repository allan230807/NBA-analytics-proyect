import pandas as pd
import numpy as np
from predict import calcular_fuerza_viva, sigmoide # Importamos tu motor

def ejecutar_backtest(num_partidos=500):
    df_games = pd.read_csv('archive/games.csv')
    df_games['GAME_DATE_EST'] = pd.to_datetime(df_games['GAME_DATE_EST'])
    
    # Tomamos una muestra de partidos recientes (para que el modelo tenga historia)
    test_set = df_games.tail(num_partidos)
    
    aciertos = 0
    total = 0
    
    # Cargar pesos del modelo (asumiendo que ya tienes modelo_nba.npy)
    pesos = np.load('modelo_nba.npy')
    
    for _, row in test_set.iterrows():
        fecha = row['GAME_DATE_EST']
        local_id = row['HOME_TEAM_ID']
        vis_id = row['VISITOR_TEAM_ID']
        real_winner = 1 if row['HOME_TEAM_WINS'] == 1 else 0
        
        # Calcular fuerza dinámica al momento de ese partido
        f_L = calcular_fuerza_viva(local_id, fecha, df_games)
        f_V = calcular_fuerza_viva(vis_id, fecha, df_games)
        
        # Calcular predicción
        inputs = np.array([f_L, f_V, 1.0]) # 1.0 como factor base
        prob = sigmoide(np.dot(inputs, pesos))
        
        pred_winner = 1 if prob > 0.5 else 0
        
        if pred_winner == real_winner:
            aciertos += 1
        total += 1
        
    winrate = (aciertos / total) * 100
    print(f"=== RESULTADOS DEL BACKTESTING ===")
    print(f"Partidos evaluados: {total}")
    print(f"Precisión del modelo: {winrate:.2f}%")

if __name__ == "__main__":
    ejecutar_backtest()