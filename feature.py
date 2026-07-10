import pandas as pd
import numpy as np
from limpieza import limpiar_datos
def preparar_datos_definitivo(df_games, df_ranking):
    df_games['GAME_DATE_EST'] = pd.to_datetime(df_games['GAME_DATE_EST'], errors='coerce')
    df_games = df_games.dropna(subset=['GAME_DATE_EST']).copy()
    df_games = df_games[(df_games['GAME_DATE_EST'].dt.year >= 2015) & (df_games['GAME_DATE_EST'].dt.year <= 2021)].copy()
    
    # --- TRADUCCIÓN DE IDs NUMÉRICOS A TEXTO (Abreviaturas) ---
    # Creamos un diccionario rápido usando el ranking {TEAM_ID: ABREVIACIÓN_TEXTO}
    # En el ranking crudo, 'TEAM_ID' es numérico y 'TEAM' es el texto (ej: LAL)
    id_a_abr = df_ranking.dropna(subset=['TEAM_ID', 'TEAM']).set_index('TEAM_ID')['TEAM'].to_dict()
    
    # Si las columnas vienen crudas como HOME_TEAM_ID, las traducimos a texto de una vez
    if 'HOME_TEAM_ID' in df_games.columns:
        df_games['TEAM_ABR_home'] = df_games['HOME_TEAM_ID'].map(id_a_abr)
        df_games['TEAM_ABR_away'] = df_games['VISITOR_TEAM_ID'].map(id_a_abr)
    
    # Forzamos que ambas columnas sean tratadas como texto string por seguridad
    df_games['TEAM_ABR_home'] = df_games['TEAM_ABR_home'].astype(str)
    df_games['TEAM_ABR_away'] = df_games['TEAM_ABR_away'].astype(str)
    # -----------------------------------------------------------

    # 2. Eliminación radical de duplicados reales
    if 'GAME_ID' in df_games.columns:
        df_games = df_games.drop_duplicates(subset=['GAME_ID']).copy()
    else:
        df_games = df_games.drop_duplicates(subset=['GAME_DATE_EST', 'TEAM_ABR_home']).copy()
        
    df_games = df_games.sort_values('GAME_DATE_EST').reset_index(drop=True)
    
    # 3. Preparación del Ranking (Asegurando tipo string)
    if 'TEAM' in df_ranking.columns and 'TEAM_ABR' not in df_ranking.columns:
        df_ranking = df_ranking.rename(columns={'TEAM': 'TEAM_ABR'})
        
    df_ranking['TEAM_ABR'] = df_ranking['TEAM_ABR'].astype(str)
    df_ranking['RANK'] = df_ranking.groupby('SEASON_ID')['W_PCT'].rank(ascending=False)
    
    # Reducimos a un solo registro por equipo/temporada usando la última fecha disponible
    df_ref = df_ranking.groupby(['SEASON_ID', 'TEAM_ABR']).last().reset_index()
    df_ref = df_ref[['SEASON_ID', 'TEAM_ABR', 'RANK']].copy()
    df_ref['SEASON'] = (df_ref['SEASON_ID'] % 10000) + 1
        
    # 4. Orden cronológico para la simulación de fuerza
    df_games = df_games.sort_values('GAME_DATE_EST').reset_index(drop=True)
    
    # Cruce seguro (Merge 1 a 1 por equipo por temporada)
    df_games = pd.merge(df_games, df_ref.rename(columns={'TEAM_ABR': 'TEAM_ABR_away', 'RANK': 'RANK_away'}), 
                        on=['SEASON', 'TEAM_ABR_away'], how='left')
    
    df_games['W_TIEMPO'] = df_games['SEASON'].map({2015: 0.5, 2016: 0.7, 2017: 0.9, 2018: 1.1, 2019: 1.4, 2020: 1.6, 2021: 1.8}).fillna(1.0)
    df_games['W_TIPO'] = 0.33

    df_games['REST_DAYS'] = df_games.groupby('TEAM_ABR_home')['GAME_DATE_EST'].diff().dt.days.fillna(3)
    df_games['W_FATIGA'] = np.where(df_games['REST_DAYS'] == 0, 0.8, 1.0)
    df_games['W_SOS'] = np.where(df_games['RANK_away'] <= 10, 1.3, 1.0)
    df_games['PESO_TOTAL'] = df_games['W_TIEMPO'] * df_games['W_TIPO'] * df_games['W_FATIGA'] * df_games['W_SOS']
    
    # Differential de puntos puro
    df_games['MOV'] = (df_games['PTS_home'] - df_games['PTS_away']) * df_games['PESO_TOTAL']
    
    # 3. Seguimiento estado de forma real (Cero fugas al futuro)
    fuerza_actual = {} # Guarda la última fuerza calculada de cada equipo
    fuerza_local_col = []
    fuerza_visitante_col = []
    
    # Recorremos cronológicamente para simular el paso del tiempo real
    for idx, row in df_games.iterrows():
        loc = row['TEAM_ABR_home']
        vis = row['TEAM_ABR_away']
        
        # Asignamos la fuerza que TENÍAN antes de jugar hoy
        fuerza_local_col.append(fuerza_actual.get(loc, 0.0))
        fuerza_visitante_col.append(fuerza_actual.get(vis, 0.0))
        
        # Actualizamos su fuerza CON el resultado de hoy para el próximo partido
        # El local suma el MOV, el visitante lo resta (si el local gana por 10, el visitante pierde por 10)
        fuerza_actual[loc] = (fuerza_actual.get(loc, 0.0) * 0.9) + (row['MOV'] * 0.1)
        fuerza_actual[vis] = (fuerza_actual.get(vis, 0.0) * 0.9) - (row['MOV'] * 0.1)
        
    df_games['FUERZA_LOCAL'] = fuerza_local_col
    df_games['FUERZA_VISITANTE'] = fuerza_visitante_col
    
    # 4. Factor H2H Veloz
    equipos = np.sort(df_games[['TEAM_ABR_home', 'TEAM_ABR_away']].values, axis=1)
    df_games['H2H_PAIR'] = equipos[:, 0] + "_" + equipos[:, 1]
    df_games['HOME_WON'] = (df_games['PTS_home'] > df_games['PTS_away']).astype(int)
    df_games['FIRST_TEAM_WON'] = np.where(df_games['TEAM_ABR_home'] == equipos[:, 0], df_games['HOME_WON'], 1 - df_games['HOME_WON'])
    
    h2h_rolling = df_games.groupby('H2H_PAIR')['FIRST_TEAM_WON'].shift(1).rolling(window=12, min_periods=1).mean()
    df_games['H2H_RATIO'] = np.where(df_games['TEAM_ABR_home'] == equipos[:, 0], h2h_rolling, 1 - h2h_rolling)
    df_games['H2H_RATIO'] = df_games['H2H_RATIO'].fillna(0.5)
    
    condiciones = [df_games['H2H_RATIO'] >= 0.9, df_games['H2H_RATIO'] >= 0.5, df_games['H2H_RATIO'] <= 0.2]
    elecciones = [1.4, 1.1, 0.7]
    df_games['H2H_FACTOR'] = np.select(condiciones, elecciones, default=1.0)
    
    # Target
    df_games['TARGET'] = (df_games['PTS_home'] > df_games['PTS_away']).astype(int)
    
    # NUEVO: Convertimos el diccionario de las fuerzas finales en un DataFrame
    df_fuerzas_finales = pd.DataFrame(list(fuerza_actual.items()), columns=['TEAM', 'FUERZA'])
    
    return df_games[['FUERZA_LOCAL', 'FUERZA_VISITANTE', 'H2H_FACTOR']], df_games['TARGET'], df_fuerzas_finales