import pandas as pd
import os

def limpiar_datos():
    carpeta_proyecto = os.path.dirname(__file__)
    
    ruta_games = os.path.join(carpeta_proyecto, 'archive', 'games.csv')
    ruta_teams = os.path.join(carpeta_proyecto, 'archive', 'teams.csv')
    ruta_players = os.path.join(carpeta_proyecto, 'archive', 'players.csv')
    ruta_ranking = os.path.join(carpeta_proyecto, 'archive', 'ranking.csv')

    df_games = pd.read_csv(ruta_games)
    df_teams = pd.read_csv(ruta_teams)
    df_players = pd.read_csv(ruta_players)
    df_ranking = pd.read_csv(ruta_ranking)

    MAPPING_EQUIPOS = {
        '1610612737': 'ATL', '1610612738': 'BOS', '1610612740': 'NOP', 
        '1610612741': 'CHI', '1610612742': 'DAL', '1610612743': 'DEN', 
        '1610612745': 'HOU', '1610612746': 'LAC', '1610612747': 'LAL', 
        '1610612748': 'MIA', '1610612749': 'MIL', '1610612750': 'MIN', 
        '1610612751': 'BKN', '1610612752': 'NYK', '1610612753': 'ORL', 
        '1610612754': 'IND', '1610612755': 'PHI', '1610612756': 'PHX', 
        '1610612757': 'POR', '1610612758': 'SAC', '1610612759': 'SAS', 
        '1610612760': 'OKC', '1610612761': 'TOR', '1610612762': 'UTA', 
        '1610612763': 'MEM', '1610612764': 'WAS', '1610612765': 'DET', 
        '1610612766': 'CHA', '1610612739': 'CLE', '1610612744': 'GSW'
    }

    MAPA_EQUIPOS = {
        'Denver': 'DEN', 'Memphis': 'MEM', 'New Orleans': 'NOP', 'Phoenix': 'PHX',
        'LA Clippers': 'LAC', 'Sacramento': 'SAC', 'Utah': 'UTA', 'Portland': 'POR',
        'Dallas': 'DAL', 'Minnesota': 'MIN', 'Golden State': 'GSW', 'Oklahoma City': 'OKC',
        'L.A. Lakers': 'LAL', 'San Antonio': 'SAS', 'Houston': 'HOU', 'Milwaukee': 'MIL',
        'Boston': 'BOS', 'Cleveland': 'CLE', 'Brooklyn': 'BKN', 'Philadelphia': 'PHI',
        'New York': 'NYK', 'Atlanta': 'ATL', 'Indiana': 'IND', 'Miami': 'MIA',
        'Toronto': 'TOR', 'Chicago': 'CHI', 'Washington': 'WAS', 'Orlando': 'ORL',
        'Charlotte': 'CHA', 'Detroit': 'DET'
    }
    
    # === LIMPIEZA DF_GAMES ===
    df_games['TEAM_ID_home'] = df_games['TEAM_ID_home'].astype(int).astype(str)
    df_games['TEAM_ID_away'] = df_games['TEAM_ID_away'].astype(int).astype(str)
    df_games['TEAM_ABR_home'] = df_games['TEAM_ID_home'].map(MAPPING_EQUIPOS)
    df_games['TEAM_ABR_away'] = df_games['TEAM_ID_away'].map(MAPPING_EQUIPOS)
    df_games.drop(columns=['TEAM_ID_home', 'TEAM_ID_away'], inplace=True)
    
    df_games = df_games.dropna(subset=['PTS_home', 'PTS_away', 'TEAM_ABR_home', 'TEAM_ABR_away'])
    df_games = df_games[(df_games['PTS_home'] > 0) & (df_games['PTS_away'] > 0)]
    df_games.drop_duplicates(inplace=True)
    df_games['GAME_DATE_EST'] = pd.to_datetime(df_games['GAME_DATE_EST'])
    
    df_games.drop(columns=['Unnamed: 0', 'GAME_STATUS_TEXT', 'HOME_TEAM_ID', 'VISITOR_TEAM_ID'], inplace=True, errors='ignore')
    
    # Clasificación de tipo de juego
    df_games['GAME_TYPE'] = 'Otro'
    df_games.loc[(df_games['GAME_ID'] >= 10000000) & (df_games['GAME_ID'] < 20000000), 'GAME_TYPE'] = 'Preseason'
    df_games.loc[(df_games['GAME_ID'] >= 20000000) & (df_games['GAME_ID'] < 30000000), 'GAME_TYPE'] = 'Regular_Season'
    df_games.loc[(df_games['GAME_ID'] >= 40000000) & (df_games['GAME_ID'] < 50000000), 'GAME_TYPE'] = 'Playoffs'
    
    # Filtro de temporadas y exclusión de pretemporada
    df_games = df_games[(df_games['SEASON'] >= 2015) & (df_games['SEASON'] <= 2021)]
    df_games = df_games[df_games['GAME_TYPE'] != 'Preseason'] # <-- Filtro que faltaba
    
    # Reordenar columnas
    cols = list(df_games.columns)
    idx = cols.index('GAME_ID')
    cols.insert(idx + 1, cols.pop(cols.index('TEAM_ABR_home')))
    cols.insert(idx + 2, cols.pop(cols.index('TEAM_ABR_away')))
    df_games = df_games[cols]
    
    # === LIMPIEZA DF_TEAMS ===
    df_teams.drop_duplicates(inplace=True)
    df_teams.drop(columns=['Unnamed: 0', 'LEAGUE_ID', 'MIN_YEAR', 'MAX_YEAR'], inplace=True, errors='ignore')

    # === LIMPIEZA DF_PLAYERS ===
    df_players.drop_duplicates(inplace=True)
    df_players.drop(columns=['Unnamed: 0'], inplace=True, errors='ignore')

    # === LIMPIEZA DF_RANKING ===
    df_ranking.drop_duplicates(inplace=True)
    df_ranking.drop(columns=['Unnamed: 0', 'LEAGUE_ID', 'RETURNTOPLAY', 'TEAM_ID'], inplace=True, errors='ignore')
    
    # Mapeo y renombrado unificado
    df_ranking['TEAM'] = df_ranking['TEAM'].map(MAPA_EQUIPOS)
    df_ranking.rename(columns={'TEAM': 'TEAM_ABR'}, inplace=True)
    df_ranking = df_ranking.dropna(subset=['TEAM_ABR'])
    
    # Fechas y filtros de temporada
    df_ranking['STANDINGSDATE'] = pd.to_datetime(df_ranking['STANDINGSDATE'])
    df_ranking = df_ranking[(df_ranking['SEASON_ID'] % 10000 >= 2015) & 
                            (df_ranking['SEASON_ID'] % 10000 <= 2021)]
    df_ranking = df_ranking[~df_ranking['SEASON_ID'].astype(str).str.startswith('1')]
    
    # Ordenar y aplicar deduplicación por rendimiento invariable
    df_ranking = df_ranking.sort_values(by=['STANDINGSDATE'])
    df_ranking = df_ranking.drop_duplicates(
        subset=['TEAM_ABR', 'SEASON_ID', 'G', 'W', 'L'], 
        keep='first'
    )

    return df_games, df_teams, df_players, df_ranking

if __name__ == "__main__":
    df_games, df_teams, df_players, df_ranking = limpiar_datos()
    print("Muestra del Ranking Limpio:")
    print(df_ranking.head())
    print(f"\nEstructura final de juegos: {df_games.shape}")