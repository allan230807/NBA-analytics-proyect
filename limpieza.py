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
    df_games['TEAM_ID_home'] = df_games['TEAM_ID_home'].astype(int).astype(str)
    df_games['TEAM_ID_away'] = df_games['TEAM_ID_away'].astype(int).astype(str)
    df_games['TEAM_ABR_home'] = df_games['TEAM_ID_home'].map(MAPPING_EQUIPOS)
    df_games['TEAM_ABR_away'] = df_games['TEAM_ID_away'].map(MAPPING_EQUIPOS)
    df_games.drop(columns=['TEAM_ID_home', 'TEAM_ID_away'], inplace=True)
    #  Limpieza de datos en df_games
    df_games = df_games.dropna(subset=['PTS_home', 'PTS_away', 'TEAM_ABR_home', 'TEAM_ABR_away'])
    df_games = df_games[df_games['PTS_home'] > 0]
    df_games = df_games[df_games['PTS_away'] > 0]
    df_games.drop_duplicates(inplace=True)
    df_games['GAME_DATE_EST'] = pd.to_datetime(df_games['GAME_DATE_EST'])
    df_games.drop(columns=['Unnamed: 0'], inplace=True, errors='ignore')
    df_games.drop(columns=['GAME_STATUS_TEXT'], inplace=True, errors='ignore')
    df_games['GAME_TYPE'] = 'Otro'
    df_games.loc[(df_games['GAME_ID'] >= 10000000) & (df_games['GAME_ID'] < 20000000), 'GAME_TYPE'] = 'Preseason'
    df_games.loc[(df_games['GAME_ID'] >= 20000000) & (df_games['GAME_ID'] < 30000000), 'GAME_TYPE'] = 'Regular_Season'
    df_games.loc[(df_games['GAME_ID'] >= 40000000) & (df_games['GAME_ID'] < 50000000), 'GAME_TYPE'] = 'Playoffs'
    df_games = df_games[(df_games['SEASON'] >= 2015) & (df_games['SEASON'] <= 2022)]
    df_games.drop(columns=['HOME_TEAM_ID', 'VISITOR_TEAM_ID'], inplace=True, errors='ignore')
    cols = list(df_games.columns)
    idx = cols.index('GAME_ID')
    cols.insert(idx + 1, cols.pop(cols.index('TEAM_ABR_home')))
    cols.insert(idx + 2, cols.pop(cols.index('TEAM_ABR_away')))
    
    df_games = df_games[cols]
    # Limpieza de datos en df_teams
    df_teams.drop_duplicates(inplace=True)
    df_teams.drop(columns=['Unnamed: 0'], inplace=True, errors='ignore')
    df_teams = df_teams.drop(columns=['LEAGUE_ID'])
    df_teams = df_teams.drop(columns=['MIN_YEAR'])
    df_teams = df_teams.drop(columns=['MAX_YEAR'])
    
    return df_games, df_teams, df_players, df_ranking


# --- Ejecución del script ---
if __name__ == "__main__":
    df_games, df_teams, df_players, df_ranking = limpiar_datos()
    
    print(df_games.head())