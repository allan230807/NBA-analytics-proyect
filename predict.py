import numpy as np
import pandas as pd
import os

def calcular_fuerza_viva(team_id, fecha_ref, df_games, lambda_decay=0.01):
    """
    Calcula la fuerza de un equipo al momento 'fecha_ref' usando decaimiento exponencial.
    lambda_decay: controla qué tan rápido olvida el pasado (0.01 es un buen balance).
    """
    # Filtramos partidos jugados antes de la fecha y donde el equipo participó
    mask = ((df_games['HOME_TEAM_ID'] == team_id) | (df_games['VISITOR_TEAM_ID'] == team_id)) & \
           (df_games['GAME_DATE_EST'] < fecha_ref)
    partidos = df_games[mask].copy()
    
    if partidos.empty:
        return 1.0 # Fuerza base si no hay historial

    # Calculamos días transcurridos desde cada partido hasta la fecha de referencia
    partidos['dias_atras'] = (fecha_ref - partidos['GAME_DATE_EST']).dt.days
    
    # Factor de decaimiento: e^(-lambda * delta_t)
    partidos['peso'] = np.exp(-lambda_decay * partidos['dias_atras'])
    
    # Resultado (1 si ganó, 0 si perdió) y margen de victoria
    def obtener_resultado(row):
        es_local = (row['HOME_TEAM_ID'] == team_id)
        gano = (row['HOME_TEAM_WINS'] == 1) if es_local else (row['HOME_TEAM_WINS'] == 0)
        margen = abs(row['PTS_home'] - row['PTS_away'])
        return 1.0 if gano else 0.0, margen

    resultados = partidos.apply(lambda r: obtener_resultado(r), axis=1, result_type='expand')
    partidos['win_val'] = resultados[0]
    partidos['margen'] = resultados[1]
    
    # Fórmula: Suma ponderada de (Resultado * Margen) * PesoTemporal
    fuerza = ((partidos['win_val'] * (1 + partidos['margen']/10)) * partidos['peso']).sum()
    
    return max(0.5, fuerza / 10)

def sigmoide(x):
    return 1 / (1 + np.exp(-x))

# --- MAPEO DE TEXTO A NOMBRES OFICIALES ---
MAPEO_EQUIPOS = {
    'gsw': 'Golden State', 'gs': 'Golden State', 'golden state': 'Golden State', 'warriors': 'Golden State',
    'lal': 'L.A. Lakers', 'lakers': 'L.A. Lakers', 'la lakers': 'L.A. Lakers', 'l.a. lakers': 'L.A. Lakers',
    'bos': 'Boston', 'boston': 'Boston', 'celtics': 'Boston',
    'uta': 'Utah', 'uth': 'Utah', 'utah': 'Utah', 'jazz': 'Utah',
    'mia': 'Miami', 'miami': 'Miami', 'heat': 'Miami',
    'chi': 'Chicago', 'chicago': 'Chicago', 'bulls': 'Chicago',
    'sas': 'San Antonio', 'san antonio': 'San Antonio', 'spurs': 'San Antonio',
    'hou': 'Houston', 'houston': 'Houston', 'rockets': 'Houston',
    'den': 'Denver', 'denver': 'Denver', 'nuggets': 'Denver',
    'phx': 'Phoenix', 'phoenix': 'Phoenix', 'suns': 'Phoenix',
    'dal': 'Dallas', 'dallas': 'Dallas', 'mavericks': 'Dallas', 'mavs': 'Dallas',
    'mil': 'Milwaukee', 'milwaukee': 'Milwaukee', 'bucks': 'Milwaukee',
    'bkn': 'Brooklyn', 'brooklyn': 'Brooklyn', 'nets': 'Brooklyn',
    'nyk': 'New York', 'ny': 'New York', 'new york': 'New York', 'knicks': 'New York',
    'lac': 'L.A. Clippers', 'clippers': 'L.A. Clippers', 'la clippers': 'L.A. Clippers', 'l.a. clippers': 'L.A. Clippers',
    'phi': 'Philadelphia', 'philadelphia': 'Philadelphia', '76ers': 'Philadelphia', 'sixers': 'Philadelphia',
    'tor': 'Toronto', 'toronto': 'Toronto', 'raptors': 'Toronto',
    'mem': 'Memphis', 'memphis': 'Memphis', 'grizzlies': 'Memphis',
    'okc': 'Oklahoma City', 'oklahoma city': 'Oklahoma City', 'thunder': 'Oklahoma City',
    'min': 'Minnesota', 'minnesota': 'Minnesota', 'timberwolves': 'Minnesota', 'wolves': 'Minnesota',
    'nop': 'New Orleans', 'new orleans': 'New Orleans', 'pelicans': 'New Orleans',
    'sac': 'Sacramento', 'sacramento': 'Sacramento', 'kings': 'Sacramento',
    'atl': 'Atlanta', 'atlanta': 'Atlanta', 'hawks': 'Atlanta',
    'ind': 'Indiana', 'indiana': 'Indiana', 'pacers': 'Indiana',
    'was': 'Washington', 'washington': 'Washington', 'wizards': 'Washington',
    'orl': 'Orlando', 'orlando': 'Orlando', 'magic': 'Orlando',
    'cha': 'Charlotte', 'charlotte': 'Charlotte', 'hornets': 'Charlotte',
    'cle': 'Cleveland', 'cleveland': 'Cleveland', 'cavaliers': 'Cleveland', 'cavs': 'Cleveland',
    'det': 'Detroit', 'detroit': 'Detroit', 'pistons': 'Detroit',
    'por': 'Portland', 'portland': 'Portland', 'trail blazers': 'Portland', 'blazers': 'Portland'
}

# --- MAPEO DIRECTO E INFALIBLE A IDs NUMÉRICOS DE LA NBA ---
NBA_IDS = {
    'Atlanta': 1610612737, 'Boston': 1610612738, 'Cleveland': 1610612739,
    'New Orleans': 1610612740, 'Chicago': 1610612741, 'Dallas': 1610612742,
    'Denver': 1610612743, 'Golden State': 1610612744, 'Houston': 1610612745,
    'L.A. Clippers': 1610612746, 'L.A. Lakers': 1610612747, 'Miami': 1610612748,
    'Milwaukee': 1610612749, 'Minnesota': 1610612750, 'Brooklyn': 1610612751,
    'New York': 1610612752, 'Orlando': 1610612753, 'Indiana': 1610612754,
    'Philadelphia': 1610612755, 'Phoenix': 1610612756, 'Portland': 1610612757,
    'Sacramento': 1610612758, 'San Antonio': 1610612759, 'Oklahoma City': 1610612760,
    'Toronto': 1610612761, 'Utah': 1610612762, 'Memphis': 1610612763,
    'Washington': 1610612764, 'Detroit': 1610612765, 'Charlotte': 1610612766
}

def obtener_nombre_oficial(entrada):
    nombre = MAPEO_EQUIPOS.get(entrada.lower().strip())
    return nombre if nombre else entrada.strip()

def predecir_partido_cualquiera(local, visitante, pesos, df_fuerzas):
    if local not in df_fuerzas.index or visitante not in df_fuerzas.index:
        return f"Error: No se reconoció el equipo local '{local}' o visitante '{visitante}'."
    
    fuerza_L = df_fuerzas.loc[local, 'FUERZA']
    fuerza_V = df_fuerzas.loc[visitante, 'FUERZA']
    h2h_factor = 1.0

    inputs = np.array([fuerza_L, fuerza_V, h2h_factor])
    prob_local = sigmoide(np.dot(inputs, pesos))
    
    mostrar_dashboard(local, visitante, fuerza_L, fuerza_V, prob_local)

def predecir_con_fecha_real(local, visitante, fecha_str, pesos, df_fuerzas):
    try:
        carpeta = os.path.dirname(__file__)
        df_games = pd.read_csv(os.path.join(carpeta, 'archive', 'games.csv'))
        df_games['GAME_DATE_EST'] = pd.to_datetime(df_games['GAME_DATE_EST']).dt.normalize()
    except Exception:
        return "Error: No se pudo cargar archive/games.csv"

    id_local = NBA_IDS.get(local, local)
    id_visitante = NBA_IDS.get(visitante, visitante)
    fecha_target = pd.to_datetime(fecha_str).normalize()

    # 1. Búsqueda de Descanso (Historial individual)
    partidos_local = df_games[((df_games['HOME_TEAM_ID'] == id_local) | (df_games['VISITOR_TEAM_ID'] == id_local)) & (df_games['GAME_DATE_EST'] < fecha_target)]
    partidos_vis = df_games[((df_games['HOME_TEAM_ID'] == id_visitante) | (df_games['VISITOR_TEAM_ID'] == id_visitante)) & (df_games['GAME_DATE_EST'] < fecha_target)]

    rest_L, rest_V = 3, 3 
    str_fecha_L, str_fecha_V = "Sin registro", "Sin registro"

    if not partidos_local.empty:
        ultima_fecha_L = partidos_local.sort_values('GAME_DATE_EST').iloc[-1]['GAME_DATE_EST']
        rest_L = (fecha_target - ultima_fecha_L).days
        str_fecha_L = ultima_fecha_L.strftime('%Y-%m-%d')
        
    if not partidos_vis.empty:
        ultima_fecha_V = partidos_vis.sort_values('GAME_DATE_EST').iloc[-1]['GAME_DATE_EST']
        rest_V = (fecha_target - ultima_fecha_V).days
        str_fecha_V = ultima_fecha_V.strftime('%Y-%m-%d')

    fatiga_L = 0.85 if rest_L <= 1 else 1.0
    fatiga_V = 0.85 if rest_V <= 1 else 1.0
    fuerza_base_L = calcular_fuerza_viva(id_local, fecha_target, df_games)
    fuerza_base_V = calcular_fuerza_viva(id_visitante, fecha_target, df_games)
    
    # Imprimir para depurar y ver que los números ya no son 3.89 y 0.95
    print(f"DEBUG: Fuerza Viva Calculada -> Local: {fuerza_base_L:.2f} | Vis: {fuerza_base_V:.2f}")

    # 2. Búsqueda H2H (Últimos 5 enfrentamientos directos)
    h2h_games = df_games[
        (((df_games['HOME_TEAM_ID'] == id_local) & (df_games['VISITOR_TEAM_ID'] == id_visitante)) | 
         ((df_games['HOME_TEAM_ID'] == id_visitante) & (df_games['VISITOR_TEAM_ID'] == id_local))) & 
        (df_games['GAME_DATE_EST'] < fecha_target)
    ].sort_values('GAME_DATE_EST').tail(5)

    # Ponderación: 40% (más reciente), 30%, 15%, 10%, 5% (más antiguo)
    pesos_h2h = [0.40, 0.30, 0.15, 0.10, 0.05]
    dominio_local = 0.0
    dominio_vis = 0.0

    # Invertir el dataframe para iterar desde el juego más reciente hacia atrás
    h2h_games_recent = h2h_games.iloc[::-1].reset_index()

    for i, row in h2h_games_recent.iterrows():
        if i >= len(pesos_h2h): break
        peso_actual = pesos_h2h[i]
        
        # Evaluar quién ganó el partido histórico
        if row['HOME_TEAM_ID'] == id_local:
            if row['HOME_TEAM_WINS'] == 1:
                dominio_local += peso_actual
            else:
                dominio_vis += peso_actual
        else: # El equipo local de ese juego fue nuestro visitante actual
            if row['HOME_TEAM_WINS'] == 1:
                dominio_vis += peso_actual
            else:
                dominio_local += peso_actual

    # Modificador dinámico (máximo +- 15% de alteración en la fuerza base por H2H)
    mod_h2h_L = 1.0 + (dominio_local * 0.15)
    mod_h2h_V = 1.0 + (dominio_vis * 0.15)

    # 3. Aplicación de Variables
    fuerza_final_L = fuerza_base_L * fatiga_L * mod_h2h_L
    fuerza_final_V = fuerza_base_V * fatiga_V * mod_h2h_V

    # La fuerza final ahora depende de la fatiga Y del momentum contra ese rival específico
    fuerza_final_L = fuerza_base_L * fatiga_L * mod_h2h_L
    fuerza_final_V = fuerza_base_V * fatiga_V * mod_h2h_V
    
    h2h_factor_localia = 1.1 if rest_L > rest_V else 0.9

    inputs = np.array([fuerza_final_L, fuerza_final_V, h2h_factor_localia])
    prob_local = sigmoide(np.dot(inputs, pesos))

    print(f"\n=== ANÁLISIS DE CALENDARIO PARA EL {fecha_str} ===")
    print(f"Último juego de {local}: {str_fecha_L} -> Diferencia: {rest_L} días")
    print(f"Último juego de {visitante}: {str_fecha_V} -> Diferencia: {rest_V} días")
    print(f"\n=== HISTORIAL RECIENTE CARA A CARA (Últimos {len(h2h_games_recent)} juegos) ===")
    print(f"Dominio H2H {local}: {dominio_local:.2%} (Modificador aplicado: {mod_h2h_L:.2f}x)")
    print(f"Dominio H2H {visitante}: {dominio_vis:.2%} (Modificador aplicado: {mod_h2h_V:.2f}x)")
    
    mostrar_dashboard(local, visitante, fuerza_final_L, fuerza_final_V, prob_local)

def mostrar_dashboard(local, visitante, f_L, f_V, prob_local):
    print(f"\n=== PREDICCIÓN CONTEXTUAL: {local} vs {visitante} ===")
    print(f"Fuerza {local} (Ajustada): {f_L:.2f}")
    print(f"Fuerza {visitante} (Ajustada): {f_V:.2f}")
    print(f"Probabilidad de victoria de {local}: {prob_local:.2%}")
    print(f"Probabilidad de victoria de {visitante}: {(1 - prob_local):.2%}")
    if prob_local > 0.5:
        print(f"\nGanador predicho: {local} 🏠")
    else:
        print(f"\nGanador predicho: {visitante} 🚀")

if __name__ == "__main__":
    try:
        pesos = np.load('modelo_nba.npy')
        df_fuerzas = pd.read_csv('fuerzas_finales.csv', index_col='TEAM')
    except FileNotFoundError:
        print("Error: Ejecuta primero main.py para generar las matrices del modelo.")
        exit()

    print("--- CONSULTOR AVANZADO NBA ANALYTICS ---")
    print("1. Partido cualquiera (Fuerza neutra)")
    print("2. Simular partido en fecha específica (Cálculo dinámico de descanso/fatiga - ej: Temporada 2022)")
    
    opcion = input("\nElige una opción (1 o 2): ").strip()

    eq_local = obtener_nombre_oficial(input("Introduce equipo LOCAL: "))
    eq_vis = obtener_nombre_oficial(input("Introduce equipo VISITANTE: "))

    if opcion == "2":
        fecha = input("Introduce la fecha a simular (AAAA-MM-DD, ej: 2022-03-15): ").strip()
        predecir_con_fecha_real(eq_local, eq_vis, fecha, pesos, df_fuerzas)
    else:
        predecir_partido_cualquiera(eq_local, eq_vis, pesos, df_fuerzas)