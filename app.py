import streamlit as st
import pandas as pd
import numpy as np
import os
import sys
import streamlit.components.v1 as components

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from predict import calcular_fuerza_viva, sigmoide, NBA_IDS

st.set_page_config(page_title="Consultor Avanzado NBA", layout="wide", page_icon="🏀")

@st.cache_data
def cargar_datos_historicos():
    carpeta = os.path.dirname(__file__)
    df = pd.read_csv(os.path.join(carpeta, 'archive', 'games.csv'))
    df['GAME_DATE_EST'] = pd.to_datetime(df['GAME_DATE_EST']).dt.normalize()
    
    df_filtrado = df[(df['GAME_DATE_EST'] >= '2018-10-16') & (df['GAME_DATE_EST'] <= '2022-06-20')].copy()
    return df, df_filtrado

df_games_completo, df_games_rango = cargar_datos_historicos()

@st.cache_resource
def cargar_pesos():
    return np.load('modelo_nba.npy')

pesos = cargar_pesos()

# --- FUNCIÓN DE FUEGOS ARTIFICIALES ---
def mostrar_fuegos_artificiales():
    components.html(
        """
        <div style="display: flex; justify-content: center; align-items: center; height: 180px;">
            <h1 style="font-family: sans-serif; color: #2e7d32; text-align: center;">✅ ¡Predicción Acertada! 🎆</h1>
        </div>
        <script src="https://cdn.jsdelivr.net/npm/canvas-confetti@1.6.0/dist/confetti.browser.min.js"></script>
        <script>
            var duration = 3 * 1000;
            var animationEnd = Date.now() + duration;
            var defaults = { startVelocity: 30, spread: 360, ticks: 60, zIndex: 0 };
            function randomInRange(min, max) { return Math.random() * (max - min) + min; }
            var interval = setInterval(function() {
                var timeLeft = animationEnd - Date.now();
                if (timeLeft <= 0) { return clearInterval(interval); }
                var particleCount = 50 * (timeLeft / duration);
                confetti(Object.assign({}, defaults, { particleCount, origin: { x: randomInRange(0.1, 0.3), y: Math.random() - 0.2 } }));
                confetti(Object.assign({}, defaults, { particleCount, origin: { x: randomInRange(0.7, 0.9), y: Math.random() - 0.2 } }));
            }, 250);
        </script>
        """,
        height=200
    )

# --- INICIALIZADOR ALEATORIO (Busca un acierto al arrancar) ---
if 'inicializado' not in st.session_state:
    for _, row in df_games_rango.sample(frac=1).iterrows():
        id_L = row['HOME_TEAM_ID']
        id_V = row['VISITOR_TEAM_ID']
        fecha = row['GAME_DATE_EST']
        
        try:
            local_name = [k for k, v in NBA_IDS.items() if v == id_L][0]
            vis_name = [k for k, v in NBA_IDS.items() if v == id_V][0]
        except IndexError:
            continue 
            
        f_L = calcular_fuerza_viva(id_L, fecha, df_games_completo)
        f_V = calcular_fuerza_viva(id_V, fecha, df_games_completo)
        
        # Inferencia rápida sin H2H pesado solo para buscar un acierto inicial
        prob = sigmoide(np.dot(np.array([f_L, f_V, 1.0]), pesos))
        if (prob > 0.5) == (row['HOME_TEAM_WINS'] == 1):
            st.session_state['init_local'] = local_name
            st.session_state['init_vis'] = vis_name
            st.session_state['init_fecha'] = fecha.strftime('%Y-%m-%d')
            st.session_state['inicializado'] = True
            break

# --- INTERFAZ DEL CONSULTOR ---
st.title("🏀 Consultor NBA Analytics: Histórico 2018-2022")
st.markdown("Analiza cualquier partido comprendido entre las temporadas 2018-2019 y 2021-2022.")

st.sidebar.header("Configuración del Emparejamiento")
lista_equipos = sorted(list(NBA_IDS.keys()))

# Cargar los equipos inicializados
def_local_idx = lista_equipos.index(st.session_state.get('init_local', 'Miami'))
def_vis_idx = lista_equipos.index(st.session_state.get('init_vis', 'Boston'))

local_nombre = st.sidebar.selectbox("Equipo Local 🏠", lista_equipos, index=def_local_idx)
visitante_nombre = st.sidebar.selectbox("Equipo Visitante 🚀", lista_equipos, index=def_vis_idx)

id_local = NBA_IDS[local_nombre]
id_visitante = NBA_IDS[visitante_nombre]

partidos_cruzados = df_games_rango[
    (df_games_rango['HOME_TEAM_ID'] == id_local) & (df_games_rango['VISITOR_TEAM_ID'] == id_visitante)
].sort_values('GAME_DATE_EST', ascending=False) 

if partidos_cruzados.empty:
    st.warning(f"No se encontraron partidos donde {local_nombre} haya sido LOCAL contra {visitante_nombre} en el rango 2018-2022.")
    fecha_target = None
else:
    opciones_fechas = {row['GAME_DATE_EST'].strftime('%Y-%m-%d'): row['GAME_DATE_EST'] for _, row in partidos_cruzados.iterrows()}
    fechas_str = list(opciones_fechas.keys())
    
    # Cargar la fecha inicializada si coincide con los equipos
    default_f = st.session_state.get('init_fecha', fechas_str[0])
    idx_fecha = fechas_str.index(default_f) if default_f in fechas_str else 0
    
    seleccion_fecha_str = st.sidebar.selectbox("Selecciona una fecha para analizar:", fechas_str, index=idx_fecha)
    fecha_target = opciones_fechas[seleccion_fecha_str]

# --- PROCESAMIENTO Y DIAGNÓSTICO ---
if fecha_target:
    st.subheader(f"📊 Diagnóstico del Consultor para la fecha: {fecha_target.strftime('%Y-%m-%d')}")
    
    partidos_local = df_games_completo[((df_games_completo['HOME_TEAM_ID'] == id_local) | (df_games_completo['VISITOR_TEAM_ID'] == id_local)) & (df_games_completo['GAME_DATE_EST'] < fecha_target)]
    partidos_vis = df_games_completo[((df_games_completo['HOME_TEAM_ID'] == id_visitante) | (df_games_completo['VISITOR_TEAM_ID'] == id_visitante)) & (df_games_completo['GAME_DATE_EST'] < fecha_target)]

    rest_L, rest_V = 3, 3 
    if not partidos_local.empty:
        rest_L = (fecha_target - partidos_local.sort_values('GAME_DATE_EST').iloc[-1]['GAME_DATE_EST']).days
    if not partidos_vis.empty:
        rest_V = (fecha_target - partidos_vis.sort_values('GAME_DATE_EST').iloc[-1]['GAME_DATE_EST']).days

    fatiga_L = 0.85 if rest_L <= 1 else 1.0
    fatiga_V = 0.85 if rest_V <= 1 else 1.0

    fuerza_base_L = calcular_fuerza_viva(id_local, fecha_target, df_games_completo)
    fuerza_base_V = calcular_fuerza_viva(id_visitante, fecha_target, df_games_completo)

    h2h_games = df_games_completo[
        (((df_games_completo['HOME_TEAM_ID'] == id_local) & (df_games_completo['VISITOR_TEAM_ID'] == id_visitante)) | 
         ((df_games_completo['HOME_TEAM_ID'] == id_visitante) & (df_games_completo['VISITOR_TEAM_ID'] == id_local))) & 
        (df_games_completo['GAME_DATE_EST'] < fecha_target)
    ].sort_values('GAME_DATE_EST').tail(5)

    pesos_h2h = [0.40, 0.30, 0.15, 0.10, 0.05]
    dominio_local, dominio_vis = 0.0, 0.0
    h2h_games_recent = h2h_games.iloc[::-1].reset_index()

    for i, row in h2h_games_recent.iterrows():
        if i >= len(pesos_h2h): break
        peso_actual = pesos_h2h[i]
        if row['HOME_TEAM_ID'] == id_local:
            if row['HOME_TEAM_WINS'] == 1: dominio_local += peso_actual
            else: dominio_vis += peso_actual
        else:
            if row['HOME_TEAM_WINS'] == 1: dominio_vis += peso_actual
            else: dominio_local += peso_actual

    mod_h2h_L = 1.0 + (dominio_local * 0.15)
    mod_h2h_V = 1.0 + (dominio_vis * 0.15)

    fuerza_final_L = fuerza_base_L * fatiga_L * mod_h2h_L
    fuerza_final_V = fuerza_base_V * fatiga_V * mod_h2h_V
    
    h2h_factor_localia = 1.1 if rest_L > rest_V else 0.9

    inputs = np.array([fuerza_final_L, fuerza_final_V, h2h_factor_localia])
    prob_local = sigmoide(np.dot(inputs, pesos))
    prob_vis = 1 - prob_local

    col1, col2 = st.columns(2)
    
    with col1:
        st.metric(label=f"Fuerza Ajustada {local_nombre} (Local)", value=f"{fuerza_final_L:.2f}", delta=f"H2H Mod: {mod_h2h_L:.2f}x")
        st.write(f"Días de descanso: {rest_L} días (Factor Fatiga: {fatiga_L}x)")
        st.progress(float(prob_local))
        st.subheader(f"Probabilidad: {prob_local:.2%}")

    with col2:
        st.metric(label=f"Fuerza Ajustada {visitante_nombre} (Visitante)", value=f"{fuerza_final_V:.2f}", delta=f"H2H Mod: {mod_h2h_V:.2f}x")
        st.write(f"Días de descanso: {rest_V} días (Factor Fatiga: {fatiga_V}x)")
        st.progress(float(prob_vis))
        st.subheader(f"Probabilidad: {prob_vis:.2%}")

    st.markdown("---")
    st.subheader("🎯 Evaluación de Resultados")
    
    ganador_predicho = local_nombre if prob_local > prob_vis else visitante_nombre
    
    partido_actual = partidos_cruzados[partidos_cruzados['GAME_DATE_EST'] == fecha_target].iloc[0]
    ganador_real_local = partido_actual['HOME_TEAM_WINS'] == 1
    ganador_real = local_nombre if ganador_real_local else visitante_nombre
    
    pts_L = int(partido_actual['PTS_home'])
    pts_V = int(partido_actual['PTS_away'])

    col_pred, col_real = st.columns(2)
    
    with col_pred:
        st.markdown("**Predicción del Consultor:**")
        if ganador_predicho == local_nombre:
            st.info(f"🏠 **{local_nombre}** es el favorito ({prob_local:.1%})")
        else:
            st.info(f"🚀 **{visitante_nombre}** es el favorito ({prob_vis:.1%})")
            
    with col_real:
        st.markdown("**Resultado en Pista:**")
        st.success(f"🏆 Ganador Real: {ganador_real} ({pts_L} - {pts_V})")
        
    if ganador_predicho == ganador_real:
        # Reemplazamos los globos por la función de inyección de fuegos artificiales
        mostrar_fuegos_artificiales()
    else:
        st.error(f"❌ **Predicción Fallida.** El partido real rompió las tendencias estadísticas calculadas.")