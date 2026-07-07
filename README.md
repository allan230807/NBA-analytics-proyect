# 🏀 NBA End-to-End Analytics Pipeline

![Python](https://img.shields.io/badge/Python-3.10+-blue?style=for-the-badge&logo=python&logoColor=white)
![Pandas](https://img.shields.io/badge/Pandas-2.0+-150458?style=for-the-badge&logo=pandas&logoColor=white)
![Seaborn](https://img.shields.io/badge/Seaborn-Statistical-4c1c24?style=for-the-badge)
![Status](https://img.shields.io/badge/Status-In_Development-orange?style=for-the-badge)

 análisis de datos utilizando el conjunto de datos relacionales de la NBA (Kaggle). Este proyecto abarca desde la ingesta de archivos crudos conectados vía API/GitHub, pasando por una limpieza radical y cruce relacional de tablas, hasta la visualización ejecutiva de métricas de rendimiento.

---

## 🗺️ Flujo de Arquitectura de Datos



```text
[ Data Cruda: Kaggle CSVs ] 
            ⬇
 [ Auditoría Inicial: .info() & .isna() ] 
            ⬇
  [ Pipeline de Limpieza Radical (Pandas) ]  <-- Remoción de duplicados por claves, tipado e histogramas
            ⬇
   [ Cruce Relacional: pd.merge() ]         <-- Conexión de IDs de Juegos, Jugadores y Equipos
            ⬇
    [ Agregación Avanzada: .groupby() ]
            ⬇
     [ Dashboard Exec: Seaborn Charts ]
