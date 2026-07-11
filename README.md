
# 🏀 NBA Predictive Analytics & Live Consultant

<div align="center">
  <a href="https://nba-analytics-proyect-998y4j5icgddxcvxjdziwc.streamlit.app/">
    <img src="https://img.shields.io/badge/Status-Live_en_Streamlit-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white" alt="Streamlit App">
  </a>
  <br>
  <img src="https://img.shields.io/badge/Python-3776AB?style=flat-square&logo=python&logoColor=white" alt="Python">
  <img src="https://img.shields.io/badge/Pandas-150458?style=flat-square&logo=pandas&logoColor=white" alt="Pandas">
  <img src="https://img.shields.io/badge/NumPy-013243?style=flat-square&logo=numpy&logoColor=white" alt="NumPy">
  <img src="https://img.shields.io/badge/Machine%20Learning-Sigmoid_Model-black?style=flat-square" alt="ML">
</div>

<br>

> Un modelo predictivo y dashboard interactivo diseñado para estimar probabilidades de victoria en partidos de la NBA mediante la ponderación matemática de la fuerza viva (EWMA), fatiga acumulada y el dominio histórico cara a cara (H2H).

### 🚀 **[Prueba el Consultor Interactivo Aquí](https://nba-analytics-proyect-998y4j5icgddxcvxjdziwc.streamlit.app/)**

---

## 📌 Visión General del Proyecto

Este proyecto procesa datos históricos de la NBA (temporadas 2018-2019 a 2021-2022) para alimentar un modelo probabilístico. En lugar de depender de estadísticas estáticas, el algoritmo evalúa el momento dinámico de los equipos, ajustando su "Fuerza Viva" a través de multiplicadores de fatiga (días de descanso) y un decaimiento exponencial de enfrentamientos directos previos.

### ⚙️ Características Principales

* **Motor de Predicción Dinámico:** Implementación de cálculo de Fuerza Ajustada combinando promedios móviles exponenciales (EWMA).
* **Ajuste de Fatiga y Localía:** Modificadores matemáticos basados en el calendario de viajes y días de descanso entre partidos.
* **Auditoría de Resultados (Backtesting en Vivo):** El dashboard permite comparar la predicción exacta del modelo contra el resultado real que ocurrió en la pista.
* **Interfaz Interactiva:** Desplegado en Streamlit, ofreciendo una experiencia fluida sin necesidad de ejecutar código localmente.

---


   git clone [https://github.com/tu-usuario/NBA-analytics-proyect.git](https://github.com/tu-usuario/NBA-analytics-proyect.git)
   cd NBA-analytics-proyect
