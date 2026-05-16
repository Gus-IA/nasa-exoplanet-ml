# 🌌 Exoplanet ML — Clasificación con datos de NASA Kepler

Proyecto de machine learning que utiliza datos reales de la NASA Exoplanet Archive para clasificar si una señal detectada corresponde a un exoplaneta confirmado o a una falsa detección.

---

## 🚀 Descripción del proyecto

Este proyecto aplica técnicas de machine learning clásico a datos astronómicos reales de la misión Kepler de la NASA.

El objetivo es clasificar las señales en:

- ✅ Exoplanetas confirmados  
- ❌ Falsos positivos  

---

## 📡 Dataset

Los datos provienen del catálogo KOI (Kepler Object of Interest):

https://exoplanetarchive.ipac.caltech.edu/

Incluye características como:
- Periodo orbital  
- Profundidad del tránsito  
- Radio del planeta  
- Parámetros de la estrella  

---

## 🧠 Enfoque de Machine Learning

Este proyecto utiliza machine learning clásico:

- Random Forest Classifier  
- XGBoost Classifier  
- Preprocesamiento con Scikit-learn  
- Imputación de valores nulos (mediana)  
- Análisis de importancia de variables  

---

## 📊 Flujo del proyecto

1. Carga del dataset desde NASA  
2. Limpieza y preprocesamiento de datos  
3. Selección de variables numéricas  
4. Tratamiento de valores nulos  
5. Entrenamiento de modelos  
6. Evaluación de resultados  
7. Análisis de importancia de variables  

---

## 📈 Resultados

El modelo consigue un buen rendimiento en la clasificación de candidatos a exoplanetas.

Conclusiones principales:
- Las variables orbitales son muy predictivas  
- Los modelos basados en árboles funcionan muy bien  
- La importancia de variables ayuda a interpretar el modelo  

---

## 🛠️ Tecnologías utilizadas

- Python  
- Pandas  
- NumPy  
- Scikit-learn  
- XGBoost  

---

## 🌠 ¿Por qué este proyecto?

Los datos astronómicos son complejos y masivos. El machine learning ayuda a:

- Detectar exoplanetas de forma más eficiente  
- Reducir falsos positivos  
- Encontrar patrones en datos del espacio  

---

## 🔮 Mejoras futuras

- Validación cruzada  
- Explicabilidad con SHAP  
- Dashboard interactivo con Streamlit  

---

🧑‍💻 Autor

Desarrollado por Gus como parte de su aprendizaje en Python e IA.
