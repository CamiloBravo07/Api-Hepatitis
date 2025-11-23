import joblib
import pandas as pd

def preparar_datos_y_predecir(paciente_nuevo,ruta_scaler='models/scaler.pkl',ruta_modelo='models/modelo_regresion_logistica.pkl'):

    # Cargar scaler y modelo
    scaler = joblib.load(ruta_scaler)
    modelo = joblib.load(ruta_modelo)
    df_nuevo = pd.DataFrame([paciente_nuevo])
    df_escalado = scaler.transform(df_nuevo)
    
    # Probabilidades
    proba = modelo.predict_proba(df_escalado)[0]
    prob_muere = round(proba[0] * 100, 2)
    prob_vive  = round(proba[1] * 100, 2)

    # Determinar estado según la probabilidad mayor
    estado = "Vive" if prob_vive > prob_muere else "Muere"

    # Retornar predicción estructurada
    return {
        "estado": estado,
        "probabilidad_vive": prob_vive,
        "probabilidad_muere": prob_muere
    }
