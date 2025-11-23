from flask import Flask, request, jsonify
from models.prediccion import preparar_datos_y_predecir
import json

app = Flask(__name__)

with open("models/modelo_regresion_logistica_info.json", "r") as f:
    FEATURES_ESPERADAS = json.load(f)["features"]

pacientes_db = []

@app.route('/', methods=['GET'])
def home():
    return jsonify({
        'mensaje': 'API de predicción de hepatitis activa.',
        'endpoints': ['/info', '/pacientes (GET/POST)', '/predict']
    })

@app.route('/info', methods=['GET'])
def info():
    return jsonify({
        'descripcion': 'Formato esperado por la API',
        'campos_requeridos': FEATURES_ESPERADAS
    })

@app.route('/pacientes', methods=['GET'])
def get_patients():
    return jsonify({
        'total': len(pacientes_db),
        'pacientes': pacientes_db
    })

@app.route('/pacientes', methods=['POST'])
def add_patient():
    datos = request.get_json()

    # Validación
    faltantes = [f for f in FEATURES_ESPERADAS if f not in datos]
    if faltantes:
        return jsonify({
            'error': 'Faltan campos en el JSON',
            'campos_faltantes': faltantes
        }), 400
    pacientes_db.append(datos)
    return jsonify({
        'mensaje': 'Paciente agregado correctamente',
        'total_actual': len(pacientes_db)
    })

@app.route('/predict', methods=['POST'])
def predict():
    paciente_nuevo = request.get_json()
    # Validación campos
    faltantes = [f for f in FEATURES_ESPERADAS if f not in paciente_nuevo]
    if faltantes:
        return jsonify({
            'error': 'Faltan campos en el JSON',
            'campos_faltantes': faltantes
        }), 400
    try:
        resultado = preparar_datos_y_predecir(paciente_nuevo)
        return jsonify(resultado)
    except Exception as e:
        return jsonify({'error': str(e)}), 400


if __name__ == '__main__':
    app.run(debug=True)
