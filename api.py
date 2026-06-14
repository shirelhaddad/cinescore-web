from flask import Flask, request, jsonify, render_template
import pandas as pd
import numpy as np
import joblib
import os
from assets_data_prep import (
    RelativeRuntimeTransformer, ActorExpTransformer, DirectorAvgTransformer,
    _safe_list, _clean_year, _clean_runtime, _clean_rating,
    _clean_genres, _clean_actors, _clean_director,
    add_static_features, prepare_data,
)

app = Flask(__name__)

_MODEL_PATHS = ['trained_model.pkl', 'model.pkl']
model = None
for _path in _MODEL_PATHS:
    if os.path.exists(_path):
        model = joblib.load(_path)
        print(f"Model loaded from: {_path}")
        break

if model is None:
    raise FileNotFoundError("Could not find trained_model.pkl or model.pkl")

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json(force=True)
        required_fields = ['startYear', 'runtimeMinutes', 'genres', 'directors', 'lead_actors_ids']
        missing = [f for f in required_fields if f not in data or str(data[f]).strip() == '']
        if missing:
            return jsonify({'error': f"שדות חסרים: {', '.join(missing)}"}), 400
        try:
            start_year = int(data['startYear'])
            runtime    = float(data['runtimeMinutes'])
        except (ValueError, TypeError):
            return jsonify({'error': 'שנת יציאה ואורך הסרט חייבים להיות מספרים'}), 400
        if not (1888 <= start_year <= 2030):
            return jsonify({'error': 'שנת יציאה חייבת להיות בין 1888 ל-2030'}), 400
        if runtime <= 0:
            return jsonify({'error': 'אורך הסרט חייב להיות גדול מ-0'}), 400
        df = pd.DataFrame([{
            'tconst': 'tt0000000',
            'primaryTitle': data.get('primaryTitle', 'Unknown'),
            'startYear': start_year,
            'runtimeMinutes': runtime,
            'genres': data['genres'],
            'directors': data['directors'],
            'lead_actors_ids': data['lead_actors_ids'],
        }])
        df_prep = prepare_data(df)
        raw_pred = float(model.predict(df_prep)[0])
        predicted = round(max(1.0, min(10.0, raw_pred)), 1)
        return jsonify({'predicted_rating': predicted})
    except KeyError as e:
        return jsonify({'error': f'שדה חסר: {e}'}), 400
    except Exception as e:
        return jsonify({'error': f'שגיאה פנימית: {str(e)}'}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
