import os
import __main__
import joblib
import numpy as np
import pandas as pd
from flask import Flask, request, jsonify, send_file

import assets_data_prep
from assets_data_prep import prepare_data

# המודל נשמר מתוך הנוטבוק, כך שהמחלקות המותאמות רשומות תחת __main__
# צריך לחשוף אותן שם כדי שה-pipeline ייטען בהצלחה
for _name in (
    "RelativeRuntimeTransformer",
    "ActorExpTransformer",
    "DirectorAvgTransformer",
    "_safe_list",
    "_clean_year",
    "_clean_runtime",
    "_clean_rating",
    "_clean_genres",
    "_clean_actors",
    "_clean_director",
    "add_static_features",
):
    setattr(__main__, _name, getattr(assets_data_prep, _name))

app = Flask(__name__)

# טעינת המודל פעם אחת בהפעלת השרת
MODEL_PATH = os.path.join(os.path.dirname(__file__), "trained_model.pkl")
model = joblib.load(MODEL_PATH)


@app.route("/")
def index():
    return send_file("index.html")


@app.route("/predict", methods=["POST"])
def predict():
    data = request.get_json(force=True)

    # בדיקה שכל השדות החובה קיימים
    required = ["startYear", "runtimeMinutes", "genres"]
    missing = [f for f in required if f not in data or data[f] in (None, "", [])]
    if missing:
        return jsonify({"error": f"שדות חסרים: {', '.join(missing)}"}), 400

    # בדיקה שהשנה וזמן הריצה הם מספרים
    try:
        start_year = float(data["startYear"])
        runtime = float(data["runtimeMinutes"])
    except (ValueError, TypeError):
        return jsonify({"error": "שנה וזמן ריצה חייבים להיות מספרים"}), 400

    genres = data.get("genres", [])
    if isinstance(genres, str):
        genres = [g.strip() for g in genres.split(",") if g.strip()]

    lead_actors = data.get("lead_actors_ids", [])
    if isinstance(lead_actors, str):
        lead_actors = [a.strip() for a in lead_actors.split(",") if a.strip()]

    directors = data.get("directors", "")
    if isinstance(directors, list):
        directors = ",".join(directors)

    # בניית שורה אחת עם נתוני הסרט
    row = {
        "startYear": start_year,
        "runtimeMinutes": runtime,
        "genres": str(genres),
        "lead_actors_ids": str(lead_actors),
        "directors": directors if directors else np.nan,
    }

    try:
        df = pd.DataFrame([row])
        df_prepared = prepare_data(df)
        prediction = model.predict(df_prepared)
        rating = round(float(prediction[0]), 2)
        # הגבלת הדירוג לטווח תקין
        rating = max(1.0, min(10.0, rating))
        return jsonify({"predicted_rating": rating})
    except Exception as e:
        return jsonify({"error": f"שגיאה בחיזוי: {str(e)}"}), 500


if __name__ == "__main__":
    app.run(debug=True)
