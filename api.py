import os
import sys
import __main__
import joblib
import numpy as np
import pandas as pd
from flask import Flask, request, jsonify, render_template

import assets_data_prep
from assets_data_prep import prepare_data

# The model was pickled from a Jupyter notebook where all classes lived in
# __main__.  We expose them there so joblib can deserialise the pipeline.
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

MODEL_PATH = os.path.join(os.path.dirname(__file__), "trained_model.pkl")
model = joblib.load(MODEL_PATH)


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/predict", methods=["POST"])
def predict():
    data = request.get_json(force=True)

    required = ["startYear", "runtimeMinutes", "genres"]
    missing = [f for f in required if f not in data or data[f] in (None, "", [])]
    if missing:
        return jsonify({"error": f"Missing required fields: {', '.join(missing)}"}), 400

    try:
        start_year = float(data["startYear"])
        runtime = float(data["runtimeMinutes"])
    except (ValueError, TypeError):
        return jsonify({"error": "startYear and runtimeMinutes must be numeric values"}), 400

    genres = data.get("genres", [])
    if isinstance(genres, str):
        genres = [g.strip() for g in genres.split(",") if g.strip()]

    lead_actors = data.get("lead_actors_ids", [])
    if isinstance(lead_actors, str):
        lead_actors = [a.strip() for a in lead_actors.split(",") if a.strip()]

    directors = data.get("directors", "")
    if isinstance(directors, list):
        directors = ",".join(directors)

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
        rating = max(1.0, min(10.0, rating))
        return jsonify({"predicted_rating": rating})
    except Exception as e:
        return jsonify({"error": f"Prediction failed: {str(e)}"}), 500


if __name__ == "__main__":
    app.run(debug=True)
