# CineScore – Movie Rating Predictor

**Course:** Machine Learning | Ariel University  
**Team:** Shirel Haddad & Einav Yinon  
**Part:** 3 / 3 – Flask Web Application

---

## Project Description

A Flask web application that wraps the Random-Forest model trained in Part 2.  
Enter a movie's release year, runtime, genres, and optionally IMDb IDs for the
director and lead actors — the app returns an estimated IMDb average rating in
real time, with no page reload.

---

## File Structure

```
├── api.py                # Flask server (GET / and POST /predict)
├── assets_data_prep.py   # prepare_data() and all transformer classes (unchanged from Part 2)
├── model.pkl             # Trained Random-Forest pipeline from Part 2
├── templates/
│   └── index.html        # Front-end: input form + live prediction display
├── requirements.txt      # Python dependencies with pinned versions
└── README.md             # This file
```

---

## Setup

### 1. Create a virtual environment

```bash
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

---

## Running the Server

```bash
python api.py
```

Open your browser at **http://localhost:5000**

---

## Using the App

Fill in the form fields and click **Predict Rating**:

| Field | Description | Expected values |
|-------|-------------|-----------------|
| Release Year | Year the movie was / will be released | 1900 – 2030 |
| Runtime (minutes) | Total runtime of the movie | 1 – 600 |
| Genres | One or more genres (checkboxes) | Action, Drama, … |
| Director IMDb ID(s) | Optional – improves accuracy | e.g. `nm0000093` |
| Lead Actor IMDb ID(s) | Optional – comma-separated | e.g. `nm0000375, nm0000129` |

IMDb IDs can be found in any person's IMDb URL:  
`https://www.imdb.com/name/nm0000093/` → ID is **nm0000093**

---

## API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/` | Returns the HTML page |
| POST | `/predict` | Accepts JSON, returns `{"predicted_rating": <float>}` |

### Example request

```bash
curl -X POST http://localhost:5000/predict \
  -H "Content-Type: application/json" \
  -d '{
    "startYear": 2019,
    "runtimeMinutes": 181,
    "genres": ["Action", "Adventure", "Drama"],
    "directors": "nm0751577",
    "lead_actors_ids": ["nm0000375", "nm0226987"]
  }'
```

### Example response

```json
{ "predicted_rating": 8.4 }
```

---

## Team Members

| Name | ID |
|------|----|
| Shirel Haddad | 212546017 |
| Einav Yinon | 211415351 |
