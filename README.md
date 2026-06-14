# CineScore – חיזוי דירוג סרטים

**קורס:** למידת מכונה | אוניברסיטת אריאל  
**צוות:** שיראל חדד ועינב ינון  
**חלק:** 3 מתוך 3 – אפליקציית Flask

---

## תיאור הפרויקט

אפליקציית Flask שעוטפת את מודל ה-Random Forest שפותח בחלק 2.  
מזינים פרמטרים של סרט (שנה, זמן ריצה, ז'אנרים ואופציונלית מזהי IMDb של הבמאי והשחקנים) ומקבלים תחזית לדירוג IMDb הממוצע – בזמן אמת, ללא רענון הדף.

---

## מבנה הקבצים

```
├── api.py                # שרת Flask – נקודות קצה GET / ו-POST /predict
├── assets_data_prep.py   # פונקציית prepare_data() וכל הטרנספורמרים מחלק 2 (ללא שינוי)
├── model.pkl             # ה-Pipeline המאומן מחלק 2
├── templates/
│   └── index.html        # ממשק המשתמש – טופס קלט והצגת תוצאה
├── requirements.txt      # תלויות Python עם גרסאות קבועות
└── README.md             # קובץ זה
```

---

## התקנה

### 1. יצירת סביבה וירטואלית

```bash
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate
```

### 2. התקנת ספריות

```bash
pip install -r requirements.txt
```

---

## הרצה

```bash
python api.py
```

פתחו את הדפדפן בכתובת **http://localhost:5000**

---

## שימוש באפליקציה

מלאו את שדות הטופס ולחצו על **Predict Rating**:

| שדה | תיאור | טווח ערכים |
|-----|--------|------------|
| Release Year | שנת יציאת הסרט | 1900 – 2030 |
| Runtime (minutes) | אורך הסרט בדקות | 1 – 600 |
| Genres | ז'אנרים (בחירה מרובה) | Action, Drama, … |
| Director IMDb ID(s) | מזהה הבמאי ב-IMDb (אופציונלי) | למשל `nm0000093` |
| Lead Actor IMDb ID(s) | מזהי שחקנים ראשיים, מופרדים בפסיק (אופציונלי) | למשל `nm0000375, nm0000129` |

מזהי IMDb נמצאים ב-URL של כל אדם באתר:  
`https://www.imdb.com/name/nm0000093/` ← המזהה הוא **nm0000093**

---

## נקודות קצה (API)

| Method | Endpoint | תיאור |
|--------|----------|--------|
| GET | `/` | מחזיר את דף ה-HTML |
| POST | `/predict` | מקבל JSON, מחזיר `{"predicted_rating": <מספר>}` |

### דוגמה לקריאה

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

### דוגמה לתגובה

```json
{ "predicted_rating": 8.4 }
```

---

## חברי הצוות

| שם | מספר ת.ז. |
|----|-----------|
| שיראל חדד | 212546017 |
| עינב ינון | 211415351 |
