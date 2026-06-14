# CineScore – ניבוי דירוג סרטים

## הוראות הרצה

### התקנת סביבה וירטואלית
```
python -m venv venv
venv\Scripts\activate       # Windows
source venv/bin/activate    # Mac/Linux
pip install -r requirements.txt
```

### הרצת השרת
```
python api.py
```
השרת יעלה בכתובת: http://localhost:5000

## שדות הקלט
- primaryTitle – שם הסרט (אופציונלי)
- startYear – שנת יציאה (1888–2030)
- runtimeMinutes – אורך בדקות
- genres – ז'אנרים מופרדים בפסיק
- directors – מזהי IMDb של במאים (nm…)
- lead_actors_ids – מזהי IMDb של שחקנים (nm…)
