import re
import ast
import bisect
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin


# --- פונקציות ניקוי ---

def _clean_year(val):
    # מחזיר שנה תקינה בת 4 ספרות, אחרת NaN
    if pd.isna(val):
        return np.nan
    try:
        year = int(float(val))
        if re.fullmatch(r'\d{4}', str(year)):
            return year
        return np.nan
    except (ValueError, TypeError):
        return np.nan


def _clean_runtime(val):
    # רק מספר, לא טקסט
    if pd.isna(val):
        return np.nan
    try:
        return float(val)
    except (ValueError, TypeError):
        return np.nan


def _clean_rating(val):
    # ערך תקין בין 1.0 ל-10.0
    if pd.isna(val):
        return np.nan
    try:
        r = float(val)
        return r if 1.0 <= r <= 10.0 else np.nan
    except (ValueError, TypeError):
        return np.nan


def _clean_genres(val):
    # מחזיר רשימת ז'אנרים נקייה
    if pd.isna(val):
        return []
    s = str(val).strip()
    if re.match(r'^(\\N|N/A|NA|\[\])$', s, re.IGNORECASE):
        return []
    try:
        parsed = ast.literal_eval(s)
        items = ([str(g).strip() for g in parsed]
                 if isinstance(parsed, list) else [str(parsed).strip()])
    except (ValueError, SyntaxError):
        items = re.split(r'[,|;]', s)
    cleaned = []
    for item in items:
        g = item.strip().title()
        if g and re.match(r'^[A-Za-z\- ]{2,30}$', g):
            if not re.match(r'^(N/A|\\N|Nan|None|Na)$', g, re.IGNORECASE):
                cleaned.append(g)
    return cleaned


def _clean_actors(val):
    # מחזיר רק מזהים תקינים שמתחילים ב-nm
    if pd.isna(val):
        return []
    s = str(val).strip()
    if re.match(r'^(\\N|\[\]|N/A)$', s, re.IGNORECASE):
        return []
    return re.findall(r'nm\d{7,8}', s)


def _clean_director(val):
    # מחזיר מזהה במאי תקין או NaN
    if pd.isna(val):
        return np.nan
    s = str(val).strip()
    if re.match(r'^(\\N|N/A|NA|nan|none)$', s, re.IGNORECASE):
        return np.nan
    matches = re.findall(r'nm\d{7,8}', s)
    if not matches:
        return np.nan
    return ','.join(matches)


# --- פונקציית עזר להמרה לרשימה ---

def _safe_list(val):
    if isinstance(val, list):
        return val
    if pd.isna(val):
        return []
    try:
        parsed = ast.literal_eval(str(val))
        return parsed if isinstance(parsed, list) else [parsed]
    except Exception:
        return [x.strip() for x in str(val).split(',') if x.strip()]


# --- פיצ'רים סטטיים פשוטים ---

def add_static_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df['num_genres'] = df['genres'].apply(lambda g: len(_safe_list(g)))
    df['is_too_short'] = (df['runtimeMinutes'] < 75).astype(float)
    df['is_too_long'] = (df['runtimeMinutes'] > 150).astype(float)
    return df


# --- טרנספורמרים ---

# חישוב יחס אורך הסרט לחציון סרטים מאותו ז'אנר - רק לפי סרטים היסטוריים
class RelativeRuntimeTransformer(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        df = X[['genres', 'startYear', 'runtimeMinutes']].copy()
        df['_first_genre'] = df['genres'].apply(
            lambda g: _safe_list(g)[0] if _safe_list(g) else 'Unknown')
        df = df.dropna(subset=['startYear', 'runtimeMinutes']).sort_values('startYear')
        self.genre_history_ = {}
        for genre, group in df.groupby('_first_genre'):
            self.genre_history_[genre] = {
                'years': group['startYear'].values,
                'runtimes': group['runtimeMinutes'].values,
            }
        return self

    def transform(self, X):
        X = X.copy()
        X['_first_genre'] = X['genres'].apply(
            lambda g: _safe_list(g)[0] if _safe_list(g) else 'Unknown')

        def calc(row):
            genre = row['_first_genre']
            current_year = row['startYear']
            if genre not in self.genre_history_ or pd.isna(current_year):
                return np.nan
            years_array = self.genre_history_[genre]['years']
            idx = bisect.bisect_left(years_array, current_year)
            if idx == 0:
                return np.nan
            past_runtimes = self.genre_history_[genre]['runtimes'][:idx]
            median = np.median(past_runtimes)
            if pd.isna(median) or median == 0:
                return np.nan
            return row['runtimeMinutes'] / median

        X['relative_runtime_by_genre'] = X.apply(calc, axis=1)
        return X.drop(columns=['_first_genre'])


# ניסיון השחקן הכי מנוסה בסרט - לוקחים מקסימום ולא ממוצע כדי לא לדלל את הכוכב
class ActorExpTransformer(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        df = X[['lead_actors_ids', 'startYear']].dropna().copy()
        df['lead_actors_ids'] = df['lead_actors_ids'].apply(_safe_list)
        df = df.explode('lead_actors_ids').sort_values('startYear')
        self.actor_years_ = df.groupby('lead_actors_ids')['startYear'].apply(list).to_dict()
        return self

    def transform(self, X):
        X = X.copy()

        def calc(row):
            actors = _safe_list(row['lead_actors_ids'])
            year = row['startYear']
            if not actors or pd.isna(year):
                return pd.Series({'max_actors_past_experience': 0.0, 'is_new_actors': 1.0})
            exp = [bisect.bisect_left(self.actor_years_.get(a, []), year) for a in actors]
            max_exp = float(max(exp))
            return pd.Series({
                'max_actors_past_experience': max_exp,
                'is_new_actors': 1.0 if max_exp == 0 else 0.0,
            })

        new_cols = X.apply(calc, axis=1)
        X = pd.concat([X, new_cols], axis=1)
        return X


# ממוצע דירוגי הבמאי בסרטים קודמים - רק סרטים שיצאו לפני השנה הנוכחית
class DirectorAvgTransformer(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        orig_df = X[['genres', 'startYear']].copy()
        orig_df['averageRating'] = y
        orig_df['_first_genre'] = orig_df['genres'].apply(
            lambda g: _safe_list(g)[0] if _safe_list(g) else 'Unknown')
        orig_df = orig_df.dropna(subset=['startYear', 'averageRating']).sort_values('startYear')

        self.global_years_ = orig_df['startYear'].values
        self.global_ratings_ = orig_df['averageRating'].values

        self.genre_history_ = {}
        for g, group in orig_df.groupby('_first_genre'):
            self.genre_history_[g] = {
                'years': group['startYear'].values,
                'ratings': group['averageRating'].values,
            }

        df = X[['directors', 'genres', 'startYear']].copy()
        df['averageRating'] = y
        df['_first_genre'] = df['genres'].apply(
            lambda g: _safe_list(g)[0] if _safe_list(g) else 'Unknown')
        df['directors'] = df['directors'].apply(_safe_list)
        df = df.explode('directors').dropna(
            subset=['directors', 'startYear', 'averageRating']).sort_values('startYear')

        self.dir_history_ = {}
        for d, group in df.groupby('directors'):
            self.dir_history_[d] = group[['startYear', 'averageRating', '_first_genre']].copy()

        return self

    def transform(self, X):
        X = X.copy()
        X['_first_genre'] = X['genres'].apply(
            lambda g: _safe_list(g)[0] if _safe_list(g) else 'Unknown')

        def calc(row):
            year = row['startYear']
            genre = row['_first_genre']
            current_directors = _safe_list(row['directors'])

            if pd.isna(year):
                return pd.Series({
                    'director_past_avg': np.nan,
                    'is_new_director': 1.0,
                    'director_genre_past_avg': np.nan,
                })

            idx_global = bisect.bisect_left(self.global_years_, year)
            global_past_avg = (np.mean(self.global_ratings_[:idx_global])
                               if idx_global > 0 else np.nan)

            genre_past_avg = global_past_avg
            if genre in self.genre_history_:
                idx_genre = bisect.bisect_left(self.genre_history_[genre]['years'], year)
                if idx_genre > 0:
                    genre_past_avg = np.mean(self.genre_history_[genre]['ratings'][:idx_genre])

            dir_past_ratings = []
            dir_genre_past_ratings = []

            for d in current_directors:
                if d in self.dir_history_:
                    hist = self.dir_history_[d]
                    # סינון קפדני - רק סרטים משנים קודמות, למניעת data leakage
                    past_movies = hist[hist['startYear'] < year]
                    if not past_movies.empty:
                        dir_past_ratings.extend(past_movies['averageRating'].tolist())
                        genre_movies = past_movies[past_movies['_first_genre'] == genre]
                        if not genre_movies.empty:
                            dir_genre_past_ratings.extend(genre_movies['averageRating'].tolist())

            if dir_past_ratings:
                res_dir_avg = np.mean(dir_past_ratings)
                res_is_new_dir = 0.0
            else:
                # במאי חדש - נשתמש בממוצע הגלובלי כברירת מחדל
                res_dir_avg = global_past_avg
                res_is_new_dir = 1.0

            res_dir_genre_avg = (np.mean(dir_genre_past_ratings)
                                 if dir_genre_past_ratings else genre_past_avg)

            return pd.Series({
                'director_past_avg': res_dir_avg,
                'is_new_director': res_is_new_dir,
                'director_genre_past_avg': res_dir_genre_avg,
            })

        new_cols = X.apply(calc, axis=1)
        X = pd.concat([X, new_cols], axis=1)
        return X.drop(columns=['_first_genre'])


# --- פונקציית prepare_data הראשית ---

def prepare_data(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    # אם עמודת הבמאים לא קיימת - ננסה לטעון מקובץ הצוות
    if 'directors' not in df.columns:
        try:
            _crew = pd.read_csv(
                "title.crew.tsv.gz", sep='\t',
                usecols=['tconst', 'directors'], low_memory=False,
            )
            df = pd.merge(df, _crew, on='tconst', how='left')
        except Exception:
            df['directors'] = np.nan

    # ניקוי כל העמודות
    df['startYear'] = df['startYear'].apply(_clean_year)
    df['runtimeMinutes'] = df['runtimeMinutes'].apply(_clean_runtime)
    df['genres'] = df['genres'].apply(_clean_genres)
    df['lead_actors_ids'] = df['lead_actors_ids'].apply(_clean_actors)
    df['directors'] = df['directors'].apply(_clean_director)

    df = add_static_features(df)

    # הסרת שורות ללא דירוג - רק בשלב האימון
    if 'averageRating' in df.columns:
        df['averageRating'] = df['averageRating'].apply(_clean_rating)
        df = df.dropna(subset=['averageRating']).reset_index(drop=True)

    # הסרת עמודות שלא בשימוש במודל
    COLS_TO_DROP = ['numVotes', 'BoxOffice', 'plot', 'Language', 'Country', 'budget']
    df = df.drop(columns=[c for c in COLS_TO_DROP if c in df.columns])

    return df
