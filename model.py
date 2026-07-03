# ============================================
# MODEL FILE (model.py)
# ============================================

import pandas as pd
import numpy as np
from xgboost import XGBClassifier
from sklearn.preprocessing import LabelEncoder
from xgboost import XGBClassifier


def load_and_preprocess():
    df = pd.read_csv('Crime_Incidents_in_2024.csv')

    df['START_DATE'] = pd.to_datetime(df['START_DATE'], errors='coerce')

    # Drop important missing values
    df = df.dropna(subset=['START_DATE', 'LATITUDE', 'LONGITUDE', 'NEIGHBORHOOD_CLUSTER'])

    # Feature extraction
    df['MONTH'] = df['START_DATE'].dt.month_name()
    df['DAY_OF_WEEK'] = df['START_DATE'].dt.day_name()
    df['HOUR'] = df['START_DATE'].dt.hour

    df['IS_WEEKEND'] = df['DAY_OF_WEEK'].isin(['Saturday', 'Sunday']).astype(int)

    df['TIME_BIN'] = pd.cut(
        df['HOUR'],
        bins=[0, 6, 12, 18, 24],
        labels=['Night', 'Morning', 'Afternoon', 'Evening']
    )

    df = df.dropna(subset=['TIME_BIN'])

    # Remove outliers
    numeric_cols = ['LATITUDE', 'LONGITUDE']
    z_scores = (df[numeric_cols] - df[numeric_cols].mean()) / df[numeric_cols].std()
    df = df[(np.abs(z_scores) < 3).all(axis=1)]
    
    top_crimes = df['OFFENSE'].value_counts().nlargest(5).index
    df = df[df['OFFENSE'].isin(top_crimes)]
    df['IS_NIGHT'] = (df['HOUR'] < 6).astype(int)
    df['IS_PEAK_HOUR'] = df['HOUR'].isin([8,9,18,19]).astype(int)

    return df


from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

def train_model(df):

    features = [
        'WARD',
        'SHIFT',
        'MONTH',
        'DAY_OF_WEEK',
        'HOUR',
        'IS_WEEKEND',
        'TIME_BIN',
        'NEIGHBORHOOD_CLUSTER'
    ]

    df_model = df[features + ['OFFENSE']].copy()
    df_model = df_model.dropna()

    le_dict = {}

    for col in features:
        le = LabelEncoder()
        df_model[col] = le.fit_transform(df_model[col].astype(str))
        le_dict[col] = le

    target_le = LabelEncoder()
    df_model['OFFENSE'] = target_le.fit_transform(df_model['OFFENSE'])

    X = df_model[features]
    y = df_model['OFFENSE']

    # ============================
    # STEP 6: Train-Test Split
    # ============================
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # ============================
    # STEP 5: Class Imbalance Handling
    # ============================
    

    model = XGBClassifier(
    n_estimators=300,
    learning_rate=0.1,
    max_depth=6,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=42,
    use_label_encoder=False,
    eval_metric='mlogloss'
)

    model.fit(X_train, y_train)

    # ============================
    # STEP 6: Evaluation
    # ============================
    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)

    print(f"Model Accuracy: {acc:.2f}")

    return model, le_dict, target_le


def predict_crime(model, le_dict, target_le, input_data):

    # Ensure correct feature order
    features = list(le_dict.keys())

    # Create dataframe with correct columns
    input_df = pd.DataFrame([input_data])

    # Reorder columns to match training
    input_df = input_df[features]

    # Encode safely
    for col in features:
        val = str(input_df[col][0])

        if val in le_dict[col].classes_:
            input_df[col] = le_dict[col].transform([val])
        else:
            input_df[col] = 0  # fallback

    # Predict
    pred = model.predict(input_df)
    predicted_crime = target_le.inverse_transform(pred)

    return predicted_crime[0]


def predict_crime_proba(model, le_dict, target_le, input_data):
    # Ensure correct feature order
    features = list(le_dict.keys())

    # Create dataframe with correct columns
    input_df = pd.DataFrame([input_data])

    # Reorder columns to match training
    input_df = input_df[features]

    # Encode safely
    for col in features:
        val = str(input_df[col][0])

        if val in le_dict[col].classes_:
            input_df[col] = le_dict[col].transform([val])
        else:
            input_df[col] = 0  # fallback

    # Predict probability
    proba = model.predict_proba(input_df)[0]
    
    # Map probability to classes
    classes = target_le.classes_
    class_probs = {cls: float(prob) for cls, prob in zip(classes, proba)}
    
    # Sort class probabilities descending
    sorted_probs = dict(sorted(class_probs.items(), key=lambda item: item[1], reverse=True))
    
    return sorted_probs