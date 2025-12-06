from flask import Flask, request, jsonify
from flask_cors import CORS
from tensorflow.keras.models import load_model
import xgboost as xgb
import pickle
import pandas as pd
import numpy as np
from tensorflow.keras.preprocessing.sequence import pad_sequences
import re

#Loading all the xgb models
ny_xgb_model = xgb.Booster()
ny_xgb_model.load_model("models/XGB_Models/ny_xgb_model.json")

la_xgb_model = xgb.Booster()
la_xgb_model.load_model("models/XGB_Models/la_xgb_model.json")

chi_xgb_model = xgb.Booster()
chi_xgb_model.load_model("models/XGB_Models/chi_xgb_model.json")

dal_xgb_model = xgb.Booster()
dal_xgb_model.load_model("models/XGB_Models/dal_xgb_model.json")

den_xgb_model = xgb.Booster()
den_xgb_model.load_model("models/XGB_Models/den_xgb_model.json")

xgb_model_map = {
    "NY": ny_xgb_model,
    "LA": la_xgb_model,
    "CHI": chi_xgb_model,
    "DAL": dal_xgb_model,
    "DEN": den_xgb_model
}

#Loading all the RNN models
CHI_model = load_model("models/CHI/CHI_model.keras")
DAL_model = load_model("models/DAL/DAL_model.keras")
DEN_model = load_model("models/DEN/DEN_model.keras")
LA_model = load_model("models/LA/LA_model.keras")
NY_model = load_model("models/NY/NY_model.keras")

rnn_model_map = {
    "CHI": CHI_model,
    "DAL": DAL_model,
    "DEN": DEN_model,
    "LA": LA_model,
    "NY": NY_model
}

with open(f"models/CHI/tokenizer.pkl", "rb") as f:
    chi_tokenizer = pickle.load(f)

with open(f"models/CHI/tokenizer.pkl", "rb") as f:
    dal_tokenizer = pickle.load(f)

with open(f"models/CHI/tokenizer.pkl", "rb") as f:
    den_tokenizer = pickle.load(f)

with open(f"models/CHI/tokenizer.pkl", "rb") as f:
    la_tokenizer = pickle.load(f)

with open(f"models/CHI/tokenizer.pkl", "rb") as f:
    ny_tokenizer = pickle.load(f)

tokenizer_map = {
    "CHI": chi_tokenizer,
    "DAL": dal_tokenizer,
    "DEN": den_tokenizer,
    "LA": la_tokenizer,
    "NY": ny_tokenizer
}

with open(f"models/CHI/dummy_cols.pkl", "rb") as f:
    chi_dummy_cols = pickle.load(f)

with open(f"models/CHI/dummy_cols.pkl", "rb") as f:
    dal_dummy_cols = pickle.load(f)

with open(f"models/CHI/dummy_cols.pkl", "rb") as f:
    den_dummy_cols = pickle.load(f)

with open(f"models/CHI/dummy_cols.pkl", "rb") as f:
    la_dummy_cols = pickle.load(f)

with open(f"models/CHI/dummy_cols.pkl", "rb") as f:
    ny_dummy_cols = pickle.load(f)

dummy_cols_map = {
    "CHI": chi_dummy_cols,
    "DAL": dal_dummy_cols,
    "DEN": den_dummy_cols,
    "LA": la_dummy_cols,
    "NY": ny_dummy_cols
}


with open("models/CHI/scaler.pkl", "rb") as f:
    chi_scaler = pickle.load(f)

with open("models/CHI/scaler.pkl", "rb") as f:
    dal_scaler = pickle.load(f)

with open("models/CHI/scaler.pkl", "rb") as f:
    den_scaler = pickle.load(f)

with open("models/CHI/scaler.pkl", "rb") as f:
    la_scaler = pickle.load(f)

with open("models/CHI/scaler.pkl", "rb") as f:
    ny_scaler = pickle.load(f)

scaler_map = {
    "CHI": chi_scaler,
    "DAL": dal_scaler,
    "DEN": den_scaler,
    "LA": la_scaler,
    "NY": ny_scaler
}



max_len = 30     
num_cols = [
    "minimum_nights", "number_of_reviews",
    "calculated_host_listings_count", "availability_365",
    "beds", "bedrooms", "accommodates",
    "review_scores_rating"
]

cat_cols = ["room_type", "host_is_superhost", "zip", "season"]

def clean_amenities(text):
    if pd.isna(text):
        return ""
    text = re.sub(r'\\u[0-9a-fA-F]{4}', ' ', text)
    text = re.sub(r'[\[\]",]', ' ', text)
    text = re.sub(r'[^a-zA-Z0-9 ]', ' ', text)
    text = re.sub(r'\s+', ' ', text).strip().lower()
    return text

def prepare_input(df, city):
    df["amenities_clean"] = df["amenities"].apply(clean_amenities)

    tokenizer = tokenizer_map.get(city, ny_tokenizer)
    dummy_cols = dummy_cols_map.get(city, ny_dummy_cols)
    scaler = scaler_map.get(city, ny_scaler)

    seqs = tokenizer.texts_to_sequences(df["amenities_clean"])
    X_text = pad_sequences(seqs, maxlen=max_len)

    numeric_df = df[num_cols].apply(pd.to_numeric, errors="coerce").fillna(0.0)
    X_num = scaler.transform(numeric_df.values.astype("float32"))

    df_cat = pd.get_dummies(df[cat_cols], drop_first=True)

    for col in dummy_cols:
        if col not in df_cat:
            df_cat[col] = 0

    df_cat = df_cat[dummy_cols]    
    X_cat = df_cat.values.astype("float32")

    return [X_text, X_num, X_cat]



app = Flask(__name__)
CORS(app) 

@app.route("/predict_RNN", methods=["POST"])
def predict_RNN():
    data = request.get_json()  

    df = pd.DataFrame([data])
    city = data.get("city", "").upper()
    model = rnn_model_map.get(city, NY_model)
    df = df.drop(columns=["city"])

    inputs = prepare_input(df)
    pred = float(round(model.predict(inputs)[0][0], 2))

    return jsonify({"prediction": pred})

@app.route("/predict_XGB", methods=["POST"])
def predict_XGB():
    data = request.get_json()  

    df = pd.DataFrame([data])

    print(df)
    city = data.get("city", "").upper()
    model = xgb_model_map.get(city, ny_xgb_model)

    df["private_room"] = (df["room_type"] == "Private room").astype(int)
    df["entire_home"] = (df["room_type"] == "Entire home/apt").astype(int)
    df["shared_room"] = (df["room_type"] == "Shared room").astype(int)
    df["hotel_room"] = (df["room_type"] == "Hotel room").astype(int)
    df["host_is_superhost"] = (df["host_is_superhost"] == "TRUE").astype(int)
 
    df = df.drop(columns="room_type")
    df = df.drop(columns=["city"])
    df = df.drop(columns=["zip"])
    print(df)
    
    df = df.apply(pd.to_numeric, errors='coerce')

    dmatrix = xgb.DMatrix(df)
    prediction = float(round(la_xgb_model.predict(dmatrix)[0], 2))

    print(prediction)
    return jsonify({"prediction": prediction})

if __name__ == '__main__':
    app.run()
