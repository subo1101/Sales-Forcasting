import streamlit as st
import pandas as pd
import pickle

# ==============================
# Load models
# ==============================
#rf_orders  = pickle.load(open("rf_orders.pkl","rb"))
xgb_orders = pickle.load(open("xgb_orders.pkl","rb"))
lgbm_orders = pickle.load(open("lgbm_orders.pkl","rb"))

#rf_sales  = pickle.load(open("rf_sales.pkl","rb"))
xgb_sales = pickle.load(open("xgb_sales.pkl","rb"))
lgbm_sales = pickle.load(open("lgbm_sales.pkl","rb"))

meta_orders = pickle.load(open("meta_orders.pkl","rb"))
meta_sales  = pickle.load(open("meta_sales.pkl","rb"))

train_columns = pickle.load(open("train_columns.pkl","rb"))

st.title("📊 Orders & Sales Prediction")

store_id = st.number_input("Store ID", min_value=1)
store_type = st.selectbox("Store Type", ["S1","S2","S3","S4"])
location_type = st.selectbox("Location Type", ["L1","L2","L3"])
region_code = st.selectbox("Region Code", ["R1","R2","R3","R4"])
date = st.date_input("Date")
holiday = st.selectbox("Holiday", [0,1])
discount = st.selectbox("Discount", [0,1])

def preprocess():
    df = pd.DataFrame({
        "Store_id":[store_id],
        "Store_Type":[store_type],
        "Location_Type":[location_type],
        "Region_Code":[region_code],
        "Date":[date],
        "Holiday":[holiday],
        "Discount":[discount]
    })

    df["Date"] = pd.to_datetime(df["Date"])
    df["day"] = df["Date"].dt.day
    df["month"] = df["Date"].dt.month
    df["dayofweek"] = df["Date"].dt.dayofweek
    df["is_weekend"] = df["dayofweek"].isin([5,6]).astype(int)
    df.drop("Date", axis=1, inplace=True)

    df = pd.get_dummies(df)
    df = df.reindex(columns=train_columns, fill_value=0)
    return df

if st.button("Predict"):
    X = preprocess()

    orders_stack = pd.DataFrame({
        #"rf": rf_orders.predict(X),
        "xgb": xgb_orders.predict(X),
        "lgb": lgbm_orders.predict(X)
    })

    sales_stack = pd.DataFrame({
        #"rf": rf_sales.predict(X),
        "xgb": xgb_sales.predict(X),
        "lgb": lgbm_sales.predict(X)
    })

    orders = int(meta_orders.predict(orders_stack)[0])
    sales  = float(meta_sales.predict(sales_stack)[0])

    st.success(f"📦 Predicted Orders: {orders}")
    st.success(f"💰 Predicted Sales: {sales:.2f}")
