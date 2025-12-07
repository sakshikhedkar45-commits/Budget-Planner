import streamlit as st
import pandas as pd
import plotly.express as px
import firebase_admin
from firebase_admin import credentials, firestore
import datetime
import numpy as np
from sklearn.linear_model import LinearRegression

# -------------------------
# FIREBASE CLOUD DATABASE
# -------------------------
if not firebase_admin._apps:
    cred = credentials.Certificate("firebase.json")  # upload firebase.json to Streamlit
    firebase_admin.initialize_app(cred)

db = firestore.client()

# -------------------------
# SIDEBAR LOGIN
# -------------------------
st.sidebar.title("🔐 Login")

email = st.sidebar.text_input("Email")
password = st.sidebar.text_input("Password", type="password")

if st.sidebar.button("Login"):
    if email != "" and password != "":
        st.session_state["login"] = True
        st.session_state["user"] = email
        st.success("Login Successful!")
    else:
        st.error("Enter valid email & password")

if st.session_state.get("login") != True:
    st.warning("Please login to continue")
    st.stop()

user = st.session_state["user"]

# -------------------------
# LOAD USER DATA
# -------------------------
def load_data():
    docs = db.collection("users").document(user).collection("records").stream()
    data = [d.to_dict() for d in docs]
    return pd.DataFrame(data)

def save_record(amount, category, dtype):
    db.collection("users").document(user).collection("records").add({
        "amount": float(amount),
        "category": category,
        "type": dtype,
        "date": datetime.datetime.now()
    })

# -------------------------
# MAIN APP
# -------------------------
st.title("💰 Smart Budget Planner App")

menu = st.selectbox("Menu", ["Add Entry", "Dashboard", "Forecasting", "Goals", "History"])

# -------------------------------------------------------
# ADD ENTRY (Income / Expense)
# -------------------------------------------------------
if menu == "Add Entry":
    st.subheader("➕ Add Income / Expense")

    amount = st.number_input("Amount", min_value=1.0)
    category = st.selectbox("Category", ["Food", "Travel", "Rent", "EMI", "Fees", "Entertainment", "Shopping", "Other"])
    dtype = st.selectbox("Type", ["Income", "Expense"])

    if st.button("Save"):
        save_record(amount, category, dtype)
        st.success("Record Saved!")

# -------------------------------------------------------
# DASHBOARD
# -------------------------------------------------------
if menu == "Dashboard":
    st.subheader("📊 Budget Dashboard")

    df = load_data()
    if df.empty:
        st.info("No data yet.")
    else:
        df["date"] = pd.to_datetime(df["date"])

        expenses = df[df["type"] == "Expense"]
        income = df[df["type"] == "Income"]

        total_expense = expenses["amount"].sum()
        total_income = income["amount"].sum()

        st.metric("Total Income", f"₹{total_income}")
        st.metric("Total Expense", f"₹{total_expense}")

        pie = px.pie(expenses, names="category", values="amount", title="Expense Breakdown")
        st.plotly_chart(pie)

# -------------------------------------------------------
# FORECASTING
# -------------------------------------------------------
if menu == "Forecasting":
    st.subheader("📈 Expense Forecasting")

    df = load_data()
    if df.empty:
        st.warning("No data available for prediction.")
    else:
        df = df[df["type"] == "Expense"]
        df["date"] = pd.to_datetime(df["date"])
        df["day"] = (df["date"] - df["date"].min()).dt.days

        X = df["day"].values.reshape(-1, 1)
        y = df["amount"].values

        model = LinearRegression()
        model.fit(X, y)

        next_30 = model.predict([[df["day"].max() + 30]])
        st.info(f"Estimated Expense After 30 Days: **₹{round(next_30[0], 2)}**")

# -------------------------------------------------------
# GOALS
# -------------------------------------------------------
if menu == "Goals":
    st.subheader("🎯 Savings Goals")

    goal_name = st.text_input("Goal Name")
    target_amount = st.number_input("Target Amount", min_value=1.0)
    
    if st.button("Save Goal"):
        db.collection("users").document(user).collection("goals").add({
            "goal": goal_name,
            "target": target_amount,
            "created": datetime.datetime.now()
        })
        st.success("Goal Saved!")

    st.write("Your Goals:")
    goals = db.collection("users").document(user).collection("goals").stream()
    for g in goals:
        gd = g.to_dict()
        st.write(f"**{gd['goal']}** – Target: ₹{gd['target']}")

# -------------------------------------------------------
# HISTORY
# -------------------------------------------------------
if menu == "History":
    st.subheader("📅 Transaction History")

    df = load_data()
    if df.empty:
        st.info("No Data Found")
    else:
        df = df.sort_values("date", ascending=False)
        st.dataframe(df)
