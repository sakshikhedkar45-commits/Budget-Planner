# app_with_auth_and_cloud.py
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
from PIL import Image
import pytesseract
import hashlib
import os
import io
import smtplib
from email.message import EmailMessage
import json
from dotenv import load_dotenv

# Optional Firebase imports
FIREBASE_AVAILABLE = False
try:
    import firebase_admin
    from firebase_admin import credentials, firestore
    FIREBASE_AVAILABLE = True
except Exception:
    FIREBASE_AVAILABLE = False

load_dotenv()  # load .env if present

st.set_page_config(page_title="Budget Planner — Auth & Cloud", layout="wide")
st.title("💼 Smart Budget Planner — Login, Cloud & Email Reminders")

# ---------- Config ----------
FIREBASE_SA_JSON = os.getenv("FIREBASE_SA_JSON")  # path to service account json
SMTP_SERVER = os.getenv("SMTP_SERVER")
SMTP_PORT = os.getenv("SMTP_PORT")
SMTP_EMAIL = os.getenv("SMTP_EMAIL")
SMTP_PASSWORD = os.getenv("SMTP_PASSWORD")

# ---------- Firebase Initialization (optional) ----------
db = None
use_firestore = False
if FIREBASE_AVAILABLE and FIREBASE_SA_JSON and os.path.exists(FIREBASE_SA_JSON):
    try:
        cred = credentials.Certificate(FIREBASE_SA_JSON)
        firebase_admin.initialize_app(cred)
        db = firestore.client()
        use_firestore = True
        st.info("Connected to Firebase Firestore (cloud DB enabled).")
    except Exception as e:
        st.warning(f"Could not initialize Firebase: {e}\nFalling back to local storage.")
else:
    if FIREBASE_SA_JSON:
        st.warning("Firebase libs missing or service account JSON path invalid. Using local mode.")
    else:
        st.info("Firebase not configured. Using local (session) storage.")

# ---------- Utility helpers ----------
def hash_password(password: str) -> str:
    return hashlib.sha256(password.encode("utf-8")).hexdigest()

def send_email(recipient_email: str, subject: str, body: str) -> bool:
    """Send email via SMTP. Returns True if ok."""
    if not (SMTP_SERVER and SMTP_PORT and SMTP_EMAIL and SMTP_PASSWORD):
        return False
    try:
        msg = EmailMessage()
        msg["Subject"] = subject
        msg["From"] = SMTP_EMAIL
        msg["To"] = recipient_email
        msg.set_content(body)
        with smtplib.SMTP(SMTP_SERVER, int(SMTP_PORT)) as smtp:
            smtp.starttls()
            smtp.login(SMTP_EMAIL, SMTP_PASSWORD)
            smtp.send_message(msg)
        return True
    except Exception as e:
        st.error(f"Email send failed: {e}")
        return False

# ---------- Cloud functions (if Firestore enabled) ----------
def firestore_get_user(email):
    doc = db.collection("users").document(email).get()
    return doc.to_dict() if doc.exists else None

def firestore_create_user(email, hashed_password):
    db.collection("users").document(email).set({
        "password_hash": hashed_password,
        "created_at": datetime.utcnow().isoformat()
    })
    # create empty subcollections
    db.collection("users").document(email).collection("transactions")  # will be created on write

def firestore_save_transaction(email, tx):
    # tx is dict
    db.collection("users").document(email).collection("transactions").add(tx)

def firestore_get_transactions(email):
    docs = db.collection("users").document(email).collection("transactions").order_by("date", direction=firestore.Query.DESCENDING).stream()
    rows = []
    for d in docs:
        doc = d.to_dict()
        rows.append(doc)
    return pd.DataFrame(rows) if rows else pd.DataFrame(columns=["date","type","category","amount","notes"])

def firestore_save_reminder(email, reminder):
    db.collection("users").document(email).collection("reminders").add(reminder)

def firestore_get_reminders(email):
    docs = db.collection("users").document(email).collection("reminders").order_by("due", direction=firestore.Query.ASCENDING).stream()
    rows = []
    for d in docs:
        rows.append(d.to_dict())
    return pd.DataFrame(rows) if rows else pd.DataFrame(columns=["name","amount","due"])

# ---------- Local session storage fallback ----------
if "users_local" not in st.session_state:
    st.session_state["users_local"] = {}  # email -> {password_hash, created_at}
if "data_local" not in st.session_state:
    st.session_state["data_local"] = {}   # email -> {"transactions":DataFrame, "reminders":DataFrame, "goals":DataFrame}

def ensure_local_user_data(email):
    if email not in st.session_state["data_local"]:
        st.session_state["data_local"][email] = {
            "transactions": pd.DataFrame(columns=["date","type","category","amount","notes"]),
            "reminders": pd.DataFrame(columns=["name","amount","due"]),
            "goals": pd.DataFrame(columns=["goal","target","saved","deadline"])
        }

# ---------- Auth UI ----------
st.sidebar.header("Account")
auth_mode = st.sidebar.radio("Mode", ["Login", "Register", "Guest"])

if auth_mode in ("Login", "Register"):
    email = st.sidebar.text_input("Email")
    password = st.sidebar.text_input("Password", type="password")

login_button = st.sidebar.button("Proceed")

current_user = None
if login_button:
    if auth_mode == "Register":
        if not email or not password:
            st.sidebar.error("Provide email and password")
        else:
            hashed = hash_password(password)
            if use_firestore:
                if firestore_get_user(email):
                    st.sidebar.error("User exists. Please login.")
                else:
                    firestore_create_user(email, hashed)
                    st.sidebar.success("Registered! You can login now.")
            else:
                if email in st.session_state["users_local"]:
                    st.sidebar.error("User exists (local). Please login.")
                else:
                    st.session_state["users_local"][email] = {"password_hash": hashed, "created_at": datetime.utcnow().isoformat()}
                    ensure_local_user_data(email)
                    st.sidebar.success("Local account created. You can login now.")
    elif auth_mode == "Login":
        if not email or not password:
            st.sidebar.error("Provide email and password")
        else:
            hashed = hash_password(password)
            if use_firestore:
                user = firestore_get_user(email)
                if not user:
                    st.sidebar.error("User not found. Register first.")
                elif user.get("password_hash") != hashed:
                    st.sidebar.error("Incorrect password.")
                else:
                    st.session_state["user_email"] = email
                    st.sidebar.success("Logged in.")
            else:
                local = st.session_state["users_local"].get(email)
                if not local:
                    st.sidebar.error("Local user not found. Register first.")
                elif local.get("password_hash") != hashed:
                    st.sidebar.error("Incorrect password.")
                else:
                    st.session_state["user_email"] = email
                    ensure_local_user_data(email)
                    st.sidebar.success("Logged in (local).")
else:
    # Guest mode
    if auth_mode == "Guest":
        if st.sidebar.button("Continue as Guest"):
            st.session_state["user_email"] = "guest@local"
            ensure_local_user_data(st.session_state["user_email"])
            st.sidebar.success("Continuing as guest (local only).")

# If user logged in, load dashboard
if "user_email" not in st.session_state:
    st.info("Please register / login in the sidebar to continue. Or choose Guest.")
    st.stop()

user_email = st.session_state["user_email"]
st.write(f"**Signed in as:** {user_email}")

# Ensure local data if not using Firestore
if not use_firestore:
    ensure_local_user_data(user_email)

# ---------- App main UI ----------
menu = st.radio("Main", ["Add Transaction", "Dashboard", "Reminders & Email", "Receipt OCR", "Export Data", "Logout"])

# Common categories
categories = ["Food","Travel","Rent","EMI","Bills","Subscriptions","Entertainment","Shopping","Groceries","Other"]

# ----- Add Transaction -----
if menu == "Add Transaction":
    st.header("➕ Add Transaction")
    col1, col2 = st.columns(2)
    with col1:
        tx_type = st.selectbox("Type", ["Expense","Income"])
        tx_cat = st.selectbox("Category", categories)
    with col2:
        tx_amount = st.number_input("Amount", min_value=1.0)
        tx_notes = st.text_input("Notes")
    tx_date = st.date_input("Date", value=datetime.now())

    if st.button("Save Transaction"):
        tx = {
            "date": tx_date.isoformat(),
            "type": tx_type,
            "category": tx_cat,
            "amount": float(tx_amount),
            "notes": tx_notes
        }
        if use_firestore:
            try:
                firestore_save_transaction(user_email, tx)
                st.success("Saved to Firestore.")
            except Exception as e:
                st.error(f"Cloud save failed: {e}")
        else:
            st.session_state["data_local"][user_email]["transactions"] = \
                st.session_state["data_local"][user_email]["transactions"].append(tx, ignore_index=True)
            st.success("Saved locally.")

# ----- Dashboard -----
elif menu == "Dashboard":
    st.header("📊 Dashboard")
    if use_firestore:
        tx_df = firestore_get_transactions(user_email)
        # convert date to datetime
        if not tx_df.empty:
            tx_df['date'] = pd.to_datetime(tx_df['date'])
    else:
        tx_df = st.session_state["data_local"][user_email]["transactions"].copy()
        if not tx_df.empty:
            tx_df['date'] = pd.to_datetime(tx_df['date'])

    if tx_df.empty:
        st.info("No transactions yet.")
    else:
        # quick summary
        total_income = tx_df[tx_df['type']=="Income"]['amount'].sum() if 'type' in tx_df else 0
        total_exp = tx_df[tx_df['type']=="Expense"]['amount'].sum() if 'type' in tx_df else 0
        st.metric("Total Income", f"₹{total_income:.2f}")
        st.metric("Total Expense", f"₹{total_exp:.2f}")
        st.metric("Net", f"₹{(total_income - total_exp):.2f}")

        st.subheader("Recent Transactions")
        st.dataframe(tx_df.sort_values('date', ascending=False).head(20))

        # Pie chart for expenses
        exp = tx_df[tx_df['type']=="Expense"].groupby('category')['amount'].sum()
        if not exp.empty:
            fig1, ax1 = plt.subplots()
            ax1.pie(exp.values, labels=exp.index, autopct="%1.1f%%")
            ax1.set_title("Expense Breakdown")
            st.pyplot(fig1)

        # Cashflow over time (daily)
        daily = tx_df.groupby(pd.Grouper(key='date', freq='D'))['amount'].sum().reset_index()
        if not daily.empty:
            fig2, ax2 = plt.subplots()
            ax2.plot(daily['date'], daily['amount'])
            ax2.set_title("Cash Flow (daily)")
            st.pyplot(fig2)

# ----- Reminders & Email -----
elif menu == "Reminders & Email":
    st.header("⏰ Bill Reminders & Email")

    r_col1, r_col2 = st.columns(2)
    with r_col1:
        r_name = st.text_input("Bill / EMI Name")
        r_amt = st.number_input("Amount", min_value=1.0)
    with r_col2:
        r_due = st.date_input("Due Date", value=datetime.now() + timedelta(days=7))
        r_note = st.text_input("Note (optional)")

    if st.button("Save Reminder"):
        reminder = {"name": r_name, "amount": float(r_amt), "due": r_due.isoformat(), "note": r_note}
        if use_firestore:
            try:
                firestore_save_reminder(user_email, reminder)
                st.success("Reminder saved to Firestore.")
            except Exception as e:
                st.error(f"Cloud save failed: {e}")
        else:
            df = st.session_state["data_local"][user_email]["reminders"]
            df = df.append(reminder, ignore_index=True)
            st.session_state["data_local"][user_email]["reminders"] = df
            st.success("Reminder saved locally.")

    # Show upcoming reminders
    st.subheader("Upcoming reminders (next 30 days)")
    if use_firestore:
        rem_df = firestore_get_reminders(user_email)
        if not rem_df.empty:
            rem_df['due'] = pd.to_datetime(rem_df['due'])
            upcoming = rem_df[rem_df['due'] <= (datetime.now() + timedelta(days=30))]
            st.dataframe(upcoming)
        else:
            st.info("No reminders saved.")
    else:
        rem_df = st.session_state["data_local"][user_email]["reminders"]
        if not rem_df.empty:
            rem_df['due'] = pd.to_datetime(rem_df['due'])
            st.dataframe(rem_df[rem_df['due'] <= (datetime.now() + timedelta(days=30))])
        else:
            st.info("No reminders saved.")

    # Send reminders via email
    st.markdown("---")
    st.subheader("Send due reminders via Email")
    st.write("This will send reminder emails immediately (requires SMTP configured in .env).")

    if st.button("Send due reminders now"):
        # collect due reminders
        if use_firestore:
            rem_df = firestore_get_reminders(user_email)
            if rem_df.empty:
                st.info("No reminders to send.")
            else:
                rem_df['due'] = pd.to_datetime(rem_df['due'])
        else:
            rem_df = st.session_state["data_local"][user_email]["reminders"]
            if rem_df.empty:
                st.info("No reminders to send.")
            else:
                rem_df['due'] = pd.to_datetime(rem_df['due'])

        if not rem_df.empty:
            to_send = rem_df[rem_df['due'] <= (datetime.now() + timedelta(days=7))]
            if to_send.empty:
                st.info("No reminders due within the next 7 days.")
            else:
                if not (SMTP_SERVER and SMTP_PORT and SMTP_EMAIL and SMTP_PASSWORD):
                    st.warning("SMTP not configured. Set SMTP_SERVER, SMTP_PORT, SMTP_EMAIL, SMTP_PASSWORD in .env to enable email sending.")
                sent_count = 0
                for _, row in to_send.iterrows():
                    subject = f"Reminder: {row.get('name','Bill')} due on {row['due'].date() if hasattr(row['due'],'date') else row['due']}"
                    body = f"Hi,\n\nThis is a reminder that {row.get('name','a bill')} of amount ₹{row.get('amount',0)} is due on {row['due'].date() if hasattr(row['due'],'date') else row['due']}.\n\nNote: {row.get('note','')}\n\n— Smart Budget Planner"
                    ok = send_email(user_email, subject, body)
                    if ok:
                        sent_count += 1
                st.success(f"Sent {sent_count} reminder(s) to {user_email} (if SMTP configured).")

# ----- Receipt OCR -----
elif menu == "Receipt OCR":
    st.header("🧾 Receipt OCR")
    uploaded = st.file_uploader("Upload receipt image", type=["png","jpg","jpeg"])
    if uploaded:
        img = Image.open(uploaded)
        st.image(img, caption="Uploaded")
        text = pytesseract.image_to_string(img)
        st.subheader("Extracted text")
        st.write(text)

# ----- Export Data -----
elif menu == "Export Data":
    st.header("📤 Export Data")
    if use_firestore:
        tx_df = firestore_get_transactions(user_email)
    else:
        tx_df = st.session_state["data_local"][user_email]["transactions"]

    if tx_df is None or tx_df.empty:
        st.info("No transactions available.")
    else:
        st.dataframe(tx_df)
        csv = tx_df.to_csv(index=False).encode()
        st.download_button("Download CSV", csv, file_name="transactions.csv", mime="text/csv")
        # Excel
        excel = io.BytesIO()
        tx_df.to_excel(excel, index=False)
        st.download_button("Download Excel", excel.getvalue(), file_name="transactions.xlsx")

# ----- Logout -----
elif menu == "Logout":
    if st.button("Logout now"):
        st.session_state.pop("user_email", None)
        st.experimental_rerun()
