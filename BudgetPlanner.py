# streamlit_budget_app.py
"""
Comprehensive personal Budget Planning Streamlit App
Features:
- Income & Expenses with category management (food, travel, rent, entertainment, etc.)
- Weekly/Monthly budgets per category
- Recurring payments (EMI, rent, subscriptions) tracking
- Spending alerts when usage exceeds budget thresholds
- History views: 1 week, 1 month, 3 months, 6 months, 1 year
- Pie charts & bar graphs (plotly)
- Cash flow timeline (income vs expense)
- Savings goals with progress bars and reminders
- Receipt OCR (pytesseract)
- Export to Excel and PDF
- Local SQLite storage + optional cloud sync stub

Notes:
- This is a single-file Streamlit app. Install dependencies listed below.
- Cloud sync and biometric login are implemented as stubs/placeholders with guidance.

Requirements (pip):
streamlit pandas plotly pillow pytesseract fpdf numpy sqlalchemy

Optional for better OCR: install tesseract-ocr on your OS.

Run:
streamlit run streamlit_budget_app.py
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import sqlite3
import io
import base64
from datetime import datetime, timedelta
from fpdf import FPDF
from PIL import Image
import pytesseract
import os
from sqlalchemy import create_engine

# ---------- Configuration ----------
DB_PATH = "budget_app.db"
DEFAULT_CATEGORIES = ["Food", "Travel", "Rent", "Entertainment", "Bills", "Shopping", "Subscriptions", "Salary", "Other"]
ALERT_THRESHOLD_DEFAULT = 0.9  # 90%

# ---------- Utility functions ----------

def get_engine(path=DB_PATH):
    engine = create_engine(f"sqlite:///{path}", connect_args={"check_same_thread": False})
    return engine


def init_db():
    engine = get_engine()
    with engine.connect() as conn:
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS transactions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                date TEXT,
                type TEXT,
                category TEXT,
                amount REAL,
                note TEXT,
                recurring_id INTEGER
            )
            """
        )
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS recurring (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                name TEXT,
                category TEXT,
                amount REAL,
                start_date TEXT,
                frequency TEXT,
                note TEXT
            )
            """
        )
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS budgets (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                category TEXT,
                period TEXT,
                amount REAL
            )
            """
        )
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS goals (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                name TEXT,
                target_amount REAL,
                current_amount REAL,
                deadline TEXT,
                note TEXT
            )
            """
        )
    return engine


@st.cache_data
def load_transactions():
    engine = get_engine()
    try:
        df = pd.read_sql("SELECT * FROM transactions ORDER BY date DESC", engine)
        if not df.empty:
            df['date'] = pd.to_datetime(df['date'])
        return df
    except Exception:
        return pd.DataFrame(columns=['id','date','type','category','amount','note','recurring_id'])


def add_transaction(date, ttype, category, amount, note='', recurring_id=None):
    engine = get_engine()
    df = pd.DataFrame([{ 'date': pd.to_datetime(date).isoformat(), 'type': ttype, 'category': category, 'amount': float(amount), 'note': note, 'recurring_id': recurring_id }])
    df.to_sql('transactions', engine, if_exists='append', index=False)
    st.session_state['dirty'] = True


def add_recurring(name, category, amount, start_date, frequency, note=''):
    engine = get_engine()
    df = pd.DataFrame([{ 'name': name, 'category': category, 'amount': amount, 'start_date': pd.to_datetime(start_date).isoformat(), 'frequency': frequency, 'note': note }])
    df.to_sql('recurring', engine, if_exists='append', index=False)
    st.session_state['dirty'] = True


def add_budget(category, period, amount):
    engine = get_engine()
    df = pd.DataFrame([{ 'category': category, 'period': period, 'amount': float(amount) }])
    df.to_sql('budgets', engine, if_exists='append', index=False)
    st.session_state['dirty'] = True


def add_goal(name, target_amount, current_amount, deadline, note=''):
    engine = get_engine()
    df = pd.DataFrame([{ 'name': name, 'target_amount': float(target_amount), 'current_amount': float(current_amount), 'deadline': pd.to_datetime(deadline).isoformat(), 'note': note }])
    df.to_sql('goals', engine, if_exists='append', index=False)
    st.session_state['dirty'] = True


# ---------- Forecasting helper (simple linear trend) ----------

def small_forecast(series: pd.Series, periods=3):
    # series index should be numeric (e.g., 0..n) or datetime
    if series.dropna().shape[0] < 2:
        return [series.iloc[-1]] * periods if not series.empty else [0]*periods
    x = np.arange(len(series))
    y = series.values
    coeffs = np.polyfit(x, y, 1)
    future_x = np.arange(len(series), len(series)+periods)
    preds = coeffs[0]*future_x + coeffs[1]
    return preds.tolist()


# ---------- OCR helper ----------

def extract_text_from_image(image_file) -> str:
    try:
        img = Image.open(image_file)
        text = pytesseract.image_to_string(img)
        return text
    except Exception as e:
        return f"OCR error: {e}"


# ---------- Export helpers ----------

def df_to_excel_bytes(df: pd.DataFrame) -> bytes:
    towrite = io.BytesIO()
    df.to_excel(towrite, index=False, sheet_name='Transactions')
    towrite.seek(0)
    return towrite.read()


def df_to_pdf_bytes(df: pd.DataFrame, title='Report') -> bytes:
    pdf = FPDF()
    pdf.set_auto_page_break(auto=True, margin=15)
    pdf.add_page()
    pdf.set_font("Arial", size=12)
    pdf.cell(200, 10, txt=title, ln=True, align='C')
    pdf.ln(5)
    # simple table
    col_width = pdf.w / 4.5
    row_height = pdf.font_size * 1.5
    for i, col in enumerate(df.columns):
        pdf.cell(col_width, row_height, str(col), border=1)
    pdf.ln(row_height)
    for index, row in df.iterrows():
        for item in row:
            pdf.cell(col_width, row_height, str(item)[:30], border=1)
        pdf.ln(row_height)
    return pdf.output(dest='S').encode('latin-1')


# ---------- App UI & Logic ----------

def main():
    st.set_page_config(page_title="Budget Planner", layout='wide')
    st.title("💸 Budget Planner & Expense Tracker")

    # initialize
    if 'dirty' not in st.session_state:
        st.session_state['dirty'] = False
    engine = init_db()

    # Sidebar: Login (simple PIN) & settings
    with st.sidebar:
        st.header("Account")
        if 'logged_in' not in st.session_state:
            st.session_state['logged_in'] = False
        pin = st.text_input("Enter 4-digit PIN (set for this session)", type='password')
        if st.button("Login / Set PIN"):
            if pin and len(pin) == 4 and pin.isdigit():
                st.session_state['pin'] = pin
                st.session_state['logged_in'] = True
                st.success("PIN set for this session. For biometric, integrate OS-level auth in deployment.")
            else:
                st.error("Please enter a 4-digit numeric PIN.")

        st.markdown("---")
        st.header("Settings")
        alert_threshold = st.slider("Spending Alert Threshold (%)", min_value=50, max_value=150, value=int(ALERT_THRESHOLD_DEFAULT*100))
        categories = st.text_area("Categories (comma separated)", value=',' .join(DEFAULT_CATEGORIES))
        categories = [c.strip() for c in categories.split(',') if c.strip()]
        st.markdown("**Cloud Sync (optional)**: Provide path to cloud-backed DB (e.g., mounted drive or URL).\nCurrently a placeholder — integrate with Firebase / GDrive in deployment.")

    if not st.session_state['logged_in']:
        st.info("Please set a PIN to continue (session-only).")
        st.stop()

    # Top menu
    menu = st.radio("Go to", ["Dashboard", "Add Transaction", "Recurring", "Budgets", "Goals", "Reports & Export", "Receipt OCR", "Settings"], index=0)

    # Load fresh data
    df = load_transactions()
    recurring = pd.read_sql('SELECT * FROM recurring', engine) if engine else pd.DataFrame()
    budgets = pd.read_sql('SELECT * FROM budgets', engine) if engine else pd.DataFrame()
    goals = pd.read_sql('SELECT * FROM goals', engine) if engine else pd.DataFrame()

    # --- ADD TRANSACTION ---
    if menu == 'Add Transaction':
        st.header("Add Income / Expense")
        col1, col2 = st.columns(2)
        with col1:
            ttype = st.selectbox("Type", ['Expense','Income'])
            date = st.date_input("Date", value=datetime.now())
            category = st.selectbox("Category", categories)
        with col2:
            amount = st.number_input("Amount", min_value=0.0, format="%.2f")
            note = st.text_input("Note")
            if st.button("Add Transaction"):
                if amount <= 0:
                    st.error("Enter a positive amount")
                else:
                    add_transaction(date, ttype, category, amount, note)
                    st.success("Transaction added")

    # --- RECURRING ---
    if menu == 'Recurring':
        st.header("Recurring Payments (EMI, Rent, Subscriptions)")
        with st.expander("Add Recurring Item"):
            name = st.text_input("Name")
            cat = st.selectbox("Category", categories)
            amt = st.number_input("Amount", min_value=0.0, format="%.2f", key='rec_amt')
            start = st.date_input("Start Date", value=datetime.now(), key='rec_start')
            freq = st.selectbox("Frequency", ['Monthly','Weekly','Daily','Yearly'])
            note = st.text_input("Note", key='rec_note')
            if st.button("Add Recurring"):
                add_recurring(name, cat, amt, start, freq, note)
                st.success("Recurring added")
        st.markdown("---")
        st.subheader("Existing Recurring Items")
        if not recurring.empty:
            recurring['start_date'] = pd.to_datetime(recurring['start_date'])
            st.dataframe(recurring)
            if st.button("Generate next month recurring transactions"):
                # expand recurring into transactions for next 30 days
                cutoff = datetime.now() + timedelta(days=30)
                for _, row in recurring.iterrows():
                    add_transaction(row['start_date'], 'Expense', row['category'], row['amount'], note=f"Recurring: {row['name']}", recurring_id=row['id'])
                st.success("Added recurring transactions (sample)")
        else:
            st.info("No recurring items set.")

    # --- BUDGETS ---
    if menu == 'Budgets':
        st.header("Budgets per Category")
        with st.expander("Add Budget"):
            bcat = st.selectbox("Category", categories, key='bcat')
            period = st.selectbox("Period", ['Weekly','Monthly','Yearly'])
            bamt = st.number_input("Budget Amount", min_value=0.0, format="%.2f", key='bamt')
            if st.button("Save Budget"):
                add_budget(bcat, period, bamt)
                st.success('Budget saved')
        st.markdown("---")
        st.subheader('Current Budgets')
        if not budgets.empty:
            st.dataframe(budgets)
        else:
            st.info('No budgets set yet.')

    # --- GOALS ---
    if menu == 'Goals':
        st.header('Savings Goals')
        with st.expander('Create Goal'):
            gname = st.text_input('Goal Name')
            gtarget = st.number_input('Target Amount', min_value=0.0, format='%.2f', key='gtarget')
            gcurrent = st.number_input('Starting Amount', min_value=0.0, format='%.2f', key='gcurrent')
            gdeadline = st.date_input('Deadline', value=datetime.now()+timedelta(days=90), key='gdeadline')
            gnote = st.text_input('Note')
            if st.button('Create Goal'):
                add_goal(gname, gtarget, gcurrent, gdeadline, gnote)
                st.success('Goal created')
        st.markdown('---')
        st.subheader('Active Goals')
        if not goals.empty:
            goals['deadline'] = pd.to_datetime(goals['deadline'])
            for _, g in goals.iterrows():
                progress = 0 if g['target_amount'] == 0 else (g['current_amount'] / g['target_amount'])
                st.write(f"**{g['name']}** — Target: {g['target_amount']} | Current: {g['current_amount']} | Deadline: {g['deadline'].date()}")
                st.progress(min(max(progress,0),1))
                if progress >= 1:
                    st.balloons()
        else:
            st.info('No goals yet.')

    # --- DASHBOARD ---
    if menu == 'Dashboard':
        st.header('Overview')
        # filter options
        range_map = {
            '1 Week': 7,
            '1 Month': 30,
            '3 Months': 90,
            '6 Months': 180,
            '1 Year': 365,
            'All': 10000
        }
        rng = st.selectbox('History Range', list(range_map.keys()), index=1)
        days = range_map[rng]
        cutoff = datetime.now() - timedelta(days=days)
        df = load_transactions()
        df_filtered = df[df['date'] >= cutoff]

        if df_filtered.empty:
            st.info('No transactions for the selected period.')
        else:
            # cash flow timeline
            flow = df_filtered.copy()
            flow['sign'] = flow['type'].apply(lambda x: 1 if x == 'Income' else -1)
            flow['net'] = flow['amount'] * flow['sign']
            timeline = flow.groupby(pd.Grouper(key='date', freq='D')).sum().reset_index()
            if timeline.empty:
                st.info('Not enough data for timeline')
            else:
                fig = px.area(timeline, x='date', y='net', title='Cash Flow (Income vs Expense)')
                st.plotly_chart(fig, use_container_width=True)

            # category breakdown
            cat_sum = df_filtered.groupby('category').apply(lambda x: x[x['type']=='Expense']['amount'].sum()).reset_index(name='expense')
            cat_sum = cat_sum.sort_values('expense', ascending=False)
            if not cat_sum.empty:
                fig2 = px.pie(cat_sum, values='expense', names='category', title='Spending by Category')
                st.plotly_chart(fig2, use_container_width=True)

            # spending pattern (bar)
            daily = flow.groupby([pd.Grouper(key='date', freq='D'), 'type'])['amount'].sum().unstack(fill_value=0).reset_index()
            if not daily.empty:
                fig3 = px.bar(daily, x='date', y=['Expense','Income'], title='Daily Income & Expense')
                st.plotly_chart(fig3, use_container_width=True)

            # Alerts if overspent vs budgets
            st.subheader('Budget Alerts')
            if not budgets.empty:
                budget_alerts = []
                for _, b in budgets.iterrows():
                    per = b['period']
                    if per == 'Monthly':
                        p_cut = datetime.now() - timedelta(days=30)
                    elif per == 'Weekly':
                        p_cut = datetime.now() - timedelta(days=7)
                    elif per == 'Yearly':
                        p_cut = datetime.now() - timedelta(days=365)
                    else:
                        p_cut = datetime.now() - timedelta(days=30)
                    used = df[(df['category']==b['category']) & (df['date'] >= p_cut) & (df['type']=='Expense')]['amount'].sum()
                    if used >= b['amount'] * (alert_threshold/100.0):
                        budget_alerts.append((b['category'], used, b['amount']))
                if budget_alerts:
                    for cat, used, amt in budget_alerts:
                        st.warning(f"You have used ₹{used:.2f} of ₹{amt:.2f} for {cat} (>= {alert_threshold}%). Consider reducing spend or increase budget.")
                else:
                    st.success('All budgets are within threshold.')
            else:
                st.info('No budgets set to show alerts.')

            # Forecast sample: predict next 3 periods total expense
            st.subheader('Simple Forecast (Trend-based)')
            total_by_day = df_filtered[df_filtered['type']=='Expense'].groupby(pd.Grouper(key='date', freq='D'))['amount'].sum()
            preds = small_forecast(total_by_day, periods=7)
            forecast_df = pd.DataFrame({'date': [datetime.now().date()+timedelta(days=i+1) for i in range(len(preds))], 'predicted_expense': preds})
            st.dataframe(forecast_df)

    # --- REPORTS & EXPORT ---
    if menu == 'Reports & Export':
        st.header('Reports & Export')
        df = load_transactions()
        st.dataframe(df)
        col1, col2 = st.columns(2)
        with col1:
            if st.button('Download Excel'):
                data = df_to_excel_bytes(df)
                st.download_button('Click to download Excel', data, file_name='transactions.xlsx', mime='application/vnd.openxmlformats-officedocument.spreadsheetml.sheet')
        with col2:
            if st.button('Download PDF'):
                data = df_to_pdf_bytes(df, title='Transactions Report')
                st.download_button('Click to download PDF', data, file_name='transactions.pdf', mime='application/pdf')

    # --- RECEIPT OCR ---
    if menu == 'Receipt OCR':
        st.header('Receipt Scanner (OCR)')
        st.write('Upload an image of a receipt (jpg, png). Tesseract must be installed on your machine for OCR to work well.')
        uploaded = st.file_uploader('Upload receipt image', type=['png','jpg','jpeg'])
        if uploaded is not None:
            txt = extract_text_from_image(uploaded)
            st.text_area('Extracted Text', value=txt, height=300)
            # attempt to parse amounts date
            amounts = []
            for token in txt.replace('\n',' ').split(' '):
                try:
                    val = float(token.replace('₹','').replace(',',''))
                    amounts.append(val)
                except:
                    pass
            if amounts:
                st.write('Detected amounts (possible totals):', amounts[:5])

    # --- SETTINGS ---
    if menu == 'Settings':
        st.header('Advanced Settings & Notes')
        st.markdown('''
- **Cloud Sync:** To enable cloud sync, mount a remote DB or integrate Firebase/Google Drive. Replace DB_PATH with your cloud path.
- **Biometric Login:** Streamlit apps run in browser—biometric auth requires integrating with the hosting platform (mobile app wrapper or OAuth). This app provides a PIN-based session login as a simple alternative.
- **Background reminders/notifications:** For push reminders, integrate with a scheduled job (cron) or use services like OneSignal / Firebase Cloud Messaging and connect via a backend.
- **Security:** Protect the SQLite file and use HTTPS in deployment. For multi-device sync, use a central DB and user authentication (Auth0 / Firebase Auth).
''')

    # If we made changes, clear cache so load_transactions updates
    if st.session_state['dirty']:
        load_transactions.clear()
        st.session_state['dirty'] = False


if __name__ == '__main__':
    main()
