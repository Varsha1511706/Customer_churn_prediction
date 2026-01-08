
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import time
import warnings
warnings.filterwarnings('ignore')

# ML stack
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix
)
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.calibration import CalibratedClassifierCV
from sklearn.utils.class_weight import compute_class_weight

import joblib
import os

st.set_page_config(
    page_title="Telecom Churn AI Pro",
    page_icon="📱",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""<style>
    .stApp { background: linear-gradient(135deg, #FFFFFF 0%, #F8F9FA 50%, #E9ECEF 100%) !important; }
    * { color: #000000 !important; font-family: 'Arial', sans-serif !important; }
    .main-title { font-size: 48px !important; font-weight: 900 !important; color: #1A56DB !important; text-align: center;
        margin: 20px 0 40px 0 !important; text-shadow: 2px 2px 4px rgba(0,0,0,0.1); padding: 20px;
        background: linear-gradient(90deg, #1A56DB, #3B82F6); -webkit-background-clip: text; -webkit-text-fill-color: transparent; }
    .section-header { font-size: 36px !important; font-weight: 800 !important; color: #1E40AF !important; margin: 30px 0 20px 0 !important;
        padding-bottom: 15px; border-bottom: 4px solid #3B82F6; display: flex; align-items: center; gap: 10px; }
    .subsection-header { font-size: 28px !important; font-weight: 700 !important; color: #2563EB !important; margin: 25px 0 15px 0 !important; }
    .stButton > button { background: linear-gradient(90deg, #1D4ED8, #3B82F6) !important; color: white !important; border: none !important;
        padding: 18px 35px !important; border-radius: 12px !important; font-size: 20px !important; font-weight: 800 !important; width: 100% !important;
        margin: 15px 0 !important; cursor: pointer !important; transition: all 0.3s !important; box-shadow: 0 6px 20px rgba(59, 130, 246, 0.3) !important;
        text-transform: uppercase; letter-spacing: 1px; }
    .stButton > button:hover { transform: translateY(-5px) !important; box-shadow: 0 12px 30px rgba(59, 130, 246, 0.5) !important;
        background: linear-gradient(90deg, #1E40AF, #2563EB) !important; }
    .metric-card { background: white !important; padding: 25px !important; border-radius: 20px !important; border: 3px solid #3B82F6 !important;
        margin: 15px !important; box-shadow: 0 10px 30px rgba(0,0,0,0.15) !important; text-align: center; transition: transform 0.3s; }
    .metric-card:hover { transform: translateY(-10px); box-shadow: 0 20px 40px rgba(0,0,0,0.2) !important; }
    .metric-value { font-size: 42px !important; font-weight: 900 !important; color: #1D4ED8 !important; margin: 10px 0 !important;
        text-shadow: 2px 2px 4px rgba(0,0,0,0.1); }
    .metric-label { font-size: 16px !important; font-weight: 600 !important; color: #4B5563 !important; text-transform: uppercase; letter-spacing: 1px; }
    .custom-card { background: white; padding: 30px; border-radius: 20px; border: 3px solid #3B82F6; box-shadow: 0 15px 35px rgba(0,0,0,0.1); margin: 25px 0; }
    .block-container { padding-top: 3rem; padding-bottom: 3rem; padding-left: 3rem; padding-right: 3rem; }
</style>""", unsafe_allow_html=True)

CITY_LATLON = {
    'Delhi': (28.6139, 77.2090),
    'Mumbai': (19.0760, 72.8777),
    'Bangalore': (12.9716, 77.5946),
    'Chennai': (13.0827, 80.2707),
    'Hyderabad': (17.3850, 78.4867),
    'Pune': (18.5204, 73.8567),
    'Kolkata': (22.5726, 88.3639),
    'Ahmedabad': (23.0225, 72.5714)
}

def generate_synthetic(n=4000, seed=42):
    np.random.seed(seed)
    cities = list(CITY_LATLON.keys())
    data = {
        'customer_id': [f'IN{str(i).zfill(6)}' for i in range(n)],
        'tenure': np.random.randint(1, 72, n),
        'monthly_charges': np.random.uniform(100, 1800, n),
        'contract': np.random.choice(['Monthly', 'Yearly', 'Two Year'], n, p=[0.55, 0.3, 0.15]),
        'internet': np.random.choice(['Fiber', 'DSL', 'None'], n, p=[0.5, 0.35, 0.15]),
        'payment': np.random.choice(['Credit Card', 'Bank', 'Cash', 'UPI'], n, p=[0.35, 0.35, 0.2, 0.1]),
        'support_calls': np.random.randint(0, 12, n),
        'satisfaction': np.random.randint(1, 10, n),
        'data_usage': np.random.uniform(1, 250, n),
        'location': np.random.choice(cities, n, p=[0.16,0.16,0.14,0.12,0.12,0.1,0.1,0.1]),
        'sentiment': np.clip(np.random.normal(0.2, 0.5, n), -1, 1),
        'infra_quality': np.random.uniform(0.4, 0.95, n)
    }
    churn_prob = 0.10
    contract_map = {'Monthly': 0.18, 'Yearly': 0.06, 'Two Year': 0.03}
    churn_prob += np.array([contract_map[c] for c in data['contract']])
    churn_prob += (10 - np.array(data['satisfaction'])) * 0.02
    churn_prob += np.array(data['support_calls']) * 0.025
    churn_prob -= np.array(data['tenure']) / 120
    churn_prob += (np.array(data['monthly_charges']) > 1200).astype(int) * 0.05
    churn_prob -= np.array(data['sentiment']) * 0.08
    churn_prob -= np.array(data['infra_quality']) * 0.06
    churn_prob = np.clip(churn_prob, 0.04, 0.85)
    data['churn'] = (np.random.random(n) < churn_prob).astype(int)

    df = pd.DataFrame(data)
    df['lat'] = df['location'].map(lambda c: CITY_LATLON[c][0])
    df['lon'] = df['location'].map(lambda c: CITY_LATLON[c][1])
    return df

TARGET = 'churn'
ID_COL = 'customer_id'
CATEGORICAL = ['contract', 'internet', 'payment', 'location']
NUMERIC = ['tenure', 'monthly_charges', 'support_calls', 'satisfaction', 'data_usage', 'sentiment', 'infra_quality']

def build_pipeline(algorithm='Logistic Regression', class_weight='balanced', calibrate=True, random_state=42):
    preprocessor = ColumnTransformer(
        transformers=[
            ('cat', OneHotEncoder(handle_unknown='ignore'), CATEGORICAL),
            ('num', StandardScaler(), NUMERIC)
        ],
        remainder='drop'
    )
    if algorithm == 'Random Forest':
        clf = RandomForestClassifier(
            n_estimators=250, max_depth=14, random_state=random_state,
            class_weight='balanced_subsample', n_jobs=-1
        )
    else:
        clf = LogisticRegression(
            C=1.0, solver='lbfgs', max_iter=1000, class_weight=class_weight, random_state=random_state
        )
    if calibrate:
        clf = CalibratedClassifierCV(clf, method='sigmoid', cv=2)
    pipe = Pipeline(steps=[('pre', preprocessor), ('clf', clf)])
    return pipe

def train_evaluate(df, test_size=0.2, algorithm='Logistic Regression', random_state=42, calibrate=True):
    X = df[CATEGORICAL + NUMERIC]
    y = df[TARGET]
    classes = np.unique(y)
    weights = compute_class_weight(class_weight='balanced', classes=classes, y=y)
    class_weight_map = {cls: w for cls, w in zip(classes, weights)}
    pipe = build_pipeline(
        algorithm=algorithm,
        class_weight=class_weight_map if algorithm=='Logistic Regression' else 'balanced',
        calibrate=calibrate,
        random_state=random_state
    )
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, stratify=y, random_state=random_state)
    pipe.fit(X_train, y_train)
    y_proba = pipe.predict_proba(X_test)[:, 1]
    y_pred = (y_proba >= 0.5).astype(int)
    metrics = {
        'accuracy': accuracy_score(y_test, y_pred),
        'precision': precision_score(y_test, y_pred),
        'recall': recall_score(y_test, y_pred),
        'f1': f1_score(y_test, y_pred),
        'roc_auc': roc_auc_score(y_test, y_proba),
        'confusion': confusion_matrix(y_test, y_pred).tolist(),
        'y_test': y_test,
        'y_proba': y_proba
    }
    return pipe, (X_train, X_test, y_train, y_test), metrics

def optimal_threshold(y_true, y_proba, cost_fp=1, cost_fn=5):
    thresholds = np.linspace(0.1, 0.9, 81)
    best_t, best_cost = 0.5, float('inf')
    for t in thresholds:
        y_pred = (y_proba >= t).astype(int)
        cm = confusion_matrix(y_true, y_pred)
        tn, fp, fn, tp = cm.ravel()
        cost = fp * cost_fp + fn * cost_fn
        if cost < best_cost:
            best_cost, best_t = cost, t
    return best_t, best_cost

def ensure_latlon(df: pd.DataFrame) -> pd.DataFrame:
    if 'lat' not in df.columns or 'lon' not in df.columns:
        if 'location' in df.columns:
            df['lat'] = df['location'].map(lambda c: CITY_LATLON.get(c, (np.nan, np.nan))[0])
            df['lon'] = df['location'].map(lambda c: CITY_LATLON.get(c, (np.nan, np.nan))[1])
    return df

if 'df' not in st.session_state:
    st.session_state.df = generate_synthetic(4000)
if 'model' not in st.session_state:
    st.session_state.model = None
if 'metrics' not in st.session_state:
    st.session_state.metrics = None
if 'threshold' not in st.session_state:
    st.session_state.threshold = 0.5
if 'page' not in st.session_state:
    st.session_state.page = "dashboard"

with st.sidebar:
    st.markdown("""
    <div style="text-align: center; padding: 30px 0;">
        <h1 style="color: #1D4ED8 !important; margin: 0; font-size: 36px; font-weight: 900;">📱 TELECOM AI</h1>
        <h3 style="color: #3B82F6 !important; margin: 10px 0; font-size: 22px; font-weight: 700;">CHURN PREDICTOR PRO</h3>
        <p style="color: #6B7280; margin: 5px 0; font-size: 14px;">Geo‑aware, gamified retention</p>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("<hr style='border: 2px solid #3B82F6;'>", unsafe_allow_html=True)

    selected_page = st.radio(
        "NAVIGATION MENU",
        ["🏠 DASHBOARD", "🔮 PREDICT CHURN", "🗺️ GEO HEATMAP", "🎮 RETENTION SIM", "🤖 TRAIN MODEL"],
        key="sidebar_nav_radio",
        label_visibility="collapsed"
    )
    st.session_state.page = {
        "🏠 DASHBOARD": "dashboard",
        "🔮 PREDICT CHURN": "predict",
        "🗺️ GEO HEATMAP": "geo",
        "🎮 RETENTION SIM": "sim",
        "🤖 TRAIN MODEL": "train"
    }[selected_page]

    st.markdown("<hr style='border: 2px solid #3B82F6;'>", unsafe_allow_html=True)

    df = st.session_state.df
    st.markdown("### 📊 LIVE STATISTICS")
    col1, col2 = st.columns(2)
    with col1:
        st.metric("Total Customers", f"{len(df):,}", delta="+5%")
    with col2:
        st.metric("Churn Rate", f"{df['churn'].mean():.1%}")

def show_dashboard():
    st.markdown('<h1 class="main-title">🏆 TELECOM CHURN PREDICTOR PRO</h1>', unsafe_allow_html=True)
    df = st.session_state.df

    st.markdown('<h2 class="section-header">📊 Key performance indicators</h2>', unsafe_allow_html=True)
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-value">{len(df):,}</div>
            <div class="metric-label">TOTAL CUSTOMERS</div>
        </div>
        """, unsafe_allow_html=True)
    with col2:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-value">{df['churn'].mean():.1%}</div>
            <div class="metric-label">CHURN RATE</div>
        </div>
        """, unsafe_allow_html=True)
    with col3:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-value">₹{df['monthly_charges'].mean():.0f}</div>
            <div class="metric-label">AVG REVENUE</div>
        </div>
        """, unsafe_allow_html=True)
    with col4:
        status = "✅" if st.session_state.model else "⚠️"
        label = "MODEL READY" if st.session_state.model else "TRAIN MODEL"
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-value">{status}</div>
            <div class="metric-label">{label}</div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown('<h2 class="section-header">📈 Visual analytics</h2>', unsafe_allow_html=True)
    c1, c2 = st.columns(2)
    with c1:
        churn_counts = df['churn'].value_counts()
        fig = px.pie(values=churn_counts.values,
                     names=['Retained', 'Churned'],
                     title='<b>CHURN DISTRIBUTION</b>',
                     color_discrete_sequence=['#10B981', '#EF4444'],
                     hole=0.4)
        fig.update_traces(textposition='inside', textinfo='percent+label',
                          textfont=dict(size=16, color='white'))
        fig.update_layout(height=450, showlegend=True)
        st.plotly_chart(fig, use_container_width=True)
    with c2:
        contract_churn = df.groupby('contract')['churn'].mean().reset_index()
        fig = px.bar(contract_churn, x='contract', y='churn',
                     title='<b>CHURN RATE BY CONTRACT TYPE</b>',
                     color='churn', color_continuous_scale='Reds', text_auto='.1%')
        fig.update_layout(yaxis_title="Churn Rate", yaxis_tickformat=".0%", height=450)
        st.plotly_chart(fig, use_container_width=True)

def show_train():
    st.markdown('<h1 class="main-title">🤖 AI MODEL TRAINING</h1>', unsafe_allow_html=True)
    df = st.session_state.df

    st.markdown('<h2 class="subsection-header">📥 Data source</h2>', unsafe_allow_html=True)
    uploaded = st.file_uploader(
        "Upload CSV (optional) with columns: customer_id, tenure, monthly_charges, contract, internet, payment, support_calls, satisfaction, data_usage, location, sentiment, infra_quality, churn",
        type=['csv'], key="train_uploader")
    if uploaded:
        try:
            user_df = pd.read_csv(uploaded)
            required = set([ID_COL, TARGET] + CATEGORICAL + NUMERIC)
            if not required.issubset(set(user_df.columns)):
                st.error("Uploaded CSV missing required columns. Using synthetic data instead.")
            else:
                user_df = ensure_latlon(user_df)
                st.session_state.df = user_df.copy()
                df = st.session_state.df
                st.success(f"Loaded {len(df):,} rows from uploaded data.")
        except Exception as e:
            st.error(f"Failed to read CSV: {e}")

    st.markdown('<h2 class="subsection-header">⚙️ Training configuration</h2>', unsafe_allow_html=True)
    colA, colB, colC, colD = st.columns(4)
    with colA:
        algo = st.selectbox("Algorithm", ["Logistic Regression", "Random Forest"], index=0, key="train_algo")
    with colB:
        test_size = st.slider("Test size (%)", 10, 40, 20, key="train_test_size") / 100.0
    with colC:
        calibrate = st.checkbox("Calibrate probabilities", value=True, key="train_calibrate")
    with colD:
        random_state = st.number_input("Random seed", 0, 9999, 42, key="train_seed")

    st.markdown("---")
    c1, c2, c3 = st.columns([1, 2, 1])
    with c2:
        if st.button("🚀 TRAIN AI MODEL NOW", use_container_width=True, type="primary", key="train_button"):
            with st.spinner("🤖 Training model..."):
                pipe, splits, metrics = train_evaluate(df, test_size=test_size, algorithm=algo, random_state=random_state, calibrate=calibrate)
                st.session_state.model = pipe
                st.session_state.metrics = metrics
                best_t, best_cost = optimal_threshold(metrics['y_test'], metrics['y_proba'], cost_fp=1, cost_fn=5)
                st.session_state.threshold = float(best_t)
                os.makedirs("models", exist_ok=True)
                joblib.dump(pipe, f"models/churn_model_{algo.replace(' ', '_').lower()}.joblib")
                time.sleep(0.4)
                st.success(f"Model trained. Best threshold: {best_t:.2f} (cost={best_cost:.0f})")

    if st.session_state.metrics:
        m = st.session_state.metrics
        st.markdown('<h2 class="subsection-header">📈 Model performance</h2>', unsafe_allow_html=True)
        c1, c2, c3, c4, c5 = st.columns(5)
        with c1: st.metric("Accuracy", f"{m['accuracy']:.2f}")
        with c2: st.metric("Precision", f"{m['precision']:.2f}")
        with c3: st.metric("Recall", f"{m['recall']:.2f}")
        with c4: st.metric("F1 Score", f"{m['f1']:.2f}")
        with c5: st.metric("ROC AUC", f"{m['roc_auc']:.2f}")

        cm = np.array(m['confusion'])
        fig = px.imshow(cm, text_auto=True, color_continuous_scale='Blues',
                        labels=dict(x="Predicted", y="Actual", color="Count"),
                        x=['No Churn', 'Churn'], y=['No Churn', 'Churn'],
                        title="<b>CONFUSION MATRIX</b>")
        fig.update_layout(height=400)
        st.plotly_chart(fig, use_container_width=True)

def show_predict():
    st.markdown('<h1 class="main-title">🔮 PREDICT CUSTOMER CHURN</h1>', unsafe_allow_html=True)

    if not st.session_state.model:
        st.error("## ⚠️ Model not trained")
        st.write("Train the model in the **TRAIN MODEL** section to enable predictions.")
        c1, c2, c3 = st.columns([1,2,1])
        with c2:
            if st.button("🚀 GO TRAIN MODEL NOW", use_container_width=True, key="predict_goto_train"):
                st.session_state.page = "train"
                st.experimental_rerun()
        return

    st.markdown('<h2 class="section-header">📋 Customer information</h2>', unsafe_allow_html=True)
    col1, col2 = st.columns(2)
    with col1:
        customer_id = st.text_input("Customer ID", value=f"IN{np.random.randint(100000, 999999)}", key="predict_cust_id")
        tenure = st.slider("Tenure (months)", 1, 72, 12, key="predict_tenure")
        monthly_charges = st.number_input("Monthly Charges (₹)", 100.0, 3000.0, 499.0, 50.0, key="predict_charges")
        contract = st.selectbox("Contract Type", ["Monthly", "Yearly", "Two Year"], key="predict_contract")
        payment = st.selectbox("Payment Method", ["Credit Card", "Bank", "Cash", "UPI"], key="predict_payment")
    with col2:
        internet = st.selectbox("Internet Service", ["Fiber", "DSL", "None"], key="predict_internet")
        satisfaction = st.slider("Satisfaction (1-10)", 1, 10, 7, key="predict_satisfaction")
        data_usage = st.slider("Data Usage (GB)", 1.0, 300.0, 25.0, key="predict_data")
        support_calls = st.slider("Support Calls", 0, 12, 1, key="predict_support")
        location = st.selectbox("Location", list(CITY_LATLON.keys()), key="predict_location")

    st.markdown('<h3 class="subsection-header">🧠 Emotional & Infra signals</h3>', unsafe_allow_html=True)
    colA, colB = st.columns(2)
    with colA:
        sentiment = st.slider("Customer Sentiment (−1 to +1)", -1.0, 1.0, 0.2, 0.05, key="predict_sentiment")
    with colB:
        infra_quality = st.slider("Infra Quality (0–1)", 0.0, 1.0, 0.7, 0.05, key="predict_infra")

    st.markdown("---")
    c1, c2, c3 = st.columns([1,2,1])
    with c2:
        if st.button("🎯 PREDICT CHURN RISK", type="primary", use_container_width=True, key="predict_button"):
            with st.spinner("🔍 Analyzing customer data..."):
                model = st.session_state.model
                threshold = st.session_state.threshold
                row = pd.DataFrame([{
                    'customer_id': customer_id,
                    'tenure': tenure,
                    'monthly_charges': monthly_charges,
                    'contract': contract,
                    'internet': internet,
                    'payment': payment,
                    'support_calls': support_calls,
                    'satisfaction': satisfaction,
                    'data_usage': data_usage,
                    'location': location,
                    'sentiment': sentiment,
                    'infra_quality': infra_quality
                }])
                proba = model.predict_proba(row[CATEGORICAL + NUMERIC])[:, 1][0]
                risk = "HIGH" if proba >= 0.7 else "MEDIUM" if proba >= 0.3 else "LOW"
                color = "#EF4444" if risk == "HIGH" else "#F59E0B" if risk == "MEDIUM" else "#10B981"

                fig = go.Figure(go.Indicator(
                    mode="gauge+number+delta",
                    value=proba * 100,
                    title={'text': "<b>CHURN RISK SCORE</b>", 'font': {'size': 24, 'color': '#1E40AF'}},
                    delta={'reference': threshold*100, 'increasing': {'color': '#DC2626'}},
                    gauge={
                        'axis': {'range': [0, 100], 'tickwidth': 3, 'tickcolor': '#1E40AF'},
                        'bar': {'color': color, 'thickness': 0.4},
                        'bgcolor': "white",
                        'borderwidth': 6,
                        'bordercolor': "#3B82F6",
                        'steps': [
                            {'range': [0, 30], 'color': '#D1FAE5'},
                            {'range': [30, 70], 'color': '#FEF3C7'},
                            {'range': [70, 100], 'color': '#FEE2E2'}],
                        'threshold': {'line': {'color': color, 'width': 8}, 'thickness': 0.8, 'value': proba*100}
                    },
                    number={'font': {'size': 48, 'color': color, 'family': "Arial Black"}, 'suffix': '%', 'valueformat': '.0f'}
                ))
                fig.update_layout(height=380, margin=dict(t=40, b=20, l=20, r=20))
                st.plotly_chart(fig, use_container_width=True)

                st.markdown("### 🧭 Narrative insight")
                narrative = []
                if contract == "Monthly": narrative.append("short‑term contract increases churn risk")
                if satisfaction <= 4: narrative.append("low satisfaction is a strong churn driver")
                if support_calls >= 6: narrative.append("frequent support calls indicate unresolved issues")
                if monthly_charges >= 1200: narrative.append("high charges may trigger price sensitivity")
                if sentiment < 0: narrative.append("negative sentiment suggests frustration")
                if infra_quality < 0.5: narrative.append("poor infra quality reduces service trust")
                if not narrative:
                    narrative.append("stable signals—no strong churn drivers detected")
                st.write(f"Customer {customer_id} shows {risk.lower()} churn risk ({proba:.1%}). Signals: " +
                         "; ".join(narrative) + ".")

                st.markdown("### 🎯 Recommended actions")
                if risk == "HIGH":
                    st.markdown("""
                    - **Priority:** Immediate
                    - **Retention offer:** 25% off for 3 months, free speed upgrade
                    - **Contact:** Phone call within 24 hours, assign VIP manager
                    - **Service:** Proactive quality check, waive support fees
                    """)
                elif risk == "MEDIUM":
                    st.markdown("""
                    - **Priority:** Proactive
                    - **Retention offer:** 10–15% discount, 5GB data top‑up
                    - **Contact:** SMS + email within 48 hours
                    - **Service:** Satisfaction survey, plan optimization
                    """)
                else:
                    st.markdown("""
                    - **Priority:** Low
                    - **Retention offer:** Loyalty rewards, appreciation message
                    - **Contact:** Monthly newsletter, quarterly check‑in
                    - **Service:** Cross‑sell premium features
                    """)

def show_geo():
    st.markdown('<h1 class="main-title">🗺️ GEO‑AWARE CHURN HEATMAP</h1>', unsafe_allow_html=True)
    df = ensure_latlon(st.session_state.df.copy())

    st.markdown('<h2 class="subsection-header">📍 City‑level risk & revenue</h2>', unsafe_allow_html=True)
    city_stats = df.groupby('location').agg(
        Customers=('customer_id', 'count'),
        Churn_Rate=('churn', 'mean'),
        Avg_Revenue=('monthly_charges', 'mean'),
        Avg_Sentiment=('sentiment', 'mean'),
        Infra_Quality=('infra_quality', 'mean'),
        lat=('lat', 'mean'),
        lon=('lon', 'mean')
    ).reset_index()

    fig = px.scatter_mapbox(
        city_stats, lat="lat", lon="lon", size="Customers", color="Churn_Rate",
        hover_name="location",
        hover_data={"Customers": True, "Churn_Rate": ':.1%', "Avg_Revenue": ':.0f', "Avg_Sentiment": ':.2f', "Infra_Quality": ':.2f', "lat": False, "lon": False},
        color_continuous_scale="Reds", size_max=50, zoom=4, height=600
    )
    fig.update_layout(mapbox_style="carto-positron", title="<b>CHURN HOTSPOTS ACROSS INDIA</b>")
    st.plotly_chart(fig, use_container_width=True)

    st.markdown('<h2 class="subsection-header">🏙️ City comparison</h2>', unsafe_allow_html=True)
    c1, c2 = st.columns(2)
    with c1:
        fig = px.bar(city_stats.sort_values('Churn_Rate', ascending=False), x='location', y='Churn_Rate',
                     title='<b>CHURN RATE BY CITY</b>', color='Churn_Rate', color_continuous_scale='Reds', text='Churn_Rate')
        fig.update_layout(yaxis_tickformat='.0%', height=420)
        st.plotly_chart(fig, use_container_width=True)
    with c2:
        fig = px.bar(city_stats.sort_values('Avg_Revenue', ascending=False), x='location', y='Avg_Revenue',
                     title='<b>AVERAGE REVENUE BY CITY</b>', color='Avg_Revenue', color_continuous_scale='Blues', text='Avg_Revenue')
        fig.update_layout(height=420)
        st.plotly_chart(fig, use_container_width=True)

def show_sim():
    st.markdown('<h1 class="main-title">🎮 RETENTION MISSION CONTROL</h1>', unsafe_allow_html=True)
    df = st.session_state.df.copy()
    model = st.session_state.model

    st.markdown('<h2 class="subsection-header">🧪 What‑if strategy simulator</h2>', unsafe_allow_html=True)
    colA, colB, colC, colD = st.columns(4)
    with colA:
        budget = st.number_input("Budget (₹)", value=500000.0, min_value=0.0, step=10000.0, key="sim_budget")
    with colB:
        discount = st.slider("Discount (%)", 0, 50, 15, key="sim_discount")
    with colC:
        upgrade = st.checkbox("Free speed upgrade", value=True, key="sim_upgrade")
    with colD:
        target_segment = st.selectbox("Target segment", ["All", "High Risk", "Medium Risk", "Low Risk"], key="sim_segment")

    st.markdown("---")
    X = df[CATEGORICAL + NUMERIC]
    if model is None:
        base_proba = (
            0.10
            + df['contract'].map({'Monthly':0.18,'Yearly':0.06,'Two Year':0.03}).fillna(0)
            + (10 - df['satisfaction']) * 0.02
            + df['support_calls'] * 0.025
            - df['tenure'] / 120
            + (df['monthly_charges'] > 1200).astype(int) * 0.05
            - df['sentiment'] * 0.08
            - df['infra_quality'] * 0.06
        ).clip(0.04, 0.85)
    else:
        base_proba = model.predict_proba(X)[:, 1]

    proba = base_proba.copy()
    discount_effect = min(discount * 0.002, 0.25)
    proba -= discount_effect
    if upgrade:
        proba -= 0.02
    proba = np.clip(proba, 0.01, 0.99)

    risk_label = np.where(proba >= 0.7, 'High', np.where(proba >= 0.3, 'Medium', 'Low'))
    df['base_proba'] = base_proba
    df['post_proba'] = proba
    df['risk'] = risk_label

    if target_segment == "All":
        candidates = df
    else:
        seg_map = {"High Risk": "High", "Medium Risk": "Medium", "Low Risk": "Low"}
        candidates = df[df['risk'] == seg_map[target_segment]]

    discount_cost = (discount / 100.0) * candidates['monthly_charges'] * 3
    upgrade_cost = 200.0 if upgrade else 0.0
    total_cost_per_user = discount_cost + upgrade_cost

    prevented = (candidates['base_proba'] - candidates['post_proba']).clip(lower=0)
    total_cost = total_cost_per_user.sum()

    if total_cost > budget and len(candidates) > 0:
        efficiency = (prevented / (total_cost_per_user + 1e-6)).replace([np.inf, -np.inf], 0).fillna(0)
        pick = candidates.assign(prevented=prevented, cost=total_cost_per_user, eff=efficiency)
        pick = pick.sort_values('eff', ascending=False)
        pick['cum_cost'] = pick['cost'].cumsum()
        selected = pick[pick['cum_cost'] <= budget].copy()
    else:
        selected = candidates.copy()
        selected = selected.assign(prevented=prevented, cost=total_cost_per_user)

    selected_count = len(selected)
    expected_prevented_churn = selected['prevented'].sum()
    total_cost = selected['cost'].sum()
    avg_revenue = df['monthly_charges'].mean()
    revenue_saved = expected_prevented_churn * avg_revenue * 12

    st.markdown('<h3 class="subsection-header">📊 Simulation results</h3>', unsafe_allow_html=True)
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.metric("Targeted Users", f"{selected_count:,}")
    with c2:
        st.metric("Estimated Prevented Churn", f"{expected_prevented_churn:.1f}")
    with c3:
        st.metric("Campaign Cost (₹)", f"{total_cost:,.0f}")
    with c4:
        st.metric("Estimated Revenue Saved (₹)", f"{revenue_saved:,.0f}")

    st.markdown("### 📈 City impact preview")
    if selected_count > 0:
        city_impact = selected.groupby('location').agg(
            Targeted=('customer_id', 'count'),
            Prevented_Churn=('prevented', 'sum'),
            Cost=('cost', 'sum')
        ).reset_index().sort_values('Prevented_Churn', ascending=False)
        fig = px.bar(city_impact, x='location', y='Prevented_Churn', title='<b>PREVENTED CHURN BY CITY</b>', color='Prevented_Churn', color_continuous_scale='Greens')
        st.plotly_chart(fig, use_container_width=True)
        st.dataframe(city_impact)
    else:
        st.info("No users selected under current budget/filters.")

def main():
    page = st.session_state.page
    if page == "dashboard":
        show_dashboard()
    elif page == "predict":
        show_predict()
    elif page == "geo":
        show_geo()
    elif page == "sim":
        show_sim()
    elif page == "train":
        show_train()
    else:
        show_dashboard()

if __name__ == "__main__":
    main()


