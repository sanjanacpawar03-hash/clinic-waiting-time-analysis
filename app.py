import streamlit as st
import pandas as pd
import numpy as np
import sqlite3
import threading, time
from datetime import datetime

# Viz
import plotly.express as px
import plotly.graph_objects as go
import matplotlib.pyplot as plt

# ML
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error

# Explainability
import shap

# DL
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from sklearn.preprocessing import MinMaxScaler

st.set_page_config(layout="wide")

# ---------------- UI ----------------
st.markdown("""
<style>

/* BACKGROUND */
.stApp {
  background: radial-gradient(circle at top left,#e0e7ff,#f8fafc);
}

/* CONTAINER */
.block-container {
  padding: 3.5rem 2rem;
}

/* SIDEBAR */
section[data-testid="stSidebar"] {
  background:#0f172a;
}
section[data-testid="stSidebar"] * {
  color:white;
}

/* TOPBAR */
.topbar {
  padding:12px;
  border-radius:14px;
  background:rgba(255,255,255,0.7);
  backdrop-filter:blur(10px);
  display:flex;
  justify-content:space-between;
  margin-bottom:20px;
}

/* CARD */
.card {
  background: rgba(255,255,255,0.75);
  backdrop-filter: blur(14px);
  border-radius: 18px;
  padding: 16px;
  margin-bottom: 24px;
  box-shadow: 0 10px 30px rgba(0,0,0,0.12);
  transition:0.3s;
}
.card:hover {
  transform:translateY(-5px);
}

/* KPI */
.kpi-title {color:#6b7280;font-size:13px;text-align:center;}
.kpi-value {font-size:28px;font-weight:bold;text-align:center;color:#4f46e5;}

/* CHART */
.chart-title {font-weight:600;margin-bottom:6px;}
.chart-caption {font-size:12px;color:gray;}

div[data-testid="stHorizontalBlock"] {gap:25px;}
.section-gap {margin-top:30px;}

</style>
""", unsafe_allow_html=True)

# ---------------- DB ----------------
conn = sqlite3.connect("clinic.db", check_same_thread=False)
cur = conn.cursor()
cur.execute("CREATE TABLE IF NOT EXISTS users(id INTEGER PRIMARY KEY, username TEXT, password TEXT)")
conn.commit()

# ---------------- SESSION ----------------
if "user" not in st.session_state:
    st.session_state.user=None

# ---------------- DATA ----------------
def generate():
    doctors=["Dr. Smith","Dr. Kumar","Dr. Patel","Dr. Khan"]
    return [{
        "time": datetime.now().strftime("%H:%M:%S"),
        "hour": datetime.now().hour,
        "doctor": np.random.choice(doctors),
        "patients": np.random.randint(5,30),
        "wait": round(max(5,np.random.normal(30,10)),2)
    } for _ in range(100)]

if "data" not in st.session_state:
    st.session_state.data=generate()

def stream():
    while True:
        st.session_state.data.append({
            "time": datetime.now().strftime("%H:%M:%S"),
            "hour": datetime.now().hour,
            "doctor": np.random.choice(["Dr. Smith","Dr. Kumar","Dr. Patel","Dr. Khan"]),
            "patients": np.random.randint(5,30),
            "wait": round(max(5,np.random.normal(30,10)),2)
        })
        if len(st.session_state.data)>300:
            st.session_state.data.pop(0)
        time.sleep(2)

if "thread" not in st.session_state:
    threading.Thread(target=stream, daemon=True).start()
    st.session_state.thread=True

# ---------------- LOGIN ----------------
if st.session_state.user is None:

    st.title("🏥 Clinic Intelligence System")

    menu=st.selectbox("Menu",["Login","Register"])
    u=st.text_input("Username")
    p=st.text_input("Password",type="password")

    if menu=="Login" and st.button("Login"):
        cur.execute("SELECT * FROM users WHERE username=? AND password=?", (u,p))
        if cur.fetchone():
            st.session_state.user=u
            st.rerun()
        else:
            st.error("Invalid login")

    if menu=="Register" and st.button("Register"):
        cur.execute("INSERT INTO users(username,password) VALUES(?,?)",(u,p))
        conn.commit()
        st.success("Registered")

# ---------------- MAIN ----------------
else:

    st.sidebar.title("🏥 Navigation")
    page=st.sidebar.radio("Go to",["Dashboard","Analytics","ML Models","AI Insights","Live Data","Logout"])

    df=pd.DataFrame(st.session_state.data)

    # ---------------- TOPBAR ----------------
    st.markdown("""
    <div class="topbar">
      <div><b>Clinic Intelligence Dashboard</b></div>
      <div style="color:green;">● Live</div>
    </div>
    """, unsafe_allow_html=True)

    # ---------------- HELPER ----------------
    def chart(title, fig, caption=""):
        fig.update_layout(height=300,margin=dict(l=10,r=10,t=30,b=10))
        st.markdown("<div class='card'>",unsafe_allow_html=True)
        st.markdown(f"<div class='chart-title'>{title}</div>",unsafe_allow_html=True)
        st.plotly_chart(fig,use_container_width=True)
        if caption:
            st.markdown(f"<div class='chart-caption'>{caption}</div>",unsafe_allow_html=True)
        st.markdown("</div>",unsafe_allow_html=True)

    # ---------------- DASHBOARD ----------------
    if page=="Dashboard":

        avg=df.wait.mean()
        peak=df.groupby("hour")['wait'].mean().idxmax()

        c1,c2,c3,c4=st.columns(4)

        def kpi(t,v):
            st.markdown(f"<div class='card'><div class='kpi-title'>{t}</div><div class='kpi-value'>{v}</div></div>",unsafe_allow_html=True)

        kpi("Avg Wait",round(avg,1))
        kpi("Patients",len(df))
        kpi("Peak Hour",peak)
        kpi("Max Wait",round(df.wait.max(),1))

        col1,col2=st.columns(2)
        with col1:
            chart("⏱ Waiting Trend",px.line(df.tail(80),x="time",y="wait"),"Real-time trend")
        with col2:
            chart("📊 Distribution",px.histogram(df,x="wait"),"Wait distribution")

  
        col3,col4=st.columns(2)
        with col3:
            chart("📦 Box Plot",px.box(df,y="wait"))
        with col4:
            chart("📈 Area Trend",px.area(df.tail(80),x="time",y="wait"))

    # ---------------- ANALYTICS ----------------
    elif page=="Analytics":

        col1,col2=st.columns(2)
        with col1:
            chart("🔍 Hour vs Wait",px.scatter(df,x="hour",y="wait"))
        with col2:
            chart("🎻 Density",px.violin(df,y="wait"))

        st.markdown("<div class='section-gap'></div>",unsafe_allow_html=True)

        col3,col4=st.columns(2)
        with col3:
            chart("⏰ Avg by Hour",px.bar(df.groupby("hour")['wait'].mean().reset_index(),x="hour",y="wait"))
        with col4:
            chart("👨‍⚕️ Doctor Comparison",px.bar(df.groupby("doctor")['wait'].mean().reset_index(),x="doctor",y="wait"))

    # ---------------- ML MODELS ----------------
    elif page=="ML Models":

        df_ml=df.copy()
        df_ml['rolling']=df_ml.wait.rolling(5).mean()
        df_ml['lag']=df_ml.wait.shift(1)
        df_ml=df_ml.dropna()

        df_ml=pd.get_dummies(df_ml,columns=['doctor'])

        X=df_ml.drop(columns=['wait','time'])
        y=df_ml['wait']

        X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2)

        rf=RandomForestRegressor().fit(X_train,y_train)
        pred=rf.predict(X_test)

        st.metric("RMSE",round(np.sqrt(mean_squared_error(y_test,pred)),2))

        chart("🎯 Actual vs Predicted",px.scatter(x=y_test,y=pred))

        try:
            explainer=shap.TreeExplainer(rf)
            shap_values=explainer.shap_values(X_test)
            fig,ax=plt.subplots()
            shap.summary_plot(shap_values,X_test,show=False)
            st.pyplot(fig)
        except:
            st.warning("SHAP issue")

        try:
            scaler=MinMaxScaler()
            scaled=scaler.fit_transform(df_ml[['wait']])
            X_lstm=[]
            for i in range(10,len(scaled)):
                X_lstm.append(scaled[i-10:i])
            X_lstm=np.array(X_lstm)

            model=Sequential([LSTM(50,input_shape=(10,1)),Dense(1)])
            model.compile(optimizer='adam',loss='mse')
            model.fit(X_lstm,scaled[10:],epochs=3,verbose=0)

            pred=model.predict(X_lstm[-1].reshape(1,10,1),verbose=0)
            st.success(f"LSTM Prediction: {scaler.inverse_transform(pred)[0][0]:.2f}")
        except:
            st.warning("LSTM issue")

    # ---------------- AI ----------------
    elif page=="AI Insights":

        model=LinearRegression().fit(df[['hour']],df['wait'])
        pred=model.predict([[datetime.now().hour]])[0]

        st.metric("Predicted Wait",round(pred,2))

        fig = go.Figure(go.Indicator(
            mode="gauge+number",
            value=int(pred*2),
            title={'text': "Risk Level"},
            gauge={'axis': {'range': [0,100]}}
        ))
        st.plotly_chart(fig,use_container_width=True)

    # ---------------- LIVE ----------------
    elif page=="Live Data":
        st.dataframe(df.tail(50))
        st.download_button("Download CSV",df.to_csv(index=False),"clinic.csv")

    elif page=="Logout":
        st.session_state.user=None
        st.rerun()