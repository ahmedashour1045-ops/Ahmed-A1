import streamlit as st
import pandas as pd
import numpy as np
import joblib
import plotly.graph_objects as go
import plotly.express as px
import time

# --------------------------------------------------
# Page Configuration
# --------------------------------------------------

st.set_page_config(
    page_title="Student Grade Predictor AI",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --------------------------------------------------
# Custom CSS
# --------------------------------------------------

st.markdown("""

<style>

html, body, [class*="css"]{

background:linear-gradient(135deg,#E0EAFC,#CFDEF3);

font-family:'Segoe UI';

}


.main{

padding-top:20px;

}


.block-container{

padding-top:1rem;

padding-bottom:1rem;

}


h1{

color:#0F172A;

font-size:52px;

font-weight:bold;

text-align:center;

}


h2{

color:#2563EB;

}


.hero{

background:linear-gradient(90deg,#2563EB,#38BDF8);

padding:30px;

border-radius:25px;

color:white;

box-shadow:0px 10px 25px rgba(0,0,0,.15);

margin-bottom:25px;

}


.card{

background:white;

padding:25px;

border-radius:20px;

box-shadow:0px 8px 25px rgba(0,0,0,.12);

margin-bottom:20px;

}


.result{

background:linear-gradient(90deg,#10B981,#34D399);

padding:30px;

border-radius:25px;

text-align:center;

color:white;

font-size:28px;

font-weight:bold;

box-shadow:0px 10px 20px rgba(0,0,0,.2);

}


.footer{

text-align:center;

padding:25px;

color:gray;

font-size:16px;

}


div.stButton>button{

width:100%;

height:60px;

border:none;

border-radius:15px;

background:linear-gradient(90deg,#2563EB,#06B6D4);

color:white;

font-size:22px;

font-weight:bold;

transition:.3s;

}


div.stButton>button:hover{

transform:scale(1.05);

background:linear-gradient(90deg,#1D4ED8,#0284C7);

}


[data-testid="metric-container"]{

background:white;

padding:20px;

border-radius:18px;

box-shadow:0px 6px 18px rgba(0,0,0,.12);

}


[data-testid="stSidebar"]{

background:#0F172A;

}


[data-testid="stSidebar"] *{

color:white;

}

</style>

""",unsafe_allow_html=True)

# --------------------------------------------------
# Load Model
# --------------------------------------------------

@st.cache_resource
def load_model():

    return joblib.load("student_gpa_model.pkl")


try:

    model=load_model()

except Exception as e:

    st.error(e)

    st.stop()

# --------------------------------------------------
# Hero Section
# --------------------------------------------------

st.markdown("""

<div class="hero">

<h1>🎓 Student Grade Class Predictor</h1>

<h3 style="text-align:center;">

Artificial Intelligence Based Prediction System

</h3>

</div>

""",unsafe_allow_html=True)

# --------------------------------------------------
# Dashboard
# --------------------------------------------------

m1,m2,m3,m4=st.columns(4)

m1.metric("🤖 Model","Random Forest")

m2.metric("📊 Features","13")

m3.metric("🎯 Classes","5")

m4.metric("⚡ AI","Ready")

st.markdown("<br>",unsafe_allow_html=True)

left,right=st.columns([1,2])
# ==================================================
# Left Column - Inputs
# ==================================================

with left:

    st.markdown('<div class="card">', unsafe_allow_html=True)

    st.subheader("📝 Student Information")

    age = st.number_input(
        "🎂 Age",
        min_value=15,
        max_value=30,
        value=18
    )

    gender = st.selectbox(
        "👤 Gender",
        [0, 1],
        format_func=lambda x: "Male" if x == 0 else "Female"
    )

    ethnicity = st.selectbox(
        "🌍 Ethnicity",
        [0, 1, 2, 3]
    )

    parental_education = st.selectbox(
        "🎓 Parental Education",
        [0, 1, 2, 3, 4]
    )

    study_time = st.slider(
        "📚 Study Time Weekly",
        0.0,
        40.0,
        10.0
    )

    absences = st.slider(
        "📅 Absences",
        0,
        50,
        5
    )

    tutoring = st.selectbox(
        "👨‍🏫 Tutoring",
        [0, 1],
        format_func=lambda x: "No" if x == 0 else "Yes"
    )

    parental_support = st.selectbox(
        "❤️ Parental Support",
        [0, 1, 2, 3, 4]
    )

    extracurricular = st.selectbox(
        "🎭 Extracurricular",
        [0, 1],
        format_func=lambda x: "No" if x == 0 else "Yes"
    )

    sports = st.selectbox(
        "⚽ Sports",
        [0, 1],
        format_func=lambda x: "No" if x == 0 else "Yes"
    )

    music = st.selectbox(
        "🎵 Music",
        [0, 1],
        format_func=lambda x: "No" if x == 0 else "Yes"
    )

    volunteering = st.selectbox(
        "🤝 Volunteering",
        [0, 1],
        format_func=lambda x: "No" if x == 0 else "Yes"
    )

    gpa = st.slider(
        "📈 GPA",
        0.0,
        4.0,
        2.50,
        0.01
    )

    predict_btn = st.button("🚀 Predict Grade")

    st.markdown("</div>", unsafe_allow_html=True)


# ==================================================
# Right Column
# ==================================================

with right:

    st.markdown('<div class="card">', unsafe_allow_html=True)

    st.subheader("📊 Prediction Dashboard")

    st.info("Fill in the student information then click Predict.")

    if predict_btn:

        progress = st.progress(0)

        with st.spinner("🤖 AI is analyzing the student's data..."):

            for i in range(100):
                progress.progress(i + 1)
                time.sleep(0.01)

        progress.empty()

        input_df = pd.DataFrame(
            [[
                age,
                gender,
                ethnicity,
                parental_education,
                study_time,
                absences,
                tutoring,
                parental_support,
                extracurricular,
                sports,
                music,
                volunteering,
                gpa
            ]],
            columns=[
                "Age",
                "Gender",
                "Ethnicity",
                "ParentalEducation",
                "StudyTimeWeekly",
                "Absences",
                "Tutoring",
                "ParentalSupport",
                "Extracurricular",
                "Sports",
                "Music",
                "Volunteering",
                "GPA"
            ]
        )

        prediction = model.predict(input_df)[0]

        st.markdown(
            f"""
            <div class='result'>
                🏆 Predicted Grade Class
                <br><br>
                <span style='font-size:55px;'>{prediction}</span>
            </div>
            """,
            unsafe_allow_html=True
        )

        st.markdown("---")

        st.subheader("📋 Student Summary")

        c1, c2 = st.columns(2)

        with c1:
            st.metric("Age", age)
            st.metric("Study Hours", study_time)
            st.metric("Absences", absences)

        with c2:
            st.metric("GPA", gpa)
            st.metric("Sports", "Yes" if sports else "No")
            st.metric("Music", "Yes" if music else "No")

        st.markdown("---")

        st.subheader("📈 GPA Chart")

        fig = px.bar(
            x=["Student GPA"],
            y=[gpa],
            color=[gpa],
            color_continuous_scale="Blues"
        )

        fig.update_layout(
            height=350,
            showlegend=False,
            yaxis_range=[0, 4],
            margin=dict(l=20, r=20, t=40, b=20)
        )

        st.plotly_chart(fig, use_container_width=True)

        st.subheader("🥧 GPA Distribution")

        pie = go.Figure(
            data=[
                go.Pie(
                    labels=["GPA", "Remaining"],
                    values=[gpa, 4 - gpa],
                    hole=.55
                )
            ]
        )

        pie.update_layout(height=350)

        st.plotly_chart(pie, use_container_width=True)

    st.markdown("</div>", unsafe_allow_html=True)
    # ==================================================
# AI Recommendation + Gauge + Report
# ==================================================

if predict_btn:

    st.markdown("---")

    # -------------------------------
    # Gauge Meter
    # -------------------------------

    st.subheader("🎯 Prediction Gauge")

    gauge_value = (4 - float(prediction)) * 25

    gauge = go.Figure(go.Indicator(
        mode="gauge+number",
        value=gauge_value,
        title={"text": "Student Performance"},
        gauge={
            "axis": {"range": [0, 100]},
            "bar": {"color": "#2563EB"},
            "steps": [
                {"range": [0, 20], "color": "#EF4444"},
                {"range": [20, 40], "color": "#F97316"},
                {"range": [40, 60], "color": "#FACC15"},
                {"range": [60, 80], "color": "#22C55E"},
                {"range": [80, 100], "color": "#15803D"},
            ]
        }
    ))

    gauge.update_layout(height=350)

    st.plotly_chart(gauge, use_container_width=True)

    # -------------------------------
    # Radar Chart
    # -------------------------------

    st.subheader("📊 Student Profile")

    radar = go.Figure()

    radar.add_trace(go.Scatterpolar(
        r=[
            study_time,
            max(0, 50 - absences),
            gpa,
            sports * 20,
            music * 20,
            volunteering * 20
        ],
        theta=[
            "Study",
            "Attendance",
            "GPA",
            "Sports",
            "Music",
            "Volunteer"
        ],
        fill='toself'
    ))

    radar.update_layout(
        polar=dict(radialaxis=dict(visible=True)),
        showlegend=False,
        height=500
    )

    st.plotly_chart(radar, use_container_width=True)

    # -------------------------------
    # AI Recommendation
    # -------------------------------

    st.subheader("💡 AI Recommendation")

    if prediction == 0:
        st.error("""
🔴 High Risk

• Increase study hours.
• Reduce absences.
• Attend tutoring sessions.
• Improve GPA gradually.
""")

    elif prediction == 1:
        st.warning("""
🟠 Needs Improvement

• Practice more every week.
• Focus on difficult subjects.
• Attend classes regularly.
""")

    elif prediction == 2:
        st.info("""
🟡 Average Performance

• Keep studying consistently.
• Participate in extracurricular activities.
• Maintain attendance.
""")

    elif prediction == 3:
        st.success("""
🟢 Very Good

• Keep your current performance.
• Continue practicing.
• Help classmates if possible.
""")

    else:

        st.balloons()

        st.success("""
🏆 Excellent Student

• Outstanding academic performance.
• Keep up the great work.
• Maintain your GPA.
• Continue participating in activities.
""")

    # -------------------------------
    # Download Report
    # -------------------------------

    report = pd.DataFrame({

        "Feature":[
            "Age",
            "Gender",
            "Ethnicity",
            "Parent Education",
            "Study Hours",
            "Absences",
            "Tutoring",
            "Parent Support",
            "Extracurricular",
            "Sports",
            "Music",
            "Volunteering",
            "GPA",
            "Prediction"
        ],

        "Value":[
            age,
            gender,
            ethnicity,
            parental_education,
            study_time,
            absences,
            tutoring,
            parental_support,
            extracurricular,
            sports,
            music,
            volunteering,
            gpa,
            prediction
        ]

    })

    csv = report.to_csv(index=False).encode("utf-8")

    st.download_button(

        "📥 Download Prediction Report",

        csv,

        file_name="student_prediction_report.csv",

        mime="text/csv"

    )

# ==================================================
# Footer
# ==================================================

st.markdown("---")

st.markdown("""

<div class="footer">

<h3>🎓 Student Grade Class Predictor</h3>

Built with ❤️ using

<b>Python • Streamlit • Scikit-Learn • Plotly</b>

<br><br>

© 2026 All Rights Reserved

</div>

""", unsafe_allow_html=True)