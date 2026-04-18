import streamlit as st
import pandas as pd
import pickle
import plotly.graph_objects as go
import plotly.express as px

# --- Page Configuration ---
st.set_page_config(
    page_title="Heart Health Predictor",
    page_icon="❤️",
    layout="wide",
    initial_sidebar_state="expanded"
)


# --- Load Data and Model ---
@st.cache_data
def load_data(path):
    """Loads the heart disease dataset."""
    try:
        return pd.read_csv(path)
    except FileNotFoundError:
        st.error("Error: The dataset file 'heart.csv' was not found.")
        st.info("Please ensure 'heart.csv' is in the same directory as the app.")
        return None


@st.cache_resource
def load_model_data(path):
    """Loads the dictionary containing the model and columns from the pickle file."""
    try:
        with open(path, 'rb') as file:
            model_data = pickle.load(file)
        return model_data
    except FileNotFoundError:
        st.error("Error: The model file 'heart_disease_model.pkl' was not found.")
        st.info("Ensure the model file is in the same directory.")
        return None


model_data = load_model_data('heart_disease_model.pkl')
df_original = load_data('heart.csv')

# --- Initialize Model and Columns ---
model = None
model_columns = None

if model_data:
    
    model = model_data.get('model')
    model_columns = model_data.get('columns')

if model is None or model_columns is None or df_original is None:
    st.error("Could not load the model or its required column data. Please check the 'heart_disease_model.pkl' file.")
    st.stop()

# --- Sidebar for User Input -
st.sidebar.markdown(
    """
    <h1 style='text-align: center; font-size: 50px;'>🩺</h1>
    <h2 style='text-align: center;'>Patient Vitals</h2>
    """,
    unsafe_allow_html=True
)

st.sidebar.info("Enter the patient's information below to get a prediction.")


def user_input_features():
    """Creates sidebar input widgets for each feature and returns a DataFrame."""
    age = st.sidebar.slider('Age', 28, 77, 50)
    sex = st.sidebar.radio('Sex', ('Male', 'Female'))
    chest_pain_type = st.sidebar.selectbox('Chest Pain Type',
                                           ('Atypical Angina', 'Non-Anginal Pain', 'Asymptomatic', 'Typical Angina'))
    resting_bp = st.sidebar.slider('Resting Blood Pressure (mm Hg)', 80, 200, 120)
    cholesterol = st.sidebar.slider('Cholesterol (mg/dl)', 85, 603, 200)
    fasting_bs = st.sidebar.radio('Fasting Blood Sugar > 120 mg/dl', ('No', 'Yes'))
    resting_ecg = st.sidebar.selectbox('Resting ECG',
                                       ('Normal', 'ST-T Wave Abnormality', 'Left Ventricular Hypertrophy'))
    max_hr = st.sidebar.slider('Maximum Heart Rate Achieved', 60, 202, 150)
    exercise_angina = st.sidebar.radio('Exercise-Induced Angina', ('No', 'Yes'))
    oldpeak = st.sidebar.slider('ST Depression (Oldpeak)', -2.6, 6.2, 1.0, 0.1)
    st_slope = st.sidebar.selectbox('ST Slope', ('Upsloping', 'Flat', 'Downsloping'))

    data = {
        'Age': age, 'RestingBP': resting_bp, 'Cholesterol': cholesterol,
        'FastingBS': 1 if fasting_bs == 'Yes' else 0, 'MaxHR': max_hr,
        'Oldpeak': oldpeak, 'Sex_M': 1 if sex == 'Male' else 0,
        'ChestPainType_ATA': 1 if chest_pain_type == 'Atypical Angina' else 0,
        'ChestPainType_NAP': 1 if chest_pain_type == 'Non-Anginal Pain' else 0,
        'ChestPainType_TA': 1 if chest_pain_type == 'Typical Angina' else 0,
        'RestingECG_Normal': 1 if resting_ecg == 'Normal' else 0,
        'RestingECG_ST': 1 if resting_ecg == 'ST-T Wave Abnormality' else 0,
        'ExerciseAngina_Y': 1 if exercise_angina == 'Yes' else 0,
        'ST_Slope_Flat': 1 if st_slope == 'Flat' else 0,
        'ST_Slope_Up': 1 if st_slope == 'Upsloping' else 0
    }
    return pd.DataFrame(data, index=[0])


input_df = user_input_features()


st.markdown(
    "<h1 style='text-align: center; font-size: 100px;'>🫀</h1>",
    unsafe_allow_html=True
)

st.title('❤️ Heart Health Prediction System')
st.markdown("This app uses a `RandomForestClassifier` to predict the likelihood of heart disease.")


tab1, tab2 = st.tabs(["📈 Prediction Analysis", "📊 3D Feature Explorer"])

with tab1:
    st.header("Prediction Results")
    st.write("Click the button below to get a risk assessment based on the input data.")

    if st.button('**Click to Predict Risk**', use_container_width=True, type="primary"):
      
        final_input = input_df.reindex(columns=model_columns, fill_value=0)

        prediction = model.predict(final_input)
        prediction_proba = model.predict_proba(final_input)

        prob_heart_disease = prediction_proba[0][1]

        with st.container(border=True):
            col1, col2 = st.columns([1, 2])
            with col1:
                st.metric(
                    label="**Heart Disease Risk**",
                    value=f"{prob_heart_disease * 100:.1f}%",
                    delta="High Risk" if prediction[0] == 1 else "Low Risk",
                    delta_color="inverse"
                )

                if prediction[0] == 1:
                    st.error('**Result:** The model indicates a high probability of Heart Disease.')
                else:
                    st.success('**Result:** The model indicates a low probability of Heart Disease.')

            with col2:
                # Gauge Chart
                fig = go.Figure(go.Indicator(
                    mode="gauge+number",
                    value=prob_heart_disease * 100,
                    title={'text': "Risk Score"},
                    gauge={
                        'axis': {'range': [None, 100]},
                        'bar': {'color': "#2E86C1"},
                        'steps': [
                            {'range': [0, 40], 'color': "#58D68D"},
                            {'range': [40, 70], 'color': "#F5B041"},
                            {'range': [70, 100], 'color': "#EC7063"}],
                    }))
                fig.update_layout(height=250, margin=dict(l=10, r=10, b=10, t=50))
                st.plotly_chart(fig, use_container_width=True)

with tab2:
    st.header("Patient Data in Context")
    st.markdown(
        "This 3D plot shows where the current patient's data (the **RED** diamond) fits within the broader dataset. This helps visualize if their metrics are outliers.")

    # 3D scatter plot
    fig_3d = px.scatter_3d(
        df_original,
        x='Age', y='Cholesterol', z='MaxHR',
        color='HeartDisease',
        color_discrete_map={0: 'green', 1: 'orange'},
        hover_name='Sex',
        opacity=0.6,
        title="Age vs. Cholesterol vs. Max Heart Rate"
    )


    fig_3d.add_trace(go.Scatter3d(
        x=input_df['Age'], y=input_df['Cholesterol'], z=input_df['MaxHR'],
        mode='markers',
        marker=dict(size=10, color='red', symbol='diamond'),
        name='Your Input'
    ))

    fig_3d.update_layout(
        scene=dict(
            xaxis_title='Age', yaxis_title='Cholesterol', zaxis_title='Max Heart Rate'
        ),
        margin=dict(l=0, r=0, b=0, t=40)
    )
    st.plotly_chart(fig_3d, use_container_width=True)


with st.expander("Show Current Input Parameters"):
    st.dataframe(input_df.T.rename(columns={0: 'Values'}))

st.divider()
st.write(
    "*Disclaimer: This is a tool for educational purposes and is not a substitute for professional medical advice. Always consult a healthcare provider for any health concerns.*")
