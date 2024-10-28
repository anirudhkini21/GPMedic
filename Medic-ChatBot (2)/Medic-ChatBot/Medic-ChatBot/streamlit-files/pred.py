import pickle
import streamlit as st
from streamlit_option_menu import option_menu
import PyPDF2  # To read PDF files
import re  # For regular expressions

# Function to extract values from the report
def extract_values_from_text(text):
    # Example extraction logic using regular expressions
    values = {
        "Pregnancies": None,
        "Glucose": None,
        "BloodPressure": None,
        "SkinThickness": None,
        "Insulin": None,
        "BMI": None,
        "DiabetesPedigreeFunction": None,
        "Age": None,
        "AgeHeart": None,
        "Chol": None,
        "Sex": None,
        "CP": None,
        "Trestbps": None,
        "FBS": None,
        "RestECG": None,
        "Thalach": None,
        "Exang": None,
        "Oldpeak": None,
        "Slope": None,
        "CA": None,
        "Thal": None,
        # New values for Parkinson's disease
        "MDVPFo": None,
        "MDVPFhi": None,
        "MDVPFlo": None,
        "Jitter_percent": None,
        "Jitter_Abs": None,
        "RAP": None,
        "PPQ": None,
        "DDP": None,
        "Shimmer": None,
        "Shimmer_dB": None,
        "APQ3": None,
        "APQ5": None,
        "APQ": None,
        "DDA": None,
        "NHR": None,
        "HNR": None,
        "RPDE": None,
        "DFA": None,
        "spread1": None,
        "spread2": None,
        "D2": None,
        "PPE": None,
        
    }

    # Regular expressions to find values (customize these based on your report format)
    patterns = {
        "Pregnancies": r"Pregnancies:\s*(\d+)",
        "Glucose": r"Glucose Level:\s*(\d+)",
        "BloodPressure": r"Blood Pressure value:\s*(\d+)",
        "SkinThickness": r"Skin Thickness value:\s*(\d+)",
        "Insulin": r"Insulin Level:\s*(\d+)",
        "BMI": r"BMI value:\s*([\d.]+)",
        "DiabetesPedigreeFunction": r"Diabetes Pedigree Function value:\s*([\d.]+)",
        "Age": r"Age of the Person:\s*(\d+)",
        "AgeHeart": r"Age:\s*(\d+)",
        "Chol": r"Serum Cholestrol in mg/dl:\s*(\d+)",
        "Sex": r"Sex:\s*(\d+)",
        "CP": r"Chest Pain types:\s*(\d+)",
        "Trestbps": r"Resting Blood Pressure:\s*(\d+)",
        "FBS": r"Fasting Blood Sugar > 120 mg/dl:\s*(\d+)",
        "RestECG": r"Resting Electrocardiographic results:\s*(\d+)",
        "Thalach": r"Maximum Heart Rate achieved:\s*(\d+)",
        "Exang": r"Exercise Induced Angina:\s*(\d+)",
        "Oldpeak": r"ST depression induced by exercise:\s*([\d.]+)",
        "Slope": r"Slope of the peak exercise ST segment:\s*(\d+)",
        "CA": r"Major vessels colored by flourosopy:\s*(\d+)",
        "Thal": r"thal:\s*(\d+)",
        # New patterns for Parkinson's disease
        "MDVPFo": r"MDVPFo\s*\(Hz\):\s*(-?[\d.]+)",
    "MDVPFhi": r"MDVPFhi\s*\(Hz\):\s*(-?[\d.]+)",
    "MDVPFlo": r"MDVPFlo\s*\(Hz\):\s*(-?[\d.]+)",
    "Jitter_percent": r"MDVPJitter\s*\(%\):\s*(-?[\d.]+)",
    "Jitter_Abs": r"MDVPJitter\s*\(Abs\):\s*(-?[\d.]+)",
    "RAP": r"MDVPRAP:\s*(-?[\d.]+)",
    "PPQ": r"MDVPPPQ:\s*(-?[\d.]+)",
    "DDP": r"JitterDDP:\s*(-?[\d.]+)",
    "Shimmer": r"MDVPShimmer:\s*(-?[\d.]+)",
    "Shimmer_dB": r"MDVPShimmer\s*\(dB\):\s*(-?[\d.]+)",
    "APQ3": r"ShimmerAPQ3:\s*(-?[\d.]+)",
    "APQ5": r"ShimmerAPQ5:\s*(-?[\d.]+)",
    "APQ": r"MDVPAPQ:\s*(-?[\d.]+)",
    "DDA": r"ShimmerDDA:\s*(-?[\d.]+)",
    "NHR": r"NHR:\s*(-?[\d.]+)",
    "HNR": r"HNR:\s*(-?[\d.]+)",
    "RPDE": r"RPDE:\s*(-?[\d.]+)",
    "DFA": r"DFA:\s*(-?[\d.]+)",
    "spread1": r"spread1:\s*(-?[\d.]+)",
    "spread2": r"spread2:\s*(-?[\d.]+)",
    "D2": r"D2:\s*(-?[\d.]+)",
    "PPE": r"PPE:\s*(-?[\d.]+)",
    }

    for key, pattern in patterns.items():
        match = re.search(pattern, text)
        if match:
            values[key] = float(match.group(1)) if '.' in match.group(1) else int(match.group(1))

    return values

# Set the page title and icon
st.set_page_config(page_title="Multiple Disease Prediction System", page_icon=":hospital:", layout="wide")

# Custom CSS for a light and black-and-white theme
custom_css = """
    <style>
        body {
            color: black;
            background-color: white;
        }
        .stApp {
            max-width: 1200px;
            margin: 0 auto;
        }
        .sidebar .sidebar-content {
            background-color: black;
        }
        .sidebar .sidebar-content .stSelectbox .stSelectbox-list .stSelectbox-options-container .stSelectbox-options .stSelectbox-option:hover {
            background-color: white !important;
            color: black !important;
        }
        .sidebar .sidebar-content .stSelectbox .stSelectbox-list .stSelectbox-options-container .stSelectbox-options .stSelectbox-option[data-baseweb="menu-option"] {
            color: black !important;
            background-color: white !important;
        }
    </style>
"""
st.markdown(custom_css, unsafe_allow_html=True)

# Load the saved models
diabetes_model = pickle.load(open('diabetes_rf_model.sav', 'rb'))
heart_disease_model = pickle.load(open('heart_disease_svm_model.sav', 'rb'))
parkinsons_model = pickle.load(open('parkinsons_model.sav', 'rb'))

# Sidebar for navigation
with st.sidebar:
    selected = option_menu('Multiple Disease Prediction System',
                          ['Diabetes Prediction', 'Heart Disease Prediction', 'Parkinsons Prediction'],
                          icons=['activity', 'heart', 'person'],
                          default_index=0)

# File uploader for medical report
uploaded_file = st.file_uploader("Upload your medical report", type=["pdf", "txt"])

if uploaded_file is not None:
    # Extract data from the uploaded file
    if uploaded_file.type == "application/pdf":
        reader = PyPDF2.PdfReader(uploaded_file)
        text = ""
        for page in reader.pages:
            text += page.extract_text()
        
        extracted_values = extract_values_from_text(text)
    elif uploaded_file.type == "text/plain":
        text = uploaded_file.getvalue().decode("utf-8")
        extracted_values = extract_values_from_text(text)
    
    # Pre-fill input fields with extracted values
else:
    extracted_values = {}

# Diabetes Prediction Page
if selected == 'Diabetes Prediction':
    # Page title
    st.title('Diabetes Prediction')
    
    # Getting the input data from the user
    col1, col2, col3 = st.columns(3)
    
    with col1:
        Pregnancies = st.number_input('Number of Pregnancies', min_value=0, value=int(extracted_values.get("Pregnancies") or 0))       
    with col2:
        Glucose = st.number_input('Glucose Level', min_value=0, value=int(extracted_values.get("Glucose") or 0))    
    with col3:
        BloodPressure = st.number_input('Blood Pressure value', min_value=0, value=int(extracted_values.get("BloodPressure") or 0))    
    with col1:
        SkinThickness = st.number_input('Skin Thickness value', min_value=0, value=int(extracted_values.get("SkinThickness") or 0))   
    with col2:
        Insulin = st.number_input('Insulin Level', min_value=0, value=int(extracted_values.get("Insulin") or 0))
    with col3:
        BMI = st.number_input('BMI value', min_value=0.0, value=float(extracted_values.get("BMI") or 0.0))
    with col1:
        DiabetesPedigreeFunction = st.number_input('Diabetes Pedigree Function value', min_value=0.0, value=float(extracted_values.get("DiabetesPedigreeFunction") or 0.0))
    with col2:
        Age = st.number_input('Age of the Person', min_value=0, value=int(extracted_values.get("Age") or 0))
    
    # Code for Prediction
    diab_diagnosis = ''
    
    # Creating a button for Prediction
    if st.button('Diabetes Test Result'):
        diab_prediction = diabetes_model.predict([[Pregnancies, Glucose, BloodPressure, SkinThickness, Insulin, BMI, DiabetesPedigreeFunction, Age]])
        
        if diab_prediction[0] == 1:
            diab_diagnosis = 'The person is diabetic'
        else:
            diab_diagnosis = 'The person is not diabetic'   
    st.success(diab_diagnosis)

# Heart Disease Prediction Page
if selected == 'Heart Disease Prediction':
    
    # Page title
    st.title('Heart Disease Prediction')
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        age = st.number_input('Age', min_value=0, value=int(extracted_values.get("AgeHeart") or 0))        
    with col2:
        sex = st.number_input('Sex', min_value=0, value=int(extracted_values.get("Sex") or 0))       
    with col3:
        cp = st.number_input('Chest Pain types', min_value=0, value=int(extracted_values.get("CP") or 0))        
    with col1:
        trestbps = st.number_input('Resting Blood Pressure', min_value=0, value=int(extracted_values.get("Trestbps") or 0))        
    with col2:
        chol = st.number_input('Serum Cholestrol in mg/dl', min_value=0, value=int(extracted_values.get("Chol") or 0))        
    with col3:
        fbs = st.number_input('Fasting Blood Sugar > 120 mg/dl', min_value=0, value=int(extracted_values.get("FBS") or 0))       
    with col1:
        restecg = st.number_input('Resting Electrocardiographic results', min_value=0, value=int(extracted_values.get("RestECG") or 0))        
    with col2:
        thalach = st.number_input('Maximum Heart Rate achieved', min_value=0, value=int(extracted_values.get("Thalach") or 0))        
    with col3:
        exang = st.number_input('Exercise Induced Angina', min_value=0, value=int(extracted_values.get("Exang") or 0))        
    with col1:
        oldpeak = st.number_input('ST depression induced by exercise', min_value=0.0, value=float(extracted_values.get("Oldpeak") or 0.0))        
    with col2:
        slope = st.number_input('Slope of the peak exercise ST segment', min_value=0, value=int(extracted_values.get("Slope") or 0))        
    with col3:
        ca = st.number_input('Major vessels colored by flourosopy', min_value=0, value=int(extracted_values.get("CA") or 0))        
    with col1:
        thal = st.number_input('thal: 0 = normal; 1 = fixed defect; 2 = reversible defect', min_value=0, value=int(extracted_values.get("Thal") or 0))
    
    # Code for Prediction
    heart_diagnosis = ''
    
    # Creating a button for Prediction
    if st.button('Heart Disease Test Result'):
        try:
            inputs = [
                float(age), float(sex), float(cp), float(trestbps), float(chol),
                float(fbs), float(restecg), float(thalach), float(exang),
                float(oldpeak), float(slope), float(ca), float(thal)
            ]
            heart_prediction = heart_disease_model.predict([inputs])
            if heart_prediction[0] == 1:
                heart_diagnosis = 'The person has heart disease'
            else:
                heart_diagnosis = 'The person does not have heart disease'
        except ValueError as e:
            heart_diagnosis = f'Error in input: {str(e)}'
    
    st.success(heart_diagnosis)

# Parkinson's Prediction Page
if selected == "Parkinsons Prediction":
    
    # Page title
    st.title('Parkinsons Prediction')
    
    # Getting the input data from the user
    col1, col2, col3 = st.columns(3)

    with col1:
        MDVPFo = st.number_input('MDVPFo (Hz)', value=float(extracted_values.get("MDVPFo") or 0.0))        
    with col2:
        MDVPFhi = st.number_input('MDVPFhi (Hz)', value=float(extracted_values.get("MDVPFhi") or 0.0))    
    with col3:
        MDVPFlo = st.number_input('MDVPFlo (Hz)', value=float(extracted_values.get("MDVPFlo") or 0.0))    
    with col1:
        Jitter_percent = st.number_input('MDVPJitter (%)', value=float(extracted_values.get("Jitter_percent") or 0.0))    
    with col2:
        Jitter_Abs = st.number_input('MDVPJitter (Abs)', value=float(extracted_values.get("Jitter_Abs") or 0.0))    
    with col3:
        RAP = st.number_input('MDVPRAP', value=float(extracted_values.get("RAP") or 0.0))    
    with col1:
        PPQ = st.number_input('MDVPPPQ', value=float(extracted_values.get("PPQ") or 0.0))    
    with col2:
        DDP = st.number_input('JitterDDP', value=float(extracted_values.get("DDP") or 0.0))    
    with col3:
        Shimmer = st.number_input('MDVPShimmer', value=float(extracted_values.get("Shimmer") or 0.0))    
    with col1:
        Shimmer_dB = st.number_input('MDVPShimmer (dB)', value=float(extracted_values.get("Shimmer_dB") or 0.0))    
    with col2:
        APQ3 = st.number_input('ShimmerAPQ3', value=float(extracted_values.get("APQ3") or 0.0))    
    with col3:
        APQ5 = st.number_input('ShimmerAPQ5', value=float(extracted_values.get("APQ5") or 0.0))    
    with col1:
        APQ = st.number_input('MDVPAPQ', value=float(extracted_values.get("APQ") or 0.0))    
    with col2:
        DDA = st.number_input('ShimmerDDA', value=float(extracted_values.get("DDA") or 0.0))    
    with col3:
        NHR = st.number_input('NHR', value=float(extracted_values.get("NHR") or 0.0))    
    with col1:
        HNR = st.number_input('HNR', value=float(extracted_values.get("HNR") or 0.0))    
    with col2:
        RPDE = st.number_input('RPDE', value=float(extracted_values.get("RPDE") or 0.0))    
    with col3:
        DFA = st.number_input('DFA', value=float(extracted_values.get("DFA") or 0.0))    
    with col1:
        spread1 = st.number_input('spread1', value=float(extracted_values.get("spread1") or 0.0))    
    with col2:
        spread2 = st.number_input('spread2', value=float(extracted_values.get("spread2") or 0.0))    
    with col3:
        D2 = st.number_input('D2', value=float(extracted_values.get("D2") or 0.0))    
    with col1:
        PPE = st.number_input('PPE', value=float(extracted_values.get("PPE") or 0.0))
        
    # Code for Prediction
    parkinsons_diagnosis = ''
    
    # Creating a button for Prediction    
    if st.button("Parkinson's Test Result"):
        try:
            # Convert inputs to appropriate numeric types
            inputs = [
                float(MDVPFo), float(MDVPFhi), float(MDVPFlo), float(Jitter_percent), 
                float(Jitter_Abs), float(RAP), float(PPQ), float(DDP), 
                float(Shimmer), float(Shimmer_dB), float(APQ3), float(APQ5), 
                float(APQ), float(DDA), float(NHR), float(HNR), float(RPDE), 
                float(DFA), float(spread1), float(spread2), float(D2), float(PPE)
            ]
            
            # Predict
            parkinsons_prediction = parkinsons_model.predict([inputs])
            
            # Display result
            if parkinsons_prediction[0] == 1:
                parkinsons_diagnosis = "The person has Parkinson's disease"
            else:
                parkinsons_diagnosis = "The person does not have Parkinson's disease"
        
        except ValueError:
            st.error('Please enter valid numeric values.')
    
    st.success(parkinsons_diagnosis)
