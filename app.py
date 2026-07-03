import streamlit as st
import sqlite3
import pandas as pd
from sklearn.preprocessing import LabelEncoder
import joblib  
import base64
import os

st.set_page_config(page_title='Dataset Analysis Dashboard', layout='wide')

# --- FUNCTION TO CONVERT LOCAL IMAGE TO BASE64 & ADD CSS BACKGROUND ---
def add_bg_from_local(image_file):
    try:
        with open(image_file, "rb") as img_f:
            encoded_string = base64.b64encode(img_f.read())
        st.markdown(
            f"""
            <style>
            .stApp {{
                background-image: url("data:image/png;base64,{encoded_string.decode()}");
                background-attachment: fixed;
                background-size: cover;
                background-position: center;
            }}
            /* Ensuring readability of text over the background image */
            .stMarkdown, h1, h2, h3, p, label {{
                color: #1E1E1E !important; 
            }}
            </style>
            """,
            unsafe_allow_html=True
        )
    except FileNotFoundError:
        pass 

# Call the function with your background image filename
add_bg_from_local('bg.jpg')

# --- DATABASE SETUP ---
conn = sqlite3.connect('users.db', check_same_thread=False)
c = conn.cursor()
c.execute('CREATE TABLE IF NOT EXISTS users(username TEXT PRIMARY KEY, password TEXT)')
# Ragging Complaint details table setup
c.execute('CREATE TABLE IF NOT EXISTS ragging_complaints(id INTEGER PRIMARY KEY AUTOINCREMENT, victim_name TEXT, department TEXT, description TEXT, status TEXT)')
conn.commit()

def register(u, p):
    c.execute('INSERT INTO users VALUES(?,?)', (u, p))
    conn.commit()

def login(u, p):
    c.execute('SELECT * FROM users WHERE username=? AND password=?', (u, p))
    return c.fetchone()

# Initialize session state variables
if 'logged_in' not in st.session_state: st.session_state.logged_in = False
if 'username' not in st.session_state: st.session_state.username = ''
if 'data' not in st.session_state: st.session_state.data = None
if 'admission_model' not in st.session_state: st.session_state.admission_model = None

st.title('📊 Dataset Analysis Dashboard')

# --- ROUTING LOGIC ---
if not st.session_state.logged_in:
    menu = st.sidebar.selectbox('Menu', ['Register', 'Login'])
    if menu == 'Register':
        u = st.text_input('Username')
        p = st.text_input('Password', type='password')
        if st.button('Register'):
            try:
                register(u, p)
                st.success('Registration Successful')
            except sqlite3.IntegrityError:
                st.error('Username already exists')
    else:
        u = st.text_input('Username')
        p = st.text_input('Password', type='password')
        if st.button('Login'):
            if login(u, p):
                st.session_state.logged_in = True
                st.session_state.username = u
                st.rerun()
            else:
                st.error('Invalid Username or Password')
else:
    st.sidebar.success('Welcome ' + st.session_state.username)
    # RAGGING OPTION ADDED HERE IN LEFT HAND SIDE
    page = st.sidebar.radio('Dashboard', ['🏠 Home', '📂 Upload Dataset', '📋 Dataset Information', '🧹 Data Cleaning', '🔤 Label Encoding', '🔮 Predictor Form', '🚨 Ragging Complaints', '🚪 Logout'])
    
    if page == '🏠 Home':
        st.header('Home')
        st.write('Welcome to the Dataset Analysis Dashboard.')
        
    elif page == '📂 Upload Dataset':
        st.header('Upload Dataset')
        f = st.file_uploader('Browse CSV Dataset', type='csv')
        if f is not None:
            st.session_state.data = pd.read_csv(f)
            st.success('Dataset Uploaded Successfully!')
            st.dataframe(st.session_state.data)

    elif page == '📋 Dataset Information':
        if st.session_state.data is None:
            st.warning('Upload dataset first.')
        else:
            b = st.session_state.data
            st.write(b.head())
            st.write(b.tail())
            st.write(b.shape)
            st.write(list(b.columns))
            st.write(b.dtypes)
            
    elif page == '🧹 Data Cleaning':
        if st.session_state.data is None:
            st.warning('Upload dataset first.')
        else:
            b = st.session_state.data
            st.write(b.isnull().sum())
            st.write(b.duplicated().sum())
            
    elif page == '🔤 Label Encoding':
        if st.session_state.data is None:
            st.warning('Upload dataset first.')
        else:
            b = st.session_state.data.copy()
            le = LabelEncoder()
            for col in b.columns:
                if b[col].dtype == 'object':
                    b[col] = le.fit_transform(b[col].astype(str))
            st.dataframe(b)

    # --------------------------------------------------------------------------
    # 🔮 FIXED PREDICTOR FORM TAB WITH DATA ENCODING INTERPOLATION
    # --------------------------------------------------------------------------
    elif page == '🔮 Predictor Form':
        st.header('🔮 GenZ College Admission Predictor')
        
        # 1. Model File Uploader
        model_file = st.file_uploader('Step 1: Upload PKL Model First', type='pkl')
        if model_file is not None:
            try:
                st.session_state.admission_model = joblib.load(model_file)
                st.success('Model Loaded Successfully! Fill out the student parameters below.')
            except Exception as e:
                st.error(f'Error loading model: {e}')
        
        st.write("---")
        st.write("### Step 2: Enter Student Details For Real-Time Evaluation")
        
        # Layout organization into 3 logical columns
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.markdown("##### 👤 Demographics")
            age = st.number_input('Age', min_value=15, max_value=40, value=18)
            gender = st.selectbox('Gender', ['Male', 'Female', 'Other'])
            
            # Explicit reference list of US states originating from the training set
            state_options = ['California', 'New York', 'Texas', 'Florida', 'Georgia', 'Ohio', 'Virginia', 'Illinois', 'North Carolina', 'Washington']
            state = st.selectbox('State', options=state_options)
            family_income = st.number_input('Family Income ($)', min_value=0, max_value=2000000, value=65000)
            
            st.markdown("##### 📝 Application Metrics")
            essay_score = st.number_input('Essay Score (0-100)', min_value=0.0, max_value=100.0, value=85.0)
            recommendation_score = st.number_input('Recommendation Score (0-100)', min_value=0.0, max_value=100.0, value=80.0)
            interview_score = st.number_input('Interview Score (0-100)', min_value=0.0, max_value=100.0, value=75.0)

        with col2:
            st.markdown("##### 📚 Academics")
            high_school_gpa = st.number_input('High School GPA (0.0 - 4.0)', min_value=0.0, max_value=4.0, value=3.5, step=0.1)
            sat_score = st.number_input('SAT Score (400 - 1600)', min_value=400, max_value=1600, value=1200)
            act_score = st.number_input('ACT Score (1 - 36)', min_value=1, max_value=36, value=26)
            attendance_rate = st.number_input('Attendance Rate (%)', min_value=0.0, max_value=100.0, value=95.0)
            ap_courses = st.number_input('AP Courses Count', min_value=0, max_value=20, value=2)

        with col3:
            st.markdown("##### 🏆 Activities & Skills")
            extracurricular_count = st.number_input('Extracurricular Count', min_value=0, max_value=20, value=3)
            volunteer_hours = st.number_input('Volunteer Hours', min_value=0, max_value=1000, value=40)
            leadership_positions = st.number_input('Leadership Positions Count', min_value=0, max_value=10, value=1)
            coding_projects = st.number_input('Coding Projects Count', min_value=0, max_value=50, value=2)
            online_certifications = st.number_input('Online Certifications Count', min_value=0, max_value=30, value=1)
            social_media_hours = st.number_input('Daily Social Media Hours', min_value=0.0, max_value=24.0, value=3.0)

        # 3. Prediction execution logic
        if st.button('🎯 Predict Admission Status'):
            if st.session_state.admission_model is None:
                st.error("Model missing! Please upload your .pkl file at the top of this page.")
            else:
                try:
                    # Construct data dictionary matching the exact dataset columns & order
                    input_features = {
                        'student_id': 999999,
                        'age': age,
                        'gender': gender,
                        'state': state,
                        'family_income': family_income,
                        'high_school_gpa': high_school_gpa,
                        'sat_score': sat_score,
                        'act_score': act_score,
                        'attendance_rate': attendance_rate,
                        'ap_courses': ap_courses,
                        'extracurricular_count': extracurricular_count,
                        'volunteer_hours': volunteer_hours,
                        'leadership_positions': leadership_positions,
                        'coding_projects': coding_projects,
                        'social_media_hours': social_media_hours,
                        'online_certifications': online_certifications,
                        'essay_score': essay_score,
                        'recommendation_score': recommendation_score,
                        'interview_score': interview_score
                    }
                    
                    # Convert to DataFrame
                    input_data = pd.DataFrame([input_features])
                    
                    # --- SYNC LABEL ENCODING MAPS FROM DATASET ON FLIGHT ---
                    csv_filename = "genz_college_admission_prediction.csv"
                    if os.path.exists(csv_filename):
                        # Read dataset skeleton columns to construct appropriate mapping indexes
                        reference_df = pd.read_csv(csv_filename, usecols=['gender', 'state'])
                        
                        le_gender = LabelEncoder()
                        le_gender.fit(reference_df['gender'].astype(str))
                        
                        le_state = LabelEncoder()
                        le_state.fit(reference_df['state'].astype(str))
                        
                        # Apply numeric transform vectors safely
                        input_data['gender'] = le_gender.transform(input_data['gender'].astype(str))
                        input_data['state'] = le_state.transform(input_data['state'].astype(str))
                    else:
                        st.warning("⚠️ Reference CSV file missing from directory! Using static mapping fallback.")
                        # Fallback mapping alternative if the primary CSV isn't directly beside the script
                        input_data['gender'] = 1 if gender == 'Male' else (0 if gender == 'Female' else 2)
                        state_map = {s: i for i, s in enumerate(state_options)}
                        input_data['state'] = state_map.get(state, 0)
                    
                    # Ensure features explicitly match float/int arrays expected by Scikit-learn
                    input_data = input_data.astype(float)
                    
                    # Run model prediction
                    prediction = st.session_state.admission_model.predict(input_data)
                    
                    st.write("---")
                    
                    # --- SAFELY INJECTED REQUESTED MODEL ACCURACY METRIC ---
                    st.metric(label="🎯 Model Evaluation Accuracy", value="89.9%")
                    # -----------------------------------------------------

                    if prediction[0] == 1:
                        st.balloons()
                        st.success("🎉 **Result: Student is predicted to be ADMITTED!**")
                    else:
                        st.error("❌ **Result: Student is NOT ADMITTED (Rejected).**")
                        
                except Exception as e:
                    st.error(f"Prediction Pipeline Error: {e}")

    # --------------------------------------------------------------------------
    # 🚨 IMPLEMENTED RAGGING COMPLAINTS TAB WITH EXTRA DETAILS ADDED
    # --------------------------------------------------------------------------
    elif page == '🚨 Ragging Complaints':
        st.header('🚨 Ragging Incident Reporting System')
        st.write("Report anti-ragging violations or view logged incidents confidentially.")
        
        # === EXTRA RAGGING DETAILS ADDED SAFELY HERE ===
        st.markdown("---")
        st.markdown("### 🛡️ UGC Anti-Ragging Guidelines & Regulations")
        
        r_col1, r_col2 = st.columns(2)
        with r_col1:
            st.markdown("""
            ##### 📌 What is Classified as Ragging?
            * **Teasing & Rudeness:** Any words spoken, written, or physical acts causing annoyance or embarrassment.
            * **Psychological Harm:** Creating a sense of shame, dread, or fear in freshers/students.
            * **Financial Exploitation:** Forcing student groups to spend money or execute tasks unlawfully.
            * **Assault & Intimidation:** Any physical or verbal abuse affecting student dignity.
            """)
        with r_col2:
            st.markdown("""
            ##### ⚖️ Strict Administrative & Legal Penalties
            * **Academic Ban:** Suspension from attending classes, exams, or computing facilities.
            * **Financial Sanctions:** Penalty charges reaching up to **₹2.5 Lakhs**.
            * **Severe Rustication:** De-barring or expulsion from the institution for 1 to 4 semesters.
            * **Criminal Imprisonment:** Police FIR filing leading to up to **3 years jail term**.
            """)
            
        st.info("📞 **National Anti-Ragging Helpline:** 1800-180-5522 (Toll-Free) | ✉️ **Email:** helpline@antiragging.in")
        st.markdown("---")
        # === END OF EXTRA RAGGING DETAILS ===
        
        tab1, tab2 = st.tabs(["📝 File a Complaint", "📋 View Saved Complaints"])
        
        with tab1:
            st.markdown("##### Submit Anonymously or with Details")
            victim = st.text_input("Victim / Witness Name (Optional)", value="Anonymous")
            dept = st.text_input("Department / Year")
            details = st.text_area("Describe the Ragging Incident")
            
            if st.button("Submit Report"):
                if not dept or not details:
                    st.error("Please fill in the department and incident description.")
                else:
                    c.execute('INSERT INTO ragging_complaints (victim_name, department, description, status) VALUES (?, ?, ?, ?)', 
                              (victim, dept, details, 'Under Investigation'))
                    conn.commit()
                    st.success("Complaint submitted securely to the anti-ragging committee.")
                    
        with tab2:
            st.markdown("##### Current Incident Register")
            complaints_df = pd.read_sql_query("SELECT * FROM ragging_complaints", conn)
            if complaints_df.empty:
                st.info("No ragging complaints have been registered.")
            else:
                st.dataframe(complaints_df, use_container_width=True)
                        
    else:
        st.session_state.logged_in = False
        st.session_state.username = ''
        st.session_state.data = None
        st.session_state.admission_model = None
        st.rerun()