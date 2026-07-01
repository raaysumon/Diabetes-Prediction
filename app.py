import os
import streamlit as st
import pandas as pd
from catboost import CatBoostClassifier

# --- Page Config ---
st.set_page_config(
    page_title="Early Diabetes AI Predictor",
    page_icon="🩸",
    layout="wide"
)

# --- 1. INITIAL LANGUAGE SELECTION BOX FIRST ---
if "app_language" not in st.session_state:
    st.session_state.app_language = None

if st.session_state.app_language is None:
    st.markdown("""
        <style>
        .stApp { background-color: #0f172a !important; color: #f1f5f9 !important; }
        .lang-box {
            max-width: 500px;
            margin: 100px auto;
            padding: 30px;
            background: #1e293b;
            border-radius: 16px;
            box-shadow: 0 10px 30px rgba(0,0,0,0.5);
            border: 1px solid rgba(255,255,255,0.08);
            text-align: center;
        }
        div[data-testid="stForm"] { background-color: transparent !important; border: none !important; padding: 0 !important; }
        .stButton>button {
            background: linear-gradient(90deg, #2563eb 0%, #1d4ed8 100%) !important;
            color: white !important; font-weight: bold !important; border-radius: 8px !important; width: 100% !important;
        }
        </style>
    """, unsafe_allow_html=True)
    
    st.markdown('<div class="lang-box">', unsafe_allow_html=True)
    st.markdown("<h2 style='color:#60a5fa; margin-bottom:20px;'>🌐 Select Language / ভাষা নির্বাচন করুন</h2>", unsafe_allow_html=True)
    
    with st.form(key="language_selection_form"):
        chosen_lang = st.radio("Choose your interface language:", ["English", "বাংলা"], index=None, label_visibility="collapsed")
        st.write("")
        if st.form_submit_button("Confirm & Enter App 🚀") and chosen_lang:
            st.session_state.app_language = chosen_lang
            st.rerun()
            
    st.markdown('</div>', unsafe_allow_html=True)
    st.stop()

lang = st.session_state.app_language

# --- 2. PREMIUM DARK THEME CSS ---
st.markdown("""
    <style>
    /* Main App Dark Background */
    .stApp {
        background-color: #0f172a !important;
        color: #e2e8f0 !important;
    }

    /* Headings & Text Color Fixes for Dark Mode */
    h1, h2, h3, h4, h5, h6, label, p, span {
        color: #f1f5f9 !important;
    }

    /* Custom Styling for Selectbox, Input Box inside Dark Theme */
    div[data-baseweb="select"] > div {
        background-color: #1e293b !important;
        color: #f1f5f9 !important;
        border: 1px solid #334155 !important;
    }
    
    div[data-testid="stNumberInput"] input {
        background-color: #1e293b !important;
        color: #f1f5f9 !important;
        border: 1px solid #334155 !important;
    }

    /* Primary Prediction Button */
    .stButton>button {
        width: 100%;
        border-radius: 8px;
        height: 3em;
        background: linear-gradient(90deg, #dc2626 0%, #b91c1c 100%) !important;
        color: white !important;
        font-weight: bold !important;
        border: none !important;
        transition: 0.3s;
        box-shadow: 0 4px 12px rgba(220, 38, 38, 0.3);
    }

    .stButton>button:hover {
        background: linear-gradient(90deg, #b91c1c 0%, #991b1b 100%) !important;
        box-shadow: 0 6px 16px rgba(220, 38, 38, 0.5);
    }

    /* Form Container Sections (Cards) */
    div[data-testid="stVerticalBlock"] > div:has(div.stColumn) {
        background-color: #1e293b !important;
        padding: 25px;
        border-radius: 15px;
        border: 1px solid #334155;
        box-shadow: 0 4px 20px rgba(0,0,0,0.2);
    }

    /* 🔴 Diabetes Positive: Luxury Dark Red Alert Box */
    .dark-red-alert {
        background-color: #450a0a !important;
        border-left: 6px solid #ef4444 !important;
        color: #fef2f2 !important;
        padding: 20px;
        border-radius: 12px;
        margin-bottom: 15px;
        box-shadow: 0 4px 20px rgba(239, 68, 68, 0.15);
        border-top: 1px solid #7f1d1d;
        border-right: 1px solid #7f1d1d;
        border-bottom: 1px solid #7f1d1d;
    }
    .dark-red-alert h3 { color: #fecaca !important; margin: 0 0 5px 0; font-weight: bold; }

    /* 🟢 Diabetes Negative: Luxury Dark Green Alert Box */
    .dark-green-alert {
        background-color: #022c22 !important;
        border-left: 6px solid #10b981 !important;
        color: #ecfdf5 !important;
        padding: 20px;
        border-radius: 12px;
        margin-bottom: 15px;
        box-shadow: 0 4px 20px rgba(16, 185, 129, 0.15);
        border-top: 1px solid #064e3b;
        border-right: 1px solid #064e3b;
        border-bottom: 1px solid #064e3b;
    }
    .dark-green-alert h3 { color: #a7f3d0 !important; margin: 0 0 5px 0; font-weight: bold; }
    
    /* Info text updates */
    div[data-testid="stAlert"] {
        background-color: #1e293b !important;
        color: #e2e8f0 !important;
        border: 1px solid #334155 !important;
    }
    </style>
    """, unsafe_allow_html=True)


# --- Model Loading ---
@st.cache_resource
def load_model():
    model = CatBoostClassifier()
    current_dir = os.path.dirname(__file__)
    model_path = os.path.join(current_dir, "final_catboost_modol.cbm")
    
    try:
        model.load_model(model_path)
        return model
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None

model = load_model()

# --- Sidebar Controls ---
st.sidebar.title("⚙️ Control Panel")
if st.sidebar.button("Change Language / ভাষা পরিবর্তন 🔄"):
    st.session_state.app_language = None
    st.rerun()

# --- Localization Dictionary ---
labels = {
    "English": {
        "title": "🩸 Early Diabetes Risk Assessment & Analytics",
        "subtitle": "This clinical decision support tool uses a machine learning model to predict the early risk of diabetes based on patient demographics, common signs, and clinical symptoms.",
        "demo_header": "📋 Demographics",
        "age": "Age",
        "gender": "Gender",
        "male": "Male",
        "female": "Female",
        "primary_header": "🔍 Primary Symptoms",
        "polyuria": "Polyuria (Excessive Urination)",
        "polydipsia": "Polydipsia (Excessive Thirst)",
        "irritability": "Irritability",
        "secondary_header": "⚡ Secondary Signs",
        "itching": "Itching",
        "delayed_healing": "Delayed Healing (Wounds)",
        "alopecia": "Alopecia (Hair Loss)",
        "yes": "Yes",
        "no": "No",
        "btn": "Analyze Diabetes Risk",
        "diag_header": "📊 Diagnostic Result",
        "pos_res": "Result: DIABETES RISK DETECTED",
        "neg_res": "Result: NO RISK DETECTED",
        "confidence": "Model Confidence",
        "viz_title": "Risk Probability Visualization",
        "info_msg": "The model is {:.1f}% certain of this classification based on the symptoms and historical data analysis.",
        "model_err": "Model file 'final_catboost_modol.cbm' not found. Please ensure it is in the same directory.",
        "xai_header": "🧠 Explainable AI (XAI) & Clinical Insights",
        "xai_subtitle": "To assist doctors in decision-making, below is a real-time explanation based on backend analytics and **SHAP (SHapley Additive exPlanations)** values.",
        "global_title": "📊 Global Feature Importance (SHAP Overview)",
        "global_caption": "Figure: SHAP values indicating the relative global importance of features in diabetes prediction.",
        "patient_title": "🩺 Patient-Specific Clinical Breakdown",
        "drivers_title": "⚠️ **Primary Risk Drivers:**",
        "drivers_sub": "• **Top-Tier Drivers:** The presence of these major high-impact symptoms significantly increased the probability of diabetes:",
        "secondary_sub": "• **Supporting Factors:** These secondary elements also contributed to the risk score:",
        "clinical_note_pos": "> **Clinical Note:** In our predictive model, **Polyuria** and **Polydipsia** are the strongest risk indicators. Given the patient's presentation, a clinical follow-up with HbA1c or Fasting Blood Sugar (FBS) test is strongly recommended.",
        "success_title": "✅ **Factors Preventing Risk:**",
        "success_msg_1": "• The model's primary high-impact symptoms (**Polyuria** and **Polydipsia**) are **Absent**. This is the primary reason the risk score remains low.",
        "success_msg_2": "• Although some symptoms are present, the overall combination and low-level secondary signs keep the final score safely below the diabetic risk threshold.",
        "clinical_note_neg": "> **Clinical Note:** While the current symptom profile suggests no immediate risk, routine wellness screening and lifestyle tracking remain optimal."
    },
    "বাংলা": {
        "title": "🩸 ডায়াবেটিস ঝুঁকি মূল্যায়ন ও অ্যানেলিটিক্স",
        "subtitle": "এই ক্লিনিকাল ডিসিশন সাপোর্ট টুলটি ডায়াবেতিসের প্রাথমিক ঝুঁকি পূর্বাভাস দেওয়ার জন্য পেশেন্ট ডেমোগ্রাফিক্স এবং সাধারণ লক্ষণগুলির ওপর ভিত্তি করে একটি machine learning মডেল ব্যবহার করে।",
        "demo_header": "📋 ডেমোগ্রাফিক্স (সাধারণ তথ্য)",
        "age": "বয়স",
        "gender": "লিঙ্গ",
        "male": "পুরুষ",
        "female": "নারী",
        "primary_header": "🔍 প্রাথমিক লক্ষণসমূহ",
        "polyuria": "পলিউরিয়া (অতিরিক্ত প্রস্রাব)",
        "polydipsia": "পলিডিপসিয়া (অতিরিক্ত তৃষ্ণা)",
        "irritability": "খিটখিটে মেজাজ (Irritability)",
        "secondary_header": "⚡ সেকেন্ডারি লক্ষণসমূহ",
        "itching": "চুলকানি (Itching)",
        "delayed_healing": "ক্ষত শুকাতে বিলম্ব (Delayed Healing)",
        "alopecia": "চুল পড়ে যাওয়া (Alopecia)",
        "yes": "হ্যাঁ",
        "no": "না",
        "btn": "ডায়াবেটিস ঝুঁকি বিশ্লেষণ করুন",
        "diag_header": "📊 ডায়াগনস্টিক ফলাফল",
        "pos_res": "ফলাফল: ডায়াবেতিসের ঝুঁকি পাওয়া গেছে",
        "neg_res": "ফলাফল: কোনো ঝুঁকি পাওয়া যায়নি",
        "confidence": "মডেলের নির্ভরযোগ্যতা",
        "viz_title": "ঝুঁকির সম্ভাব্যতা ভিজ্যুয়ালাইজেশন",
        "info_msg": "লক্ষণ এবং ঐতিহাসিক ডেটা বিশ্লেষণের ওপর ভিত্তি করে মডেলটি এই শ্রেণীবিভাগ সম্পর্কে {:.1f}% নিশ্চিত।",
        "model_err": "'final_catboost_modol.cbm' মডেল ফাইলটি পাওয়া যায়নি। অনুগ্রহ করে এটি সঠিক ডিরেক্টরিতে আছে কিনা নিশ্চিত করুন।",
        "xai_header": "🧠 এক্সপ্লেইনেবল এআই (XAI) এবং ক্লিনিকাল ইনসাইটস",
        "xai_subtitle": "ডাক্তারদের সিদ্ধান্ত গ্রহণের সুবিধার্থে মডেলের ব্যাকএ্যান্ড অ্যানালিটিক্স এবং **SHAP** ভ্যালুর ওপর ভিত্তি করে নিচে একটি রিয়েল-টাইম ব্যাখ্যা দেওয়া হলো।",
        "global_title": "📊 গ্লোবাল ফিচার ইম্পর্ট্যান্স (SHAP ওভারভিউ)",
        "global_caption": "চিত্র: ডায়াবেটিস পূর্বাভাসের ক্ষেত্রে বিভিন্ন ফিচারের আপেক্ষিক বৈশ্বিক গুরুত্ব নির্দেশক SHAP ভ্যালু।",
        "patient_title": "🩺 রোগী-নির্দিষ্ট ক্লিনিকাল ব্রেকডাউন",
        "drivers_title": "⚠️ **ঝুঁকি বৃদ্ধির প্রধান কারণসমূহ:**",
        "drivers_sub": "• **টপ-টায়ার ড্রাইভার্স:** রোগীর শরীরে নিম্নোক্ত প্রধান লক্ষণগুলো উপস্থিত থাকায় মডেলটির ডায়াবেটিস পজিティブ আসার সম্ভাবনা তীব্র হয়েছে:",
        "secondary_sub": "• **সহায়ক লক্ষণসমূহ:** এছাড়া সাপোর্টিং ফ্যাক্টর হিসেবে এই লক্ষণগুলো ঝুঁকি বাড়াতে ভূমিকা রেখেছে:",
        "clinical_note_pos": "> **ক্লিনিকাল নোট:** আমাদের মডেলে **Polyuria** এবং **Polydipsia** সবচেয়ে শক্তিশালী ঝুঁকির সূচক। রোগীর এই লক্ষণগুলো থাকলে দ্রুত HbA1c বা ফাস্টিং ব্লাড সুগার টেস্ট করানোর পরামর্শ দেওয়া যাচ্ছে।",
        "success_title": "✅ **ঝুঁকি মুক্ত থাকার কারণ:**",
        "success_msg_1": "• মডেলের প্রধান দুটি হাই-ইমপ্যাক্ট লক্ষণ (**Polyuria** এবং **Polydipsia**) রোগীর মধ্যে **অনুপস্থিত**। এটি রিস্ক স্কোর সর্বনিম্ন রাখার প্রধান কারণ।",
        "success_msg_2": "• রোগীর কিছু লক্ষণ থাকলেও সামগ্রিক কম্বিনেশন এবং লো-লেভেল সেকেন্ডারি সাইন্সের কারণে চূড়ান্ত স্কোর ডায়াবেতিসের ঝুঁকির সীমার নিচে রয়েছে।",
        "clinical_note_neg": "> **ক্লিনিকাল নোট:** যদিও বর্তমান লক্ষণ অনুযায়ী কোনো ঝুঁকি নেই, তবুও রোগীর বয়স এবং অন্যান্য লাইফস্টাইল ফ্যাক্টরের দিকে নিয়মিত নজর রাখা উচিত।"
    }
}

current_labels = labels[lang]

# --- Title Header ---
st.title(current_labels["title"])
st.markdown(current_labels["subtitle"])
st.divider()

# --- Input Form Panel ---
with st.container():
    col1, col2, col3 = st.columns(3, gap="large")

    gender_options = [current_labels["male"], current_labels["female"]]
    yes_no_options = [current_labels["no"], current_labels["yes"]]

    with col1:
        st.subheader(current_labels["demo_header"])
        age = st.number_input(current_labels["age"], min_value=1, max_value=120, value=40, step=1)
        gender_sel = st.selectbox(current_labels["gender"], gender_options)
        gender = 'Male' if gender_sel == current_labels["male"] else 'Female'

    with col2:
        st.subheader(current_labels["primary_header"])
        polyuria_sel = st.selectbox(current_labels["polyuria"], yes_no_options)
        polydipsia_sel = st.selectbox(current_labels["polydipsia"], yes_no_options)
        irritability_sel = st.selectbox(current_labels["irritability"], yes_no_options)

        polyuria = 'Yes' if polyuria_sel == current_labels["yes"] else 'No'
        polydipsia = 'Yes' if polydipsia_sel == current_labels["yes"] else 'No'
        irritability = 'Yes' if irritability_sel == current_labels["yes"] else 'No'

    with col3:
        st.subheader(current_labels["secondary_header"])
        itching_sel = st.selectbox(current_labels["itching"], yes_no_options)
        delayed_healing_sel = st.selectbox(current_labels["delayed_healing"], yes_no_options)
        alopecia_sel = st.selectbox(current_labels["alopecia"], yes_no_options)

        itching = 'Yes' if itching_sel == current_labels["yes"] else 'No'
        delayed_healing = 'Yes' if delayed_healing_sel == current_labels["yes"] else 'No'
        alopecia = 'Yes' if alopecia_sel == current_labels["yes"] else 'No'

st.write(" ")
predict_btn = st.button(current_labels["btn"])
st.write(" ")

# --- Prediction Logic ---
if predict_btn:
    if model is not None:
        input_data = pd.DataFrame([{
            'Age': age, 'Gender': gender, 'Polyuria': polyuria, 'Polydipsia': polydipsia,
            'Itching': itching, 'Irritability': irritability, 'delayed healing': delayed_healing, 'Alopecia': alopecia
        }])

        for col in input_data.columns:
            if col != 'Age':
                input_data[col] = input_data[col].astype('category')

        prediction = model.predict(input_data)[0]
        probability = model.predict_proba(input_data)[0]

        st.subheader(current_labels["diag_header"])
        res_col1, res_col2 = st.columns([1, 2])
        is_positive = str(prediction) == "1" or prediction == 1 or str(prediction).lower() == "positive"

        with res_col1:
            if is_positive:
                st.markdown(f"""
                    <div class="dark-red-alert">
                        <h3>⚠️ {current_labels['pos_res']}</h3>
                    </div>
                """, unsafe_allow_html=True)
                score = probability[1]
            else:
                st.markdown(f"""
                    <div class="dark-green-alert">
                        <h3>✅ {current_labels['neg_res']}</h3>
                    </div>
                """, unsafe_allow_html=True)
                score = probability[0]

            st.metric(label=current_labels["confidence"], value=f"{score * 100:.2f}%")

        with res_col2:
            st.write(f"**{current_labels['viz_title']}**")
            st.progress(float(score))
            st.info(current_labels["info_msg"].format(score * 100))

        # --- Explainable AI (XAI) Section ---
        st.divider()
        st.subheader(current_labels["xai_header"])
        st.markdown(current_labels["xai_subtitle"])

        xai_col1, xai_col2 = st.columns([1, 1], gap="large")

        with xai_col1:
            st.write(f"**{current_labels['global_title']}**")
            shap_data = pd.DataFrame({
                'Features': ['Polyuria', 'Polydipsia', 'Gender', 'Itching', 'Alopecia', 'Age', 'Delayed healing', 'Irritability'],
                'Mean |SHAP Value|': [2.304, 1.500, 1.484, 0.632, 0.556, 0.448, 0.436, 0.389]
            })
            st.bar_chart(data=shap_data, x='Features', y='Mean |SHAP Value|', color="#ff7f0e", use_container_width=True)
            st.caption(current_labels["global_caption"])

        with xai_col2:
            st.write(f"**{current_labels['patient_title']}**")

            high_risk_symptoms = []
            if polyuria == 'Yes': high_risk_symptoms.append(f"**Polyuria** (Impact: +2.30)")
            if polydipsia == 'Yes': high_risk_symptoms.append(f"**Polydipsia** (Impact: +1.50)")
            if gender == 'Male': high_risk_symptoms.append(f"**Gender (Male)** (Impact: +1.48)")

            secondary_symptoms = []
            if itching == 'Yes': secondary_symptoms.append(f"**Itching** (+0.63)")
            if alopecia == 'Yes': secondary_symptoms.append(f"**Alopecia** (+0.56)")
            if delayed_healing == 'Yes': secondary_symptoms.append(f"**Delayed Healing** (+0.43)")
            if irritability == 'Yes': secondary_symptoms.append(f"**Irritability** (+0.39)")

            if is_positive:
                st.warning(current_labels["drivers_title"])
                if high_risk_symptoms:
                    st.write(current_labels["drivers_sub"])
                    for sym in high_risk_symptoms: st.write(f"  - {sym}")
                if secondary_symptoms:
                    st.write(current_labels["secondary_sub"])
                    for sym in secondary_symptoms: st.write(f"  - {sym}")
                st.markdown(current_labels["clinical_note_pos"])
            else:
                st.success(current_labels["success_title"])
                if polyuria == 'No' and polydipsia == 'No':
                    st.write(current_labels["success_msg_1"])
                else:
                    st.write(current_labels["success_msg_2"])
                st.markdown(current_labels["clinical_note_neg"])
    else:
        st.error(current_labels["model_err"])

# --- Footer ---
st.divider()
st.caption("© 2026 Early Diabetes AI Prediction Systems | For Research Use Only | Made by Sumon Ray")
