import os
import streamlit as st
import pandas as pd
from catboost import CatBoostClassifier

# --- Page Config ---
st.set_page_config(
    page_title="DECat-AI Predictor",
    page_icon="🩸",
    layout="wide"
)

# --- Custom CSS for a Professional Medical Look ---
st.markdown("""
    <style>
    /* Main background color */
    .stApp {
        background-color: #f8f9fa;
    }

    /* Styling the prediction button */
    .stButton>button {
        width: 100%;
        border-radius: 8px;
        height: 3em;
        background-color: #dc3545; /* Crimson Red for Diabetes theme */
        color: white;
        font-weight: bold;
        border: none;
        transition: 0.3s;
    }

    .stButton>button:hover {
        background-color: #bd2130;
        border: none;
        color: white;
    }

    /* Card-like styling for sections */
    div[data-testid="stVerticalBlock"] > div:has(div.stColumn) {
        background-color: white;
        padding: 20px;
        border-radius: 15px;
        box-shadow: 0 2px 10px rgba(0,0,0,0.05);
    }
    
    /* Chat Container styling */
    .chat-box {
        background-color: #ffffff;
        padding: 15px;
        border-radius: 10px;
        margin-bottom: 10px;
        box-shadow: 0 1px 3px rgba(0,0,0,0.1);
    }
    </style>
    """, unsafe_allow_html=True)


# --- Model Loading (Updated for Streamlit Server) ---
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

# --- Language Selection Sidebar ---
st.sidebar.title("🌐 Language / ভাষা")
lang = st.sidebar.radio("Select Interface Language", ["English", "বাংলা"], index=0)

# --- Localization Dictionary ---
labels = {
    "English": {
        "title": "🩸 DECat-AI: Early Diabetes Risk Assessment & Chat",
        "subtitle": "This clinical decision support tool uses a CatBoost machine learning model to predict the early risk of diabetes based on patient demographics and clinical symptoms.",
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
        "pos_res": "### Result: DIABETES RISK DETECTED",
        "neg_res": "### Result: NO RISK DETECTED",
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
        "success_msg_2": "• Although some symptoms are present, the overall combination keeps the final score safely below the diabetic risk threshold.",
        "clinical_note_neg": "> **Clinical Note:** While the current symptom profile suggests no immediate risk, routine wellness screening and lifestyle tracking remain optimal.",
        "chat_header": "💬 DECat-AI Medical Assistant",
        "chat_welcome": "Hello! I am DECat-AI, your digital screening companion. Before we begin, may I know your name and how you are feeling today?",
        "chat_placeholder": "Type your name or reply here...",
        "disclaimer": "⚠️ **Disclaimer:** This AI screening tool is for research and educational purposes only. It does not replace professional clinical diagnosis."
    },
    "বাংলা": {
        "title": "🩸 DECat-AI: ডায়াবেটিস ঝুঁকি মূল্যায়ন ও চ্যাট",
        "subtitle": "এই ক্লিনিকাল ডিসিশন সাপোর্ট টুলটি ডায়াবেতিসের প্রাথমিক ঝুঁকি পূর্বাভাস দেওয়ার জন্য পেশেন্ট ডেমোগ্রাফিক্স এবং সাধারণ লক্ষণগুলির ওপর ভিত্তি করে একটি CatBoost ML মডেল ব্যবহার করে।",
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
        "pos_res": "### ফলাফল: ডায়াবেতিসের ঝুঁকি পাওয়া গেছে",
        "neg_res": "### ফলাফল: কোনো ঝুঁকি পাওয়া যায়নি",
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
        "success_msg_2": "• রোগীর কিছু লক্ষণ থাকলেও সামগ্রিক কম্বিনেশনের কারণে চূড়ান্ত স্কোর ডায়াবেতিসের ঝুঁকির সীমার নিচে রয়েছে।",
        "clinical_note_neg": "> **ক্লিনিকাল নোট:** যদিও বর্তমান লক্ষণ অনুযায়ী কোনো ঝুঁকি নেই, তবুও রোগীর বয়স এবং অন্যান্য লাইফস্টাইল ফ্যাক্টরের দিকে নিয়মিত নজর রাখা উচিত।",
        "chat_header": "💬 DECat-AI মেডিকেল অ্যাসিস্ট্যান্ট",
        "chat_welcome": "হ্যালো! আমি DECat-AI, আপনার ডিজিটাল স্ক্রিনিং পার্টনার। আমাদের স্ক্রিনিং শুরু করার আগে আপনার নাম এবং আপনি আজ কেমন বোধ করছেন তা জানতে পারি কি?",
        "chat_placeholder": "আপনার নাম অথবা উত্তর এখানে লিখুন...",
        "disclaimer": "⚠️ **সতর্কবার্তা:** এই এআই স্ক্রিনিং টুলটি শুধুমাত্র গবেষণা ও শিক্ষার উদ্দেশ্যে তৈরি। এটি কোনো পেশাদার চিকিৎসকের বিকল্প নয়।"
    }
}

current_labels = labels[lang]

# --- Session State Initialization for Conversational Chat ---
if "chat_history" not in st.session_state:
    st.session_state.chat_history = [{"role": "assistant", "content": current_labels["chat_welcome"]}]
if "patient_name" not in st.session_state:
    st.session_state.patient_name = ""

# --- Layout Structure: Left Side (Form & Prediction) | Right Side (DECat-AI Chat) ---
main_col, chat_col = st.columns([1.8, 1.2], gap="large")

with main_col:
    st.title(current_labels["title"])
    st.markdown(current_labels["subtitle"])
    st.divider()

    # --- Input Form ---
    with st.container():
        col1, col2, col3 = st.columns(3, gap="small")

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

    # --- Prediction & Analytics Logic ---
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
            res_col1, res_col2 = st.columns([1, 1.5])
            is_positive = str(prediction) == "1" or prediction == 1 or str(prediction).lower() == "positive"

            with res_col1:
                if is_positive:
                    st.error(current_labels["pos_res"])
                    score = probability[1]
                else:
                    st.success(current_labels["neg_res"])
                    score = probability[0]
                st.metric(label=current_labels["confidence"], value=f"{score * 100:.2f}%")

            with res_col2:
                st.write(f"**{current_labels['viz_title']}**")
                st.progress(float(score))
                st.info(current_labels["info_msg"].format(score * 100))

            # --- Explainable AI Section ---
            st.divider()
            st.subheader(current_labels["xai_header"])
            
            xai_col1, xai_col2 = st.columns([1, 1], gap="medium")
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
                if is_positive:
                    st.warning(current_labels["drivers_title"])
                    st.markdown(current_labels["clinical_note_pos"])
                else:
                    st.success(current_labels["success_title"])
                    st.markdown(current_labels["clinical_note_neg"])
        else:
            st.error(current_labels["model_err"])

# --- Right Column: Synchronized DECat-AI Chat Doctor Interface ---
with chat_col:
    st.subheader(current_labels["chat_header"])
    
    # Custom conversational layout window
    chat_container = st.container(height=450, border=True)
    with chat_container:
        for msg in st.session_state.chat_history:
            with st.chat_message(msg["role"]):
                st.write(msg["content"])

    # Conversational Chat Input
    if user_chat_input := st.chat_input(placeholder=current_labels["chat_placeholder"]):
        # Append User Response
        st.session_state.chat_history.append({"role": "user", "content": user_chat_input})
        
        # Simple State Handling to greet by Name dynamically
        if not st.session_state.patient_name:
            st.session_state.patient_name = user_chat_input
            response = f"Thank you, {st.session_state.patient_name}. Let's keep an eye on your symptoms. Please review or adjust your profile parameters on the left pane and trigger the risk analysis whenever you are ready!" if lang == "English" else f"ধন্যবাদ, {st.session_state.patient_name}। আমাদের স্ক্রিনিং শুরু করার জন্য আপনার লক্ষণগুলো বাম পাশের প্যানেলে ইনপুট দিয়ে নিচের অ্যানালাইসিস বাটনটি প্রেস করুন।"
        else:
            response = f"Understood. If you've updated any criteria like Polyuria or Age, click 'Analyze Diabetes Risk' to compute precise clinical outputs." if lang == "English" else "বুঝতে পেরেছি। নতুন কোনো লক্ষণ পরিবর্তন করে থাকলে বাম পাশের বাটনে ক্লিক করে পুনরায় মূল্যায়ন করতে পারেন।"

        st.session_state.chat_history.append({"role": "assistant", "content": response})
        st.rerun()

# --- Footer ---
st.divider()
st.caption(f"© 2026 Early Diabetes AI Prediction Systems | {current_labels['disclaimer']} | Made by Sumon Ray")
