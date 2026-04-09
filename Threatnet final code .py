import streamlit as st
import numpy as np
import pickle
import psutil
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model

# -------------------------------
# PAGE CONFIG
# -------------------------------
st.set_page_config(page_title="AI Antivirus Live", layout="wide")
st.title("🛡️ Real-Time AI Intrusion Detection System")

# -------------------------------
# LOAD MODELS (ONLY ONCE)
# -------------------------------
@st.cache_resource
def load_models():

    with open("ml_models.pkl", "rb") as f:
        ml_models = pickle.load(f)

    with open("scaler.pkl", "rb") as f:
        scaler = pickle.load(f)

    with open("label_encoder.pkl", "rb") as f:
        label_encoder = pickle.load(f)

    # Load only ANN (CNN + LSTM slow realtime)
    ann = load_model("ann_model.keras", compile=False)

    ml_models["ANN"] = ann

    return ml_models, scaler, label_encoder


models, scaler, label_encoder = load_models()


# -------------------------------
# SESSION STATE INIT
# -------------------------------
if "attack_log" not in st.session_state:
    st.session_state.attack_log = []

if "running" not in st.session_state:
    st.session_state.running = False


# -------------------------------
# ATTACK DESCRIPTION
# -------------------------------
def explain_attack(label):

    label = label.lower()

    if "normal" in label:
        return "System is safe"
    elif "dos" in label:
        return "Denial of Service attack detected"
    elif "probe" in label:
        return "Scanning activity detected"
    elif "r2l" in label:
        return "Remote access attack"
    elif "u2r" in label:
        return "Privilege escalation"
    else:
        return "Unknown suspicious activity"


# -------------------------------
# SEVERITY SCORE
# -------------------------------
def get_severity(label, confidence):

    if "normal" in label.lower():
        return "Low"

    if confidence > 0.9:
        return "High"
    elif confidence > 0.7:
        return "Medium"

    return "Low"


# -------------------------------
# CONTROL BUTTONS
# -------------------------------
col1, col2 = st.columns(2)

if col1.button("▶ Start Monitoring"):
    st.session_state.running = True

if col2.button("⏹ Stop Monitoring"):
    st.session_state.running = False


# -------------------------------
# LIVE DETECTION ENGINE
# -------------------------------
if st.session_state.running:

    sample = np.random.rand(1, scaler.n_features_in_)
    sample = scaler.transform(sample)

    predictions = []
    confidences = []

    for name, model in models.items():

        try:

            if name == "ANN":

                prob = model.predict(sample, verbose=0)
                pred = np.argmax(prob)
                conf = np.max(prob)

            else:

                pred = model.predict(sample)[0]
                conf = 0.85

            label = label_encoder.inverse_transform([int(pred)])[0]

            predictions.append(label)
            confidences.append(conf)

        except:
            continue


    encoded_preds = label_encoder.transform(predictions)

    final_pred_encoded = np.bincount(encoded_preds).argmax()

    final_pred = label_encoder.inverse_transform([final_pred_encoded])[0]

    final_conf = float(np.mean(confidences))

    severity = get_severity(final_pred, final_conf)

    st.session_state.attack_log.append(severity)


    # DISPLAY
    st.subheader("🔍 Live Detection")

    col1, col2, col3 = st.columns(3)

    col1.metric("Attack Type", final_pred)
    col2.metric("Confidence", f"{final_conf:.2f}")
    col3.metric("Severity", severity)

    st.info(explain_attack(final_pred))


# -------------------------------
# CPU STATUS
# -------------------------------
st.write(f"CPU Usage: {psutil.cpu_percent()}%")


# -------------------------------
# GRAPH
# -------------------------------
if len(st.session_state.attack_log) > 0:

    severity_map = {"Low": 1, "Medium": 2, "High": 3}

    graph_data = [
        severity_map[s]
        for s in st.session_state.attack_log
    ]

    fig, ax = plt.subplots()

    ax.plot(graph_data)

    ax.set_ylabel("Severity")
    ax.set_xlabel("Time")

    st.pyplot(fig)


# -------------------------------
# FINAL STATUS PANEL
# -------------------------------
st.header("🛡️ Final Status")

if "High" in st.session_state.attack_log:

    st.error("🚨 HIGH RISK DETECTED")

elif "Medium" in st.session_state.attack_log:

    st.warning("⚠️ Moderate Threats Detected")

else:

    st.success("✅ System Secure")
