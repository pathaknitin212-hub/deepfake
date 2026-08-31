import streamlit as st
import torch
import torch.nn as nn
import torchvision.models as models
from torchvision import transforms
from PIL import Image
import json
import matplotlib.pyplot as plt

# ---------------- PAGE CONFIG ----------------
st.set_page_config(page_title="DEEPFAKE", page_icon="🕵️", layout="wide")

# ---------------- CUSTOM CSS (modern look) ----------------
st.markdown("""
<style>
    .stApp {
        background-color: #0e1117;
    }
    .main-title {
        font-size: 3rem;
        font-weight: 800;
        background: linear-gradient(90deg, #ff4b4b, #ff9d4b);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 0px;
        letter-spacing: 2px;
    }
    .subtitle {
        color: #888;
        font-size: 1rem;
        margin-bottom: 2rem;
    }
    .result-card {
        padding: 1.5rem;
        border-radius: 16px;
        text-align: center;
        margin-top: 1rem;
    }
    .fake-card {
        background: linear-gradient(135deg, #2b0f0f, #3a1414);
        border: 1px solid #ff4b4b;
    }
    .real-card {
        background: linear-gradient(135deg, #0f2b17, #14351b);
        border: 1px solid #4bff88;
    }
    .metric-box {
        background-color: #161b22;
        padding: 1rem;
        border-radius: 12px;
        border: 1px solid #262d3a;
        text-align: center;
    }
    section[data-testid="stSidebar"] {
        background-color: #10141c;
    }
    div[data-testid="stFileUploader"] {
        border: 2px dashed #333;
        border-radius: 12px;
        padding: 1rem;
    }
</style>
""", unsafe_allow_html=True)

# ---------------- LOAD MODEL ----------------
@st.cache_resource
def load_model():
    model = models.resnet18(weights=None)
    model.fc = nn.Linear(model.fc.in_features, 1)
    model.load_state_dict(torch.load("deepfake_model.pth", map_location="cpu"))
    model.eval()
    return model

@st.cache_data
def load_history():
    try:
        with open("training_history.json") as f:
            return json.load(f)
    except FileNotFoundError:
        return None

model = load_model()
history = load_history()

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                          std=[0.229, 0.224, 0.225])
])

# ---------------- SIDEBAR ----------------
with st.sidebar:
    st.markdown("## 🕵️ DEEPFAKE")
    st.caption("AI-generated face detector")
    st.divider()

    st.markdown("### ℹ️ About")
    st.markdown("""
    **DEEPFAKE** is a small scale project  and please do not expect any accuracy and real world needed output from this model. **.

    **Approach:** Transfer Learning
    **Backbone:** ResNet18 (pretrained on ImageNet)
    **Trainable parameters:** 513 / 11,177,025
    **Dataset:** ~924 real & fake face images
    (Kaggle — 140k Real and Fake Faces)

    This is a simple model trained on just 500 parameters to please do not expect that much great results.
    """)

    st.divider()
    st.markdown("### 🛠️ Tech Stack")
    st.markdown("- PyTorch + torchvision\n- ResNet18 (transfer learning)\n- Streamlit UI")

    st.divider()
    if history:
        st.markdown("### 📈 Quick Stats")
        st.metric("Final Val Accuracy", f"{history['val_accuracies'][-1]*100:.1f}%")
        st.metric("Final Train Loss", f"{history['train_losses'][-1]:.3f}")

# ---------------- MAIN HEADER ----------------
st.markdown('<p class="main-title">DEEPFAKE</p>', unsafe_allow_html=True)
st.markdown('<p class="subtitle">Upload a face image to check if it\'s real or AI-generated</p>', unsafe_allow_html=True)

# ---------------- TABS ----------------
tab1, tab2 = st.tabs(["🔍  Detect", "📊  Training Results"])

# ---------------- TAB 1: DETECTION ----------------
with tab1:
    col1, col2 = st.columns([1, 1], gap="large")

    with col1:
        st.markdown("#### Upload Image")
        uploaded = st.file_uploader("", type=["jpg", "jpeg", "png"], label_visibility="collapsed")

        if uploaded:
            img = Image.open(uploaded).convert("RGB")
            st.image(img, use_container_width=True)

    with col2:
        st.markdown("#### Result")
        if uploaded:
            img_tensor = transform(img).unsqueeze(0)
            with torch.no_grad():
                prob_fake = torch.sigmoid(model(img_tensor)).item()

            label = "FAKE" if prob_fake > 0.5 else "REAL"
            confidence = prob_fake if label == "FAKE" else 1 - prob_fake
            card_class = "fake-card" if label == "FAKE" else "real-card"
            emoji = "🔴" if label == "FAKE" else "🟢"

            st.markdown(f"""
            <div class="result-card {card_class}">
                <h1 style="margin:0;">{emoji} {label}</h1>
                <p style="color:#aaa; margin-top:0.5rem;">Confidence: <b>{confidence*100:.1f}%</b></p>
            </div>
            """, unsafe_allow_html=True)

            st.write("")
            st.progress(confidence)
            st.caption(f"Raw fake probability score: {prob_fake:.4f}")

            with st.expander("What does this mean?"):
                st.write(
                    "The model outputs a probability between 0 and 1 representing how likely "
                    "the image is AI-generated. A score above 0.5 is classified as **FAKE**, "
                    "below 0.5 as **REAL**. Confidence reflects how far the score sits from the "
                    "0.5 decision boundary."
                )
        else:
            st.info("👈 Upload an image to see the prediction here.")

# ---------------- TAB 2: TRAINING RESULTS ----------------
with tab2:
    st.markdown("Progressing")
    # if history:
    #     losses = history["train_losses"]
    #     accs = history["val_accuracies"]

    #     m1, m2, m3, m4 = st.columns(4)
    #     with m1:
    #         st.markdown(f'<div class="metric-box"><h3>{len(losses)}</h3><p>Epochs</p></div>', unsafe_allow_html=True)
    #     with m2:
    #         st.markdown(f'<div class="metric-box"><h3>{losses[-1]:.3f}</h3><p>Final Loss</p></div>', unsafe_allow_html=True)
    #     with m3:
    #         st.markdown(f'<div class="metric-box"><h3>{accs[-1]*100:.1f}%</h3><p>Final Accuracy</p></div>', unsafe_allow_html=True)
    #     with m4:
    #         improvement = (accs[-1] - accs[0]) * 100
    #         st.markdown(f'<div class="metric-box"><h3>+{improvement:.1f}%</h3><p>Accuracy Gain</p></div>', unsafe_allow_html=True)

    #     st.write("")
    #     plt.style.use('dark_background')
    #     fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4))
    #     fig.patch.set_facecolor('#0e1117')

    #     ax1.plot(range(1, len(losses)+1), losses, marker='o', color='#ff4b4b', linewidth=2)
    #     ax1.set_title("Training Loss", fontsize=12, fontweight='bold')
    #     ax1.set_xlabel("Epoch"); ax1.set_ylabel("Loss")
    #     ax1.set_facecolor('#0e1117')
    #     ax1.grid(alpha=0.2)

    #     ax2.plot(range(1, len(accs)+1), accs, marker='o', color='#4bff88', linewidth=2)
    #     ax2.set_title("Validation Accuracy", fontsize=12, fontweight='bold')
    #     ax2.set_xlabel("Epoch"); ax2.set_ylabel("Accuracy")
    #     ax2.set_facecolor('#0e1117')
    #     ax2.grid(alpha=0.2)

    #     st.pyplot(fig)

    #     with st.expander("📋 Model Architecture Details"):
    #         st.markdown("""
    #         | Component | Details |
    #         |---|---|
    #         | Backbone | ResNet18 (pretrained on ImageNet) |
    #         | Frozen parameters | 11,176,512 |
    #         | Trainable parameters | 513 |
    #         | Loss function | Binary Cross-Entropy |
    #         | Optimizer | Adam (lr=0.001) |
    #         | Batch size | 16 |
    #         """)
    # else:
    #     st.warning("⚠️ `training_history.json` not found. Run training with history-saving enabled to see graphs here.")

st.divider()
st.caption("DEEPFAKE — Project by Yogesh, Ayush and Nitin")
