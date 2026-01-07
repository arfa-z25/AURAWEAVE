# ================= SAFETY & ENV =================
import os
os.environ["CUDA_VISIBLE_DEVICES"] = ""  # Force CPU everywhere

import warnings
warnings.filterwarnings("ignore")

# ================= IMPORTS =================
import streamlit as st
import pickle
import numpy as np
import faiss
import torch
from PIL import Image
from transformers import CLIPProcessor, CLIPModel

# ================= PATHS =================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(BASE_DIR, "database", "fashion_data.pkl")
IMAGE_DIR = os.path.join(BASE_DIR, "images2")

# ================= PAGE CONFIG =================
st.set_page_config(
    page_title="AURAWEAVE™ | AI Fashion Stylist",
    layout="wide",
)

# ================= GLOBAL STYLES =================
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Playfair+Display:wght@600;700&family=Poppins:wght@300;400;500;600&display=swap');

html, body, [class*="css"] {
    font-family: 'Poppins', sans-serif;
}

.stApp {
    background: linear-gradient(135deg, #d8cfc4, #c7b8e2, #24243e);
    color: #1f1f1f;
}

.hero {
    padding: 3.5rem 1rem 3rem;
    border-radius: 30px;
    background: linear-gradient(
        120deg,
        rgba(255,255,255,0.55),
        rgba(240,235,255,0.35)
    );
    backdrop-filter: blur(16px);
    box-shadow: 0 25px 60px rgba(0,0,0,0.15);
    margin-bottom: 2.5rem;
    text-align: center;
}

.main-title {
    font-family: 'Playfair Display', serif;
    font-size: 3.8rem;
    font-weight: 700;
    background: linear-gradient(90deg, #5c5470, #352f44);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
}

.sub-title {
    font-size: 1.2rem;
    color: #4a4458;
    margin-top: 0.7rem;
}

.card {
    padding: 1.4rem;
    border-radius: 22px;
    background: rgba(255,255,255,0.55);
    backdrop-filter: blur(14px);
    box-shadow: 0 20px 45px rgba(0,0,0,0.18);
}

.tag {
    display: inline-block;
    padding: 7px 18px;
    border-radius: 999px;
    font-weight: 500;
    font-size: 0.85rem;
    background: linear-gradient(90deg, #b8a1d9, #8e9ad6);
    color: #ffffff;
    margin-right: 10px;
}

h3 {
    font-weight: 600;
    color: #2e2940;
}

.stTabs [data-baseweb="tab"] {
    font-size: 1rem;
    font-weight: 500;
    color: #4f4b63;
}

.stTabs [aria-selected="true"] {
    background: linear-gradient(90deg, #b8a1d9, #8e9ad6);
    color: white !important;
    border-radius: 14px;
}

.stFileUploader {
    background: rgba(255,255,255,0.4);
    border-radius: 18px;
    padding: 1rem;
}

.stAlert {
    background: rgba(255,255,255,0.6);
    border-radius: 16px;
}
</style>
""", unsafe_allow_html=True)

# ================= LOAD MODEL & DATA =================
@st.cache_resource(show_spinner="Loading AI model...")
def load_all():
    model = CLIPModel.from_pretrained(
        "openai/clip-vit-base-patch32"
    ).to("cpu")

    processor = CLIPProcessor.from_pretrained(
        "openai/clip-vit-base-patch32"
    )

    with open(DATA_PATH, "rb") as f:
        filenames, embs = pickle.load(f)

    embs = embs.astype("float32")
    faiss.normalize_L2(embs)

    index = faiss.IndexFlatIP(embs.shape[1])
    index.add(embs)

    return filenames, model, processor, index

filenames, model, processor, index = load_all()

# ================= CLASSIFICATION =================
def classify_item(image):
    categories = ["skirt", "coat", "pants", "shirt", "scarf", "frock"]
    colors = ["brown", "black", "white", "denim blue", "red"]

    inputs_cat = processor(
        text=[f"a photo of a {c}" for c in categories],
        images=image,
        return_tensors="pt",
        padding=True
    )

    with torch.no_grad():
        out_cat = model(**inputs_cat).logits_per_image.softmax(dim=1)
    detected_cat = categories[out_cat.argmax().item()]

    inputs_col = processor(
        text=[f"a {c} garment" for c in colors],
        images=image,
        return_tensors="pt",
        padding=True
    )

    with torch.no_grad():
        out_col = model(**inputs_col).logits_per_image.softmax(dim=1)
    detected_col = colors[out_col.argmax().item()]

    return detected_cat, detected_col

# ================= HERO =================
st.markdown("""
<div class="hero">
    <div class="main-title">AURAWEAVE™</div>
    <div class="sub-title">Exact Category • Exact Color • Elevated Style</div>
</div>
""", unsafe_allow_html=True)

# ================= TABS =================
tab1, tab2 = st.tabs([" Image Styling", " Text Styling"])

# ================= IMAGE STYLING =================
with tab1:
    st.markdown("###  Upload & Style Your Item")

    uploaded_file = st.file_uploader(
        "Upload fashion item",
        type=["jpg", "png", "jpeg"]
    )

    if uploaded_file:
        query_img = Image.open(uploaded_file).convert("RGB")

        with st.spinner("Analyzing garment..."):
            item_type, item_color = classify_item(query_img)

        col1, col2 = st.columns([1, 2])

        with col1:
            st.image(query_img, use_container_width=True)
            st.markdown(
                f"""
                <div class="card">
                    <span class="tag">🎨 {item_color.upper()}</span>
                    <span class="tag">👗 {item_type.upper()}</span>
                </div>
                """,
                unsafe_allow_html=True
            )

        style_prompt = (
            f"a high quality street style photo of a person wearing a "
            f"{item_color} {item_type}"
        )

        text_inputs = processor(
            text=[style_prompt],
            return_tensors="pt",
            padding=True
        )

        img_inputs = processor(
            images=query_img,
            return_tensors="pt"
        )

        with torch.no_grad():
            txt_vec = model.get_text_features(**text_inputs).numpy()
            img_vec = model.get_image_features(**img_inputs).numpy()

        final_query = (0.2 * img_vec) + (0.8 * txt_vec)
        faiss.normalize_L2(final_query)

        D, I = index.search(final_query.astype("float32"), 20)

        with col2:
            st.markdown("### ✨ Styling Suggestions")
            grid = st.columns(3)
            shown = 0

            for idx in I[0]:
                if shown >= 6:
                    break

                img_path = os.path.join(IMAGE_DIR, filenames[idx])
                if not os.path.exists(img_path):
                    continue

                res_img = Image.open(img_path).convert("RGB")
                res_type, res_color = classify_item(res_img)

                if res_type == item_type and res_color == item_color:
                    with grid[shown % 3]:
                        st.image(img_path, use_container_width=True)
                    shown += 1

            if shown == 0:
                st.warning("No strict matches found.")

# ================= TEXT STYLING =================
with tab2:
    st.markdown("### 🔍 Text-Based Styling Search")

    text_query = st.text_input(
        "Describe the outfit you want",
        placeholder="e.g. red coat street style winter outfit"
    )

    if text_query:
        text_inputs = processor(
            text=[text_query],
            return_tensors="pt",
            padding=True
        )

        with torch.no_grad():
            txt_vec = model.get_text_features(**text_inputs).numpy()

        faiss.normalize_L2(txt_vec)
        D, I = index.search(txt_vec.astype("float32"), 12)

        st.markdown("### ✨ Styled Results")
        grid = st.columns(4)

        for i, idx in enumerate(I[0][:8]):
            img_path = os.path.join(IMAGE_DIR, filenames[idx])
            if os.path.exists(img_path):
                with grid[i % 4]:
                    st.image(img_path, use_container_width=True)
