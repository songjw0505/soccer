# streamlit_app.py - 축구팀 이미지 인식 버전
import os, re
from io import BytesIO
import numpy as np
import streamlit as st
from PIL import Image, ImageOps
from fastai.vision.all import *
import gdown

# ======================
# 페이지 스타일
# ======================
st.set_page_config(page_title="⚽ 축구팀 이미지 인식 AI", page_icon="⚽", layout="wide")
st.markdown("""
<style>
h1 { color:#1E88E5; text-align:center; font-weight:800; letter-spacing:-0.5px; }
.prediction-box { background:#E3F2FD; border:2px solid #1E88E5; border-radius:12px; padding:22px; text-align:center; margin:16px 0; box-shadow:0 4px 10px rgba(0,0,0,.06);}
.prediction-box h2 { color:#0D47A1; margin:0; font-size:2.0rem; }
.prob-card { background:#fff; border-radius:10px; padding:12px 14px; margin:10px 0; box-shadow:0 2px 6px rgba(0,0,0,.06); }
.prob-bar-bg { background:#ECEFF1; border-radius:6px; width:100%; height:22px; overflow:hidden; }
.prob-bar-fg { background:#4CAF50; height:100%; border-radius:6px; transition:width .5s; }
.prob-bar-fg.highlight { background:#FF6F00; }
.info-grid { display:grid; grid-template-columns:repeat(12,1fr); gap:14px; }
.card { border:1px solid #e3e6ea; border-radius:12px; padding:14px; background:#fff; box-shadow:0 2px 6px rgba(0,0,0,.05); }
.card h4 { margin:0 0 10px; font-size:1.05rem; color:#0D47A1; }
.thumb { width:100%; height:auto; border-radius:10px; display:block; }
.helper { color:#607D8B; font-size:.9rem; }
</style>
""", unsafe_allow_html=True)

st.title("⚽ 축구팀 로고 / 이미지 인식 AI (FastAI 기반)")

# ======================
# 세션 상태
# ======================
if "img_bytes" not in st.session_state:
    st.session_state.img_bytes = None
if "last_prediction" not in st.session_state:
    st.session_state.last_prediction = None

# ======================
# FastAI 모델 로드
# ======================
FILE_ID = st.secrets.get("GDRIVE_FILE_ID", "YOUR_GOOGLE_DRIVE_MODEL_ID")
MODEL_PATH = "soccer_model.pkl"

@st.cache_resource
def load_model(file_id, output_path):
    if not os.path.exists(output_path):
        url = f"https://drive.google.com/uc?id={file_id}"
        gdown.download(url, output_path, quiet=False)
    return load_learner(output_path, cpu=True)

with st.spinner("⚽ 축구팀 인식 모델 로드 중..."):
    learner = load_model(FILE_ID, MODEL_PATH)

st.success("✅ 모델 로드 완료!")

labels = [str(x) for x in learner.dls.vocab]

st.write(f"### 🏷 분류 가능한 팀 목록: {', '.join(labels)}")
st.markdown("---")


# ======================
# 팀 정보 DB (커스텀)
# ======================
TEAM_INFO = {
    "FC Barcelona": {
        "league": "La Liga",
        "star": "Lewandowski",
        "logo": "https://upload.wikimedia.org/wikipedia/en/4/47/FC_Barcelona_%28crest%29.svg"
    },
    "Real Madrid": {
        "league": "La Liga",
        "star": "Vinicius Jr",
        "logo": "https://upload.wikimedia.org/wikipedia/en/5/56/Real_Madrid_CF.svg"
    },
    "Manchester United": {
        "league": "Premier League",
        "star": "Bruno Fernandes",
        "logo": "https://upload.wikimedia.org/wikipedia/en/7/7a/Manchester_United_FC_crest.svg"
    },
    "Liverpool FC": {
        "league": "Premier League",
        "star": "Mohamed Salah",
        "logo": "https://upload.wikimedia.org/wikipedia/en/0/0c/Liverpool_FC.svg"
    },
    "Manchester City": {
        "league": "Premier League",
        "star": "Erling Haaland",
        "logo": "https://upload.wikimedia.org/wikipedia/en/e/eb/Manchester_City_FC_badge.svg"
    },
    "Chelsea FC": {
        "league": "Premier League",
        "star": "Raheem Sterling",
        "logo": "https://upload.wikimedia.org/wikipedia/en/c/cc/Chelsea_FC.svg"
    },
    "Bayern Munich": {
        "league": "Bundesliga",
        "star": "Harry Kane",
        "logo": "https://upload.wikimedia.org/wikipedia/en/1/1f/FC_Bayern_M%C3%BCnchen_logo_%282017%29.svg"
    },
    "Juventus": {
        "league": "Serie A",
        "star": "Vlahović",
        "logo": "https://upload.wikimedia.org/wikipedia/en/2/2c/Juventus_FC_2017_logo.svg"
    },
    "Paris Saint-Germain": {
        "league": "Ligue 1",
        "star": "Mbappé",
        "logo": "https://upload.wikimedia.org/wikipedia/en/a/a7/Paris_Saint-Germain_F.C..svg"
    },
}


# ======================
# 이미지 변환 함수
# ======================
def load_img(bytes_data):
    img = Image.open(BytesIO(bytes_data))
    img = ImageOps.exif_transpose(img)
    return img.convert("RGB")

# ======================
# 입력
# ======================
tab1, tab2 = st.tabs(["📸 카메라", "📁 이미지 업로드"])
new_bytes = None

with tab1:
    cam = st.camera_input("팀 로고를 촬영하세요")
    if cam:
        new_bytes = cam.getvalue()

with tab2:
    file = st.file_uploader("팀 로고 이미지 업로드", type=["jpg", "png"])
    if file:
        new_bytes = file.getvalue()

if new_bytes:
    st.session_state.img_bytes = new_bytes

# ======================
# 예측
# ======================
if st.session_state.img_bytes:
    img = load_img(st.session_state.img_bytes)

    left_col, right_col = st.columns([1,1])

    with left_col:
        st.image(img, caption="업로드된 이미지")

    with st.spinner("🔍 축구팀 분석 중..."):
        pred, pred_idx, probs = learner.predict(PILImage.create(np.array(img)))
        st.session_state.last_prediction = str(pred)

    with right_col:
        st.markdown(f"""
        <div class="prediction-box">
            <h2>{st.session_state.last_prediction}</h2>
            <p>예측된 축구팀</p>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("---")

    # 확률 막대
    st.subheader("📊 팀별 예측 확률")
    prob_list = sorted(
        [(labels[i], float(probs[i])) for i in range(len(labels))],
        key=lambda x: x[1], reverse=True
    )

    for lbl, p in prob_list:
        pct = p * 100
        hi = "highlight" if lbl == st.session_state.last_prediction else ""
        st.markdown(
            f"""
            <div class="prob-card">
              <div style="display:flex;justify-content:space-between;margin-bottom:6px;">
                <strong>{lbl}</strong><span>{pct:.2f}%</span>
              </div>
              <div class="prob-bar-bg">
                <div class="prob-bar-fg {hi}" style="width:{pct:.4f}%;"></div>
              </div>
            </div>
            """, unsafe_allow_html=True
        )

    # ======================
    # 팀 정보 표시
    # ======================
    st.markdown("---")
    st.subheader("📌 팀 정보")

    team = st.session_state.last_prediction

    if team in TEAM_INFO:
        info = TEAM_INFO[team]

        st.image(info["logo"], width=150)
        st.write(f"### 🏆 팀명: **{team}**")
        st.write(f"🌍 리그: **{info['league']}**")
        st.write(f"⭐ 대표 선수: **{info['star']}**")

    else:
        st.info("해당 팀 정보가 등록되지 않았습니다.")
else:
    st.info("축구팀 로고를 업로드하면 분석이 시작됩니다!")
