# streamlit_app.py
import streamlit as st
from PIL import Image
import torch
import clip
import json

# -----------------------------
# 1) 팀 리스트와 팀 정보 불러오기
# -----------------------------
teams = [
    "FC Barcelona", "Real Madrid", "Manchester United",
    "Liverpool FC", "Manchester City", "Chelsea FC",
    "Bayern Munich", "Juventus", "Paris Saint-Germain"
]

team_info = {
     "FC Barcelona": {"league": "La Liga", "star": "Lewandowski"},
     "Real Madrid": {"league": "La Liga", "star": "Vinicius Junior"},
     "Manchester United": {"league": "Premier League", "star": "Bruno Fernandes"},
     "Liverpool FC": {"league": "Premier League", "star": "Mohamed Salah"},
     "Manchester City": {"league": "Premier League", "star": "Haaland"},
     "Chelsea FC": {"league": "Premier League", "star": "Raheem Sterling"},
     "Bayern Munich": {"league": "Bundesliga", "star": "Harry Kane"},
     "Juventus": {"league": "Serie A", "star": "Vlahovic"},
     "Paris Saint-Germain": {"league": "Ligue 1", "star": "Mbappé"},
}

# -----------------------------
# 2) CLIP 모델 로드
# -----------------------------
device = "cuda" if torch.cuda.is_available() else "cpu"
model, preprocess = clip.load("ViT-B/32", device=device)

st.title("⚽ 축구팀 이미지 인식 서비스")

uploaded = st.file_uploader("축구팀 로고 또는 유니폼 사진을 업로드하세요", type=["jpg","png"])

if uploaded:
    img = Image.open(uploaded)
    st.image(img, caption="업로드된 이미지", width=300)

    # 이미지 전처리
    image_input = preprocess(img).unsqueeze(0).to(device)

    # 팀 리스트 텍스트 인코딩
    text_tokens = clip.tokenize(teams).to(device)

    with torch.no_grad():
        image_features = model.encode_image(image_input)
        text_features = model.encode_text(text_tokens)

    # 코사인 유사도로 가장 가까운 팀 찾기
    similarities = (image_features @ text_features.T).squeeze(0)
    best_idx = similarities.argmax().item()
    predicted_team = teams[best_idx]

    st.subheader(f"팀 이름 : **{predicted_team}**")

    info = team_info[predicted_team]
    st.write(f"🌍 리그: {info['league']}")
    st.write(f"⭐ 대표 선수: {info['star']}")
