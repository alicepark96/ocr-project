import streamlit as st
import numpy as np
import cv2
from PIL import Image
import pytesseract

st.set_page_config(page_title="손글씨 인식 (OCR)", layout="wide")
st.title("📝 손글씨 사진 -> 텍스트 변환기 (OCR + CNN)")
st.write("아래에 손글씨가 적힌 사진을 업로드해 주세요. 이미지 처리 과정을 단계별로 보여줍니다.")

uploaded_file = st.file_uploader("이미지 업로드 (JPG, PNG)", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.subheader("1️⃣ 원본 이미지")
    st.image(image, caption="원본 이미지", use_column_width=True)

    # OpenCV용 변환
    img_array = np.array(image.convert("RGB"))
    gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)

    st.subheader("2️⃣ 그레이스케일 변환")
    st.image(gray, caption="Grayscale", channels="GRAY")

    # 이진화 (Threshold)
    _, thresh = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY_INV)
    st.subheader("3️⃣ 이진화 (Threshold)")
    st.image(thresh, caption="Binary Image", channels="GRAY")

    # 경계선 추출 (Optional)
    edged = cv2.Canny(thresh, 30, 150)
    st.subheader("4️⃣ 윤곽선 추출 (Edge Detection)")
    st.image(edged, caption="Edges", channels="GRAY")

    # OCR 처리 (Tesseract 사용)
    st.subheader("5️⃣ 문자 인식 결과 (OCR)")
    text = pytesseract.image_to_string(gray, lang='eng+kor')  # 한국어+영어 지원
    st.text_area("인식된 텍스트:", value=text, height=200)

    st.success("✅ 모든 단계 완료!")
