import streamlit as st
import pytesseract
from PIL import Image
import easyocr
import torch
import numpy as np
from transformers import AutoProcessor, AutoModelForImageTextToText
import time

device = "cpu"

def ocr_tesseract(image):
    return pytesseract.image_to_string(image)

def load_easyocr_reader():
    return easyocr.Reader(['en','id'], gpu=False)

def ocr_easyocr(image):
    reader = load_easyocr_reader()
    image_np = np.array(image)
    result = reader.readtext(image_np, detail=0)
    return ' '.join(result)

def load_gotocr_model_processor():
    model = AutoModelForImageTextToText.from_pretrained("stepfun-ai/GOT-OCR-2.0-hf").to(device)
    processor = AutoProcessor.from_pretrained("stepfun-ai/GOT-OCR-2.0-hf")
    return model, processor

def ocr_gotocr(image):
    model, processor= load_gotocr_model_processor()
    inputs = processor(image, return_tensors="pt").to(device)
    generate_ids = model.generate(
        **inputs,
        do_sample=False,
        tokenizer=processor.tokenizer,
        stop_strings="<|im_end|>",
        max_new_tokens=4096,
    )
    generated_text = processor.decode(generate_ids[0, inputs["input_ids"].shape[1]:], skip_special_tokens=True)
    return generated_text

st.set_page_config(page_title="Aplikasi OCR Sederhana", layout="centered")

st.title("Aplikasi OCR Sederhana")
st.write("Unggah gambar teks, pilih model OCR, dan dapatkan teks yang diprediksi.")

uploaded_file = st.file_uploader("Unggah Gambar Teks", type=["png", "jpg", "jpeg"])

model_choice = st.selectbox(
    "Pilih Model OCR",
    ("EasyOCR", "Tesseract", "GOTOCR")
)

predicted_text = ""
execution_time = 0.0

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("L")
    st.image(image, caption="Gambar yang Diunggah", use_column_width=True)

    st.subheader("Hasil Prediksi Teks")

    start_time = time.time()
    try:
        if model_choice == "EasyOCR":
            with st.spinner("Melakukan OCR dengan EasyOCR..."):
                predicted_text = ocr_easyocr(image)
        elif model_choice == "Tesseract":
            with st.spinner("Melakukan OCR dengan Tesseract..."):
                predicted_text = ocr_tesseract(image)
        elif model_choice == "GOTOCR":
            with st.spinner("Melakukan OCR dengan GOTOCR..."):
                predicted_text = ocr_gotocr(image)
    except Exception as e:
        predicted_text = f"Terjadi kesalahan: {e}"

    execution_time = time.time() - start_time

    st.text_area("Teks yang Diprediksi", predicted_text, height=200)
    st.write(f"Waktu Eksekusi: **{execution_time:.2f} detik**")

st.markdown("""
<style>
.reportview-container .main .block-container{
    padding-top: 2rem;
    padding-right: 2rem;
    padding-left: 2rem;
    padding-bottom: 2rem;
}
</style>
""", unsafe_allow_html=True)