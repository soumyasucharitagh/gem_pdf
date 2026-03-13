import streamlit as st
from utils import get_image_caption, summarize_pdf, summarize_text

st.set_page_config(page_title="AI Multi-Tool App", layout="wide")

st.sidebar.title("Navigation")
choice = st.sidebar.radio("Go to", ["Image Captioning", "PDF Summarizer", "Text Summarizer"])

st.title(f"🚀 {choice}")

# --- Image Captioning UI ---
if choice == "Image Captioning":
    uploaded_image = st.file_uploader("Upload an image...", type=["jpg", "jpeg", "png"])
    if uploaded_image:
        st.image(uploaded_image, caption="Uploaded Image", use_column_width=True)
        if st.button("Generate Caption"):
            with st.spinner("Analyzing image..."):
                caption = get_image_caption(uploaded_image)
                st.success(f"**Caption:** {caption}")

# --- PDF Summarizer UI ---
elif choice == "PDF Summarizer":
    uploaded_pdf = st.file_uploader("Upload a PDF file...", type=["pdf"])
    if uploaded_pdf:
        if st.button("Summarize PDF"):
            with st.spinner("Reading and summarizing PDF..."):
                summary = summarize_pdf(uploaded_pdf)
                st.subheader("Summary:")
                st.write(summary)

# --- Text Summarizer UI ---
elif choice == "Text Summarizer":
    input_text = st.text_area("Enter text to summarize:", height=200)
    if st.button("Summarize Text"):
        if input_text:
            with st.spinner("Generating summary..."):
                summary = summarize_text(input_text)
                st.subheader("Summary:")
                st.write(summary)
        else:
            st.warning("Please enter some text.")
