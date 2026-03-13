import streamlit as st
import torch
import pdfplumber
from PIL import Image
from transformers import pipeline

# Use cache_resource so the model stays in memory and doesn't reload
@st.cache_resource
def load_summarizer():
    # Use a smaller/distilled model if you still face memory errors
    # "sshleifer/distilbart-cnn-12-6" is a lighter version of BART
    return pipeline("summarization", model="facebook/bart-large-cnn")

@st.cache_resource
def load_captioner():
    return pipeline("image-to-text", model="Salesforce/blip-image-captioning-base")

def get_image_caption(image_file):
    captioner = load_captioner()
    image = Image.open(image_file).convert("RGB")
    result = captioner(image)
    return result[0]['generated_text']

def summarize_pdf(pdf_file):
    text = ""
    with pdfplumber.open(pdf_file) as pdf:
        for page in pdf.pages:
            content = page.extract_text()
            if content:
                text += content + " "
    
    if not text.strip():
        return "No readable text found in PDF."
    
    return generate_summary(text)

def generate_summary(text):
    summarizer = load_summarizer()
    # Truncate text to ~3000 characters to stay within BART's token limit
    # and prevent memory crashes
    truncated_text = text[:3000] 
    summary = summarizer(truncated_text, max_length=150, min_length=40, do_sample=False)
    return summary[0]['summary_text']
