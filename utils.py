import torch
import pdfplumber
from PIL import Image
from transformers import (
    BlipProcessor, BlipForConditionalGeneration,
    BartTokenizer, BartForConditionalGeneration,
    pipeline
)

# --- 1. Image Captioning Logic ---
def get_image_caption(image_file):
    processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
    model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base")
    
    image = Image.open(image_file).convert("RGB")
    inputs = processor(image, return_tensors="pt")
    
    with torch.no_grad():
        out = model.generate(**inputs)
    return processor.decode(out[0], skip_special_tokens=True)

# --- 2. PDF Summarization Logic ---
def summarize_pdf(pdf_file):
    # Extract text from PDF
    text = ""
    with pdfplumber.open(pdf_file) as pdf:
        for page in pdf.pages:
            text += page.extract_text()
            
    # Summarize using BART
    summarizer = pipeline("summarization", model="facebook/bart-large-cnn")
    summary = summarizer(text[:1024], max_length=150, min_length=40, do_sample=False)
    return summary[0]['summary_text']

# --- 3. Text Summarization Logic (from BARTYT) ---
def summarize_text(input_text):
    model_name = "facebook/bart-large-cnn"
    tokenizer = BartTokenizer.from_pretrained(model_name)
    model = BartForConditionalGeneration.from_pretrained(model_name)
    
    inputs = tokenizer([input_text], max_length=1024, return_tensors="pt", truncation=True)
    summary_ids = model.generate(inputs["input_ids"], num_beams=4, max_length=150, early_stopping=True)
    return tokenizer.decode(summary_ids[0], skip_special_tokens=True)
