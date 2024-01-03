import streamlit as st
from PIL import Image
from transformers import BlipProcessor, BlipForConditionalGeneration

processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-large")
model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-large")

file_name = st.file_uploader("Upload image 1")

image = Image.open(file_name)

inputs = processor(image, return_tensors="pt")

out = model.generate(**inputs)
st.write(processor.decode(out[0], skip_special_tokens=True))
