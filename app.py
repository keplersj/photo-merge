import streamlit as st
from PIL import Image
from transformers import BlipProcessor, BlipForConditionalGeneration
from diffusers import StableDiffusionPipeline

processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-large")
model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-large")
pipe = StableDiffusionPipeline.from_pretrained("CompVis/stable-diffusion-v1-4")

file_name = st.file_uploader("Upload image 1")

if file_name is not None:
    image = Image.open(file_name)

    inputs = processor(image, return_tensors="pt")

    out = model.generate(**inputs)
    description = processor.decode(out[0], skip_special_tokens=True)
    st.write(description)
    
    image = pipe(description).images[0]
    st.image(image)
