import streamlit as st
from PIL import Image
from transformers import BlipProcessor, BlipForConditionalGeneration
from diffusers import StableDiffusionPipeline

processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-large")
model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-large")
pipe = StableDiffusionPipeline.from_pretrained("CompVis/stable-diffusion-v1-4")

files = st.file_uploader("Upload images to blend", accept_multiple_files=True)
descs = []

for file_name in files:
    image = Image.open(file_name)
    st.image(image)

    inputs = processor(image, return_tensors="pt")

    out = model.generate(**inputs)
    description = processor.decode(out[0], skip_special_tokens=True)
    descs.append(description)
    st.write(description)

if descs.count() > 0:
    description = ' '.join(descs)
    st.write(description)
    for image in pipe(description):
        st.image(image)
