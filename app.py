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

    inputs = processor(image, return_tensors="pt")

    out = model.generate(**inputs)
    description = processor.decode(out[0], skip_special_tokens=True)
    descs.append(description)
    st.image(image, caption=description)

if len(descs) > 0:
    description = ' '.join(descs)
    for image in pipe(description).images:
        st.image(image, caption=description)
