import streamlit as st
from PIL import Image
from transformers import BlipProcessor, BlipForConditionalGeneration
from diffusers import StableDiffusionPipeline

processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-large")
model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-large")
pipe = StableDiffusionPipeline.from_pretrained("CompVis/stable-diffusion-v1-4")

captions = []

with st.sidebar:
    image_gen_guidance = st.slider("Stable Diffusion: Guidance Scale", value=7.5)
    image_gen_steps = st.slider("stable Diffusion: Inference Steps", value=50)

col1, col2 = st.columns(2)

with col1:
    files = st.file_uploader("Upload images to blend", accept_multiple_files=True)

    for file_name in files:
        image = Image.open(file_name)

        with st.spinner('Captioning Provided Image'):
            inputs = processor(image, return_tensors="pt")
            out = model.generate(**inputs)
            description = processor.decode(out[0], skip_special_tokens=True)
            captions.append(description)

        st.image(image, caption=description)

with col2:
    if len(captions) > 0:
        description = ' '.join(captions)

        with st.spinner('Generating Photo from Caption'):
            images = pipe(description, guidance_scale=image_gen_guidance, num_inference_steps=image_gen_steps).images

        for image in images:
            st.image(image, caption=description)
