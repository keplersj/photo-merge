import streamlit as st
from PIL import Image
from transformers import pipeline as transformer
from diffusers import StableDiffusionPipeline

pipe = StableDiffusionPipeline.from_pretrained("CompVis/stable-diffusion-v1-4")

captions = []

with st.sidebar:
    files = st.file_uploader("Upload images to blend", accept_multiple_files=True)
    st.divider()
    caption_model = st.selectbox("Caption Model", [
        "ydshieh/vit-gpt2-coco-en",
        "Salesforce/blip-image-captioning-large",
        "nlpconnect/vit-gpt2-image-captioning",
        "microsoft/git-base"
    ])
    st.divider()
    image_gen_guidance = st.slider("Stable Diffusion: Guidance Scale", value=7.5)
    image_gen_steps = st.slider("stable Diffusion: Inference Steps", value=50)

col1, col2 = st.columns(2)

with col1:
    for file_name in files:
        image = Image.open(file_name)

        with st.spinner('Captioning Provided Image'):
            captioner = transformer(model=caption_model)
            caption = captioner(image)[0].generated_text

        captions.append(caption)
        st.image(image, caption=caption)

with col2:
    if len(captions) > 0:
        description = ' '.join(captions)

        with st.spinner(f'Generating Photo for {description}'):
            images = pipe(description, guidance_scale=image_gen_guidance, num_inference_steps=image_gen_steps).images

        for image in images:
            st.image(image, caption=description)
