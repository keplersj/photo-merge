import streamlit as st
from PIL import Image
from transformers import pipeline as transformer
from diffusers import StableDiffusionPipeline

captions = []

with st.sidebar:
    files = st.file_uploader("Upload images to blend", accept_multiple_files=True)
    st.divider()
    caption_model = st.selectbox("Caption Model", [
        "Salesforce/blip-image-captioning-large",
        "nlpconnect/vit-gpt2-image-captioning",
        "microsoft/git-base",
        "ydshieh/vit-gpt2-coco-en"
    ])
    caption_max_tokens = st.number_input("Image Caption: Max Tokens")
    st.divider()
    caption_concat_joiner = st.text_input("Caption Concatenation Joiner", value=" ")
    st.divider()
    diffusion_model = st.selectbox("Diffusion Model", [
        "stabilityai/stable-diffusion-xl-base-1.0",
        "runwayml/stable-diffusion-v1-5",
        "stabilityai/stable-diffusion-2-1",
        "CompVis/stable-diffusion-v1-4"
    ])
    image_gen_height = st.number_input("Stable Diffusion: Height", value=512)
    image_gen_width = st.number_input("Stable Diffusion: Width", value=512)
    image_gen_steps = st.slider("Stable Diffusion: Inference Steps", value=50)
    image_gen_guidance = st.slider("Stable Diffusion: Guidance Scale", value=7.5)
    image_gen_number = st.number_input("Stable Diffusion: Images Generates", value=1)

for file_name in files:
    image = Image.open(file_name)

    with st.spinner('Captioning Provided Image'):
        captioner = transformer(model=caption_model)
        caption = captioner(image, max_new_tokens=caption_max_tokens)[0]['generated_text']

    captions.append(caption)
    st.image(image, caption=caption)

if len(captions) > 0:
    st.divider()

    description = caption_concat_joiner.join(captions)

    pipe = StableDiffusionPipeline.from_pretrained(diffusion_model)

    with st.spinner(f'Generating Photo for "{description}"'):
        images = pipe(description, height=image_gen_height, width=image_gen_width, num_inference_steps=image_gen_steps, guidance_scale=image_gen_guidance, num_images_per_prompt=image_gen_number).images

    for image in images:
        st.image(image, caption=description)
