from PIL import Image
import requests
import streamlit as st

uploaded_image = st.file_uploader(
    "Choose an image to colorize...",
    type=["jpg", "jpeg", "png"],
)

if uploaded_image is not None:
    # Display the uploaded image
    image = Image.open(uploaded_image)
    st.image(image, caption="Uploaded Image.", use_column_width=True)

    # Button to run inference
    if st.button("Run Inference"):
        st.write("Running inference...")

        # Run inference on the image
        response = requests.post(
            "http://localhost:8000/colorize", files={"image": uploaded_image.getvalue()}
        )

        # Display the result
        st.write(f"Inference Result: {response.json()['class']}")
