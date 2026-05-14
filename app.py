import streamlit as st
import cv2
import numpy as np
from PIL import Image

st.set_page_config(page_title="Blurred Image Restoration")

st.title("🔧 Blurred Image Restoration System")

uploaded_file = st.file_uploader(
    "Upload Blurred Image",
    type=["jpg", "jpeg", "png"]
)

if uploaded_file is not None:

    image = Image.open(uploaded_file)

    image_np = np.array(image)

    st.subheader("Original Blurred Image")
    st.image(image_np, use_column_width=True)

    operation = st.selectbox(
        "Choose Restoration Technique",
        [
            "Gaussian Deblurring",
            "Median Restoration",
            "Sharpening Restoration",
            "Bilateral Filter",
            "Wiener Approximation"
        ]
    )

    if operation == "Gaussian Deblurring":

        processed = cv2.GaussianBlur(
            image_np,
            (3, 3),
            0
        )

        kernel = np.array([
            [0, -1, 0],
            [-1, 5,-1],
            [0, -1, 0]
        ])

        processed = cv2.filter2D(
            processed,
            -1,
            kernel
        )

    elif operation == "Median Restoration":

        processed = cv2.medianBlur(
            image_np,
            3
        )

    elif operation == "Sharpening Restoration":

        kernel = np.array([
            [-1,-1,-1],
            [-1, 9,-1],
            [-1,-1,-1]
        ])

        processed = cv2.filter2D(
            image_np,
            -1,
            kernel
        )

    elif operation == "Bilateral Filter":

        processed = cv2.bilateralFilter(
            image_np,
            9,
            75,
            75
        )

    elif operation == "Wiener Approximation":

        gray = cv2.cvtColor(
            image_np,
            cv2.COLOR_RGB2GRAY
        )

        blur = cv2.GaussianBlur(
            gray,
            (5,5),
            0
        )

        processed = cv2.addWeighted(
            gray,
            1.5,
            blur,
            -0.5,
            0
        )

    st.subheader("Restored Image")
    st.image(processed, use_column_width=True)

    # Download Option

    if len(processed.shape) == 2:
        processed_pil = Image.fromarray(processed)
    else:
        processed_pil = Image.fromarray(
            processed.astype(np.uint8)
        )

    st.download_button(
        label="Download Restored Image",
        data=processed_pil.tobytes(),
        file_name="restored_image.png",
        mime="image/png"
    )