from pathlib import Path

import cv2
import numpy as np
import requests
import streamlit as st
import torch
from PIL import Image
from fastai.learner import load_learner
from patchify import patchify, unpatchify


def download_file_from_google_drive(id, destination):
    URL = "https://docs.google.com/uc?export=download"

    session = requests.Session()

    response = session.get(URL, params = { 'id' : id }, stream = True)
    token = get_confirm_token(response)

    if token:
        params = { 'id' : id, 'confirm' : token }
        response = session.get(URL, params = params, stream = True)

    save_response_content(response, destination)


def get_confirm_token(response):
    for key, value in response.cookies.items():
        if key.startswith('download_warning'):
            return value

    return None


def save_response_content(response, destination):
    CHUNK_SIZE = 32768

    with open(destination, "wb") as f:
        for chunk in response.iter_content(CHUNK_SIZE):
            if chunk: # filter out keep-alive new chunks
                f.write(chunk)


def padding(array, xx, yy):
    """
    :param array: numpy array
    :param xx: desired height
    :param yy: desirex width
    :return: padded array
    """

    h = array.shape[0]
    w = array.shape[1]

    a = (xx - h) // 2
    aa = xx - a - h

    b = (yy - w) // 2
    bb = yy - b - w

    if len(array.shape) == 3:
        return np.pad(array, pad_width=((a, aa), (b, bb), (0, 0)), mode='constant')
    else:
        return np.pad(array, pad_width=((a, aa), (b, bb)), mode='constant')


def detect_cracks(model, rgb_image, dest_img_path=None):

    step = 224
    patch_shape = (224, 224, 3)

    learn_inf = model

    st.warning("Inferencing from Model...")

    large_image_inf = rgb_image
    # Make into Numpy array of RGB and get dimensions
    np_large_image_inf = np.array(large_image_inf)

    # Compiute dimensions for padding
    old_shape = np_large_image_inf.shape
    new_height = np_large_image_inf.shape[0] + (patch_shape[0] - np_large_image_inf.shape[0] % patch_shape[0])
    new_width = np_large_image_inf.shape[1] + (patch_shape[1] - np_large_image_inf.shape[1] % patch_shape[1])
    # print(old_shape)
    # print(new_height, new_width)

    # Execute padding
    np_large_image_inf = padding(np_large_image_inf, new_height, new_width)
    # print(np_large_image_inf.shape)

    # Extract patches
    patches_large_image = patchify(np_large_image_inf, patch_shape, step=step)
    # print(patches_large_image.shape)

    # Create empty result to fill
    predicted_result_patches = np.zeros(shape=patches_large_image.shape, dtype=np.uint8)

    for i in range(patches_large_image.shape[0]):
        for j in range(patches_large_image.shape[1]):

            single_patch_img = patches_large_image[i, j, 0, :, :, :]
            # print(single_patch_img.shape)

            # trasformazione in tensore
            single_patch_img_tensor = torch.from_numpy(single_patch_img.astype(np.uint8, copy=False))

            with learn_inf.no_bar(), learn_inf.no_logging():
                pred, pred_idx, probs = learn_inf.predict(single_patch_img_tensor)
            # print(pred, pred_idx, probs)

            if pred == 'no_crack':
                predicted_result_patches[i, j, 0, :, :, :] = single_patch_img
            else:
                cv2.rectangle(single_patch_img, pt1=(0,0), pt2=(223,223), color=(255,0,0), thickness=10)
                predicted_result_patches[i, j, 0, :, :, :] = single_patch_img # np.full(shape=patch_shape, fill_value=255)

    reconstructed_image = unpatchify(predicted_result_patches, np_large_image_inf.shape)

    # Eliminate padding
    reconstructed_image_clipped = reconstructed_image[:old_shape[0], :old_shape[1], :]

    # print(reconstructed_image.shape)
    # print(reconstructed_image_clipped.shape)

    # save image patch and mask
    if dest_img_path is not None:
        Image.fromarray(reconstructed_image_clipped).save(dest_img_path)

    return reconstructed_image_clipped


def load_model():
    f_checkpoint = Path("fastai2_resnet50.pkl")

    if not f_checkpoint.exists():
        with st.spinner("Downloading model... this may take a while! \n Don't stop it!"):
            # download model and weights from Google Drive
            file_id = "1f1JEKI5zxBOM8kYtJSvHVgvC6bMOxuMW"
            download_file_from_google_drive(file_id, f_checkpoint)

    # load model
    st.warning("Loading Model..🤞")
    model = load_learner(f_checkpoint)
    st.success("Loaded Model Succesfully!!🤩👍")

    return model


def do_inference(rgb_image):
    model = load_model()
    img = detect_cracks(model=model,
                        rgb_image=rgb_image)
    st.success("Crack Detection Successfully Completed!! Plotting Image..")
    st.image(img, width=720)


def main():
    image_logo = Image.open('logo_unina.jpg')
    st.image(image_logo, width=600)

    st.title("Detection of Surface Crack in Building Structures")
    st.text("With this tool you can detect crack into images of buildings.")
    st.text("This tool uses a Deep Learning approach to detect cracks.")

    image_file = st.file_uploader("Upload Image", type=['jpg', 'png', 'jpeg'])

    if image_file is None:
        st.warning("Upload Image and Run Model  (Use Image size <1MB for faster inference)")

    if image_file is not None:
        image1 = Image.open(image_file)
        rgb_im = image1.convert('RGB')
        # image2 = rgb_im.save("saved_image.jpg")
        # image_path = "saved_image.jpg"
        st.image(image1, width=720)

    if st.button("Run Model"):
        do_inference(rgb_im)


def footer():
    st.markdown("""* * *Built with ❤️ by Unina""")


if __name__ == "__main__":
    main()
    footer()
