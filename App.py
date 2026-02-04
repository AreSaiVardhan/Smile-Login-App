import streamlit as st
import cv2
import numpy as np
from PIL import Image

# -------------------- Page Config --------------------
st.set_page_config(page_title="Smile Based Login", page_icon="😄")

st.title("😄 Smile Based Login System")
st.write("Smile and capture your photo to login")

# -------------------- Session State --------------------
if "show_camera" not in st.session_state:
    st.session_state.show_camera = False

if "logged_in" not in st.session_state:
    st.session_state.logged_in = False

if "play_video" not in st.session_state:
    st.session_state.play_video = False

# -------------------- Load Haar Cascades --------------------
face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
)
smile_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_smile.xml"
)

# -------------------- LOGIN BUTTON --------------------
if not st.session_state.logged_in:
    if st.button("🔐 Login"):
        st.session_state.show_camera = True

# -------------------- CAMERA SECTION --------------------
if st.session_state.show_camera and not st.session_state.logged_in:
    st.subheader("📷 Start Camera to Login")
    st.info("Smile clearly and click Take Photo 😄")

    camera_image = st.camera_input("Capture Image")

    if camera_image is not None:
        # Original image
        original_image = Image.open(camera_image)

        # Convert to OpenCV format
        frame = np.array(original_image)
        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, 1.1, 4)

        smile_detected = False

        for (x, y, w, h) in faces:
            cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)

            roi_gray = gray[y:y+h, x:x+w]
            roi_color = frame[y:y+h, x:x+w]

            smiles = smile_cascade.detectMultiScale(
                roi_gray,
                scaleFactor=1.7,
                minNeighbors=20
            )

            if len(smiles) > 0:
                smile_detected = True

                for (sx, sy, sw, sh) in smiles:
                    cv2.rectangle(
                        roi_color,
                        (sx, sy),
                        (sx+sw, sy+sh),
                        (0, 255, 0),
                        2
                    )

        processed_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # -------------------- SIDE BY SIDE DISPLAY --------------------
        col1, col2 = st.columns(2)

        with col1:
            st.image(original_image, caption="📸 Captured Image")

        with col2:
            st.image(processed_frame, caption="😄 Smile Detection Result")

        # -------------------- LOGIN CONDITION --------------------
        if smile_detected:
            st.success("✅ Smile detected! Login successful 😄")
            st.session_state.logged_in = True
        else:
            st.warning("🙂 No smile detected. Please try again.")

# -------------------- PLAY VIDEO BUTTON --------------------
if st.session_state.logged_in and not st.session_state.play_video:
    if st.button("▶️ Play Video"):
        st.session_state.play_video = True

# -------------------- VIDEO SECTION --------------------
if st.session_state.play_video:
    st.subheader("🎬 Welcome Video")

    with open("los_angeles.mp4", "rb") as video_file:
        st.video(video_file.read())

