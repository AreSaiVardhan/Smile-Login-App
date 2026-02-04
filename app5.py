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

if "smile_count" not in st.session_state:
    st.session_state.smile_count = 0

if "logged_in" not in st.session_state:
    st.session_state.logged_in = False

if "show_video" not in st.session_state:
    st.session_state.show_video = False

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
    st.info("Click **Take Photo** while smiling 😄 (2–3 times)")

    camera_image = st.camera_input("Capture Image")

    if camera_image is not None:
        # Convert to OpenCV format
        image = Image.open(camera_image)
        frame = np.array(image)
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
                st.session_state.smile_count += 1

                for (sx, sy, sw, sh) in smiles:
                    cv2.rectangle(
                        roi_color,
                        (sx, sy),
                        (sx+sw, sy+sh),
                        (0, 255, 0),
                        2
                    )

        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        st.image(frame, caption="Processed Image")

        if smile_detected:
            st.success(f"😄 Smile detected ({st.session_state.smile_count}/3)")
        else:
            st.warning("🙂 No smile detected. Try again.")

        # -------------------- LOGIN CONDITION --------------------
        if st.session_state.smile_count >= 3:
            st.session_state.logged_in = True
            st.session_state.show_video = True
            st.session_state.show_camera = False

# -------------------- SUCCESS + VIDEO --------------------
if st.session_state.show_video:
    st.success("✅ Login Successful 😄")
    st.balloons()

    st.subheader("🎬 Welcome Video")

    with open("los_angeles.mp4", "rb") as video_file:
        video_bytes = video_file.read()
        st.video(video_bytes)
