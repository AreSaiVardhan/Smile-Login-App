import streamlit as st
import cv2

# -------------------- Page Config --------------------
st.set_page_config(page_title="Smile Login", page_icon="😄")

st.title("😄 Smile Based Login System")

# -------------------- Session State --------------------
if "show_camera_popup" not in st.session_state:
    st.session_state.show_camera_popup = False

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
        st.session_state.show_camera_popup = True

# -------------------- CAMERA POPUP --------------------
if st.session_state.show_camera_popup and not st.session_state.logged_in:
    with st.expander("📷 Start Camera to Login", expanded=True):

        run = st.checkbox("Start Camera")
        FRAME_WINDOW = st.image([])

        cap = cv2.VideoCapture(0)

        while run and not st.session_state.logged_in:
            ret, frame = cap.read()
            if not ret:
                st.error("Camera not accessible")
                break

            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = face_cascade.detectMultiScale(gray, 1.1, 4)

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
                    st.session_state.smile_count += 1

                    cv2.putText(
                        frame,
                        f"Smiling... {st.session_state.smile_count}",
                        (x, y-10),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.9,
                        (0, 255, 0),
                        2
                    )

                    for (sx, sy, sw, sh) in smiles:
                        cv2.rectangle(
                            roi_color,
                            (sx, sy),
                            (sx+sw, sy+sh),
                            (0, 255, 0),
                            2
                        )

                    if st.session_state.smile_count >= 15:
                        st.session_state.logged_in = True
                        st.session_state.show_video = True


                else:
                    st.session_state.smile_count = 0

            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            FRAME_WINDOW.image(frame)

        if not run:
            cap.release()

# -------------------- LOGIN SUCCESS + VIDEO --------------------
if st.session_state.show_video:
    st.success("✅ Login Successful 😄")
    st.balloons()

    st.subheader("🎬 Welcome Video")

    video_file = open("los_angeles.mp4", "rb")
    video_bytes = video_file.read()

    st.video(video_bytes)

