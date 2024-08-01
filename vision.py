import cv2
import mediapipe as mp
import threading

# Initialize MediaPipe Face Mesh
mp_face_mesh = mp.solutions.face_mesh
mp_drawing = mp.solutions.drawing_utils

face_center = (0, 0)  # Global variable to store face center coordinates
left_eye_center = (0, 0)
right_eye_center = (0, 0)
lock = threading.Lock()  # Lock for thread-safe access to face_center

def process_frame(image, display):
    global face_center, left_eye_center, right_eye_center  # Use the global variable
    # Convert the BGR image to RGB
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # Process the image and detect face mesh
    results = face_mesh.process(image_rgb)

    if results.multi_face_landmarks:
        for face_landmarks in results.multi_face_landmarks:
            ih, iw, _ = image.shape

            # Get coordinates for the face center
            cx = int(face_landmarks.landmark[1].x * iw)
            cy = int(face_landmarks.landmark[1].y * ih)

            # Get coordinates for the eyes
            left_eye_center = (int(face_landmarks.landmark[33].x * iw), int(face_landmarks.landmark[33].y * ih))
            right_eye_center = (int(face_landmarks.landmark[263].x * iw), int(face_landmarks.landmark[263].y * ih))

            # Update the global variable
            with lock:
                face_center = (cx, cy)

            # Draw the face mesh and eye positions on the image
            if display:
                mp_drawing.draw_landmarks(image, face_landmarks, mp_face_mesh.FACE_CONNECTIONS)
                cv2.circle(image, left_eye_center, 5, (255, 0, 0), cv2.FILLED)
                cv2.circle(image, right_eye_center, 5, (255, 0, 0), cv2.FILLED)
                cv2.circle(image, (cx, cy), 5, (0, 255, 0), cv2.FILLED)

    if display:
        # Display the output
        cv2.imshow('Face Mesh and Eye Tracking', image)

def face_mesh_thread(input_source, display):
    global face_mesh
    face_mesh = mp_face_mesh.FaceMesh(min_detection_confidence=0.5, min_tracking_confidence=0.5)
    cap = cv2.VideoCapture(0) if input_source == 'webcam' else cv2.VideoCapture(input_source)

    with face_mesh:
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            process_frame(frame, display)

            if display and cv2.waitKey(5) & 0xFF == 27:
                break

    cap.release()
    if display:
        cv2.destroyAllWindows()

def start_face_mesh(input_source, display):
    thread = threading.Thread(target=face_mesh_thread, args=(input_source, display))
    thread.start()
    return thread

def get_face_center():
    global face_center
    with lock:
        return face_center

def get_eye_centers():
    global left_eye_center, right_eye_center
    with lock:
        return left_eye_center, right_eye_center
