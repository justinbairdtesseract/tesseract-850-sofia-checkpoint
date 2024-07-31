import cv2
import mediapipe as mp
import threading

mp_face_detection = mp.solutions.face_detection
mp_drawing = mp.solutions.drawing_utils

face_center = (0, 0)  # Global variable to store face center coordinates
lock = threading.Lock()  # Lock for thread-safe access to face_center

def process_frame(image, display):
    global face_center  
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    results = face_detection.process(image_rgb)

    if results.detections:
        for detection in results.detections:
            bboxC = detection.location_data.relative_bounding_box
            ih, iw, _ = image.shape
            bbox = int(bboxC.xmin * iw), int(bboxC.ymin * ih), \
                   int(bboxC.width * iw), int(bboxC.height * ih)

            cx = int(bbox[0] + bbox[2] / 2)
            cy = int(bbox[1] + bbox[3] / 2)

            with lock:
                face_center = (cx, cy)

            if display:
                cv2.circle(image, (cx, cy), 5, (0, 255, 0), cv2.FILLED)
                mp_drawing.draw_detection(image, detection)

    if display:
        # Display the output
        cv2.imshow('Face Center Tracking', image)

def face_detection_thread(input_source, display):
    global face_detection
    face_detection = mp_face_detection.FaceDetection(min_detection_confidence=0.2)
    cap = cv2.VideoCapture(0) if input_source == 'webcam' else cv2.VideoCapture(input_source)

    with face_detection:
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

def start_face_detection(input_source, display):
    thread = threading.Thread(target=face_detection_thread, args=(input_source, display))
    thread.start()
    return thread

def get_face_center():
    global face_center
    with lock:
        return face_center
