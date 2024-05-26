import cv2
import numpy as np
import argparse
import os
from deepface import DeepFace

def overlay_image_alpha(img, img_overlay, pos, alpha_mask):
    x, y = pos
    y1, y2 = max(0, y), min(img.shape[0], y + img_overlay.shape[0])
    x1, x2 = max(0, x), min(img.shape[1], x + img_overlay.shape[1])
    y1o, y2o = max(0, -y), min(img_overlay.shape[0], img.shape[0] - y)
    x1o, x2o = max(0, -x), min(img_overlay.shape[1], img.shape[1] - x)

    if y1 >= y2 or x1 >= x2 or y1o >= y2o or x1o >= x2o:
        return

    img_crop = img[y1:y2, x1:x2]
    img_overlay_crop = img_overlay[y1o:y2o, x1o:x2o]
    alpha = alpha_mask[y1o:y2o, x1o:x2o, None] / 255.0
    img_crop[:] = alpha * img_overlay_crop[:, :, :3] + (1 - alpha) * img_crop

def load_face_images(folder_path):
    face_images = {}
    for filename in os.listdir(folder_path):
        if filename.endswith(".png"):
            emotion = os.path.splitext(filename)[0]
            face_image = cv2.imread(os.path.join(folder_path, filename), cv2.IMREAD_UNCHANGED)
            if face_image.shape[2] != 4:
                face_image = cv2.cvtColor(face_image, cv2.COLOR_BGR2BGRA)
            face_images[emotion] = face_image
    return face_images

parser = argparse.ArgumentParser(description='Process webcam video with dynamic face overlay or blur.')
parser.add_argument('--faces_folder', type=str, help='Path to the folder containing face images for different emotions and talking states')

args = parser.parse_args()

if args.faces_folder:
    face_images = load_face_images(args.faces_folder)
else:
    face_images = None

face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    faces = face_cascade.detectMultiScale(gray_frame, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

    if len(faces) == 0:
        # If no faces are detected, just display the frame
        cv2.imshow('Webcam', frame)
    else:
        for (x, y, w, h) in faces:
            face_region = frame[y:y+h, x:x+w]

            if face_images:
                try:
                    analysis = DeepFace.analyze(face_region, actions=['emotion'])
                    emotion = analysis[0]['dominant_emotion']
                    print("Detected emotion:", emotion)
                except Exception as e:
                    print("Error analyzing face:", e)
                    emotion = "neutral"

                overlay_face = face_images[emotion] if emotion in face_images else face_images["neutral"]
                scaling_factor = 1.8
                new_width = int(w * scaling_factor)
                new_height = int(h * scaling_factor)

                resized_face_image = cv2.resize(overlay_face, (new_width, new_height))

                x_offset = x - (new_width - w) // 2
                y_offset = y - (new_height - h) // 2

                x1 = max(0, x_offset)
                y1 = max(0, y_offset)
                x2 = min(frame.shape[1], x_offset + new_width)
                y2 = min(frame.shape[0], y_offset + new_height)

                roi_x1 = max(0, -x_offset)
                roi_y1 = max(0, -y_offset)
                roi_x2 = roi_x1 + (x2 - x1)
                roi_y2 = roi_y1 + (y2 - y1)

                if resized_face_image.shape[2] == 4:
                    alpha_mask = resized_face_image[roi_y1:roi_y2, roi_x1:roi_x2, 3] / 255.0
                    alpha_image = resized_face_image[roi_y1:roi_y2, roi_x1:roi_x2, :3]

                    # Composite the resized face image onto the frame
                    for c in range(0, 3):
                        frame[y1:y2, x1:x2, c] = (alpha_mask * alpha_image[:, :, c] + (1 - alpha_mask) * frame[y1:y2, x1:x2, c])
                else:
                    frame[y1:y2, x1:x2] = resized_face_image[roi_y1:roi_y2, roi_x1:roi_x2]
            else:
                blurred_face = cv2.GaussianBlur(face_region, (99, 99), 0)
                frame[y:y+h, x:x+w] = blurred_face

    cv2.imshow('Webcam', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()


