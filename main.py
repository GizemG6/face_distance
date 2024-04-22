import cv2
import numpy as np

# Distance from camera to object (measured in centimeters)
Known_distance = 60

# Width of face in the real world or Object Plane (centimeters)
Known_width = 23

# Colors
GREEN = (0, 255, 0)
RED = (0, 0, 255)
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)

# Font
fonts = cv2.FONT_HERSHEY_COMPLEX

# Face detector object
face_detector = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')


# Focal length finder function
def Focal_Length_Finder(measured_distance, real_width, width_in_rf_image):
    focal_length = (width_in_rf_image * measured_distance) / real_width
    return focal_length


# Distance estimation function
def Distance_finder(Focal_Length, real_object_width, object_width_in_frame):
    distance = (real_object_width * Focal_Length) / object_width_in_frame
    return distance


# Initialize the camera
cap = cv2.VideoCapture(0)


# Get reference face width in the frame
def get_reference_face_width():
    _, frame = cap.read()
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_detector.detectMultiScale(gray_frame, 1.3, 5)
    if len(faces) > 0:
        x, y, w, h = faces[0]
        return w  # Reference face width in pixels
    return None


# Get focal length from camera (using reference image)
ref_width = get_reference_face_width()
if ref_width is not None:
    Focal_length_found = Focal_Length_Finder(Known_distance, Known_width, ref_width)
else:
    print("Referans genişlik algılanamadı. Kapatılıyor...")
    cap.release()
    cv2.destroyAllWindows()
    exit(1)

# Main loop to capture frames and calculate distance
while True:
    _, frame = cap.read()

    # Detect faces in the frame
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_detector.detectMultiScale(gray_frame, 1.3, 5)

    if len(faces) > 0:
        x, y, w, h = faces[0]
        face_width_in_frame = w

        # Calculate distance using the detected face width
        if face_width_in_frame > 0:
            Distance = Distance_finder(Focal_length_found, Known_width, face_width_in_frame)

            # Display distance on the frame
            cv2.putText(frame, f"Distance: {round(Distance, 2)} CM", (30, 35), fonts, 0.6, GREEN, 2)

    # Display the frame
    cv2.imshow("Distance Measurement", frame)

    # Quit the program if 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the camera and close all windows
cap.release()
cv2.destroyAllWindows()
