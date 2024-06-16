import cv2
import pygame
from ultralytics import YOLO

# Initialize Pygame and Pygame Mixer
pygame.init()
pygame.mixer.init()

# Load sound files for the piano notes
sound_files = [
    "do-80236.mp3",
    "re-78500.mp3",
    "mi-80239.mp3",
    "fa-78409.mp3",
    "la-80237.mp3",
    "si-80238.mp3",
    "re-78500.mp3",
    "g6-82013.mp3"
]
sounds = [pygame.mixer.Sound(file) for file in sound_files]

# Define the piano keys as tiles (x, y, width, height)
key_width = 101  # Width of each key
key_height = 480  # Height of each key
key_spacing = 100  # Space between each key

piano_keys = [
    (i * (key_width + key_spacing), 180, key_width, key_height) 
    for i in range(8)
]

# Load the YOLOv8 model for person detection
model = YOLO('yolov8n.pt')

def draw_piano(image, piano_keys):
    """Draws the piano keys on the image."""
    for key in piano_keys:
        x, y, w, h = key
        cv2.rectangle(image, (x, y), (x + w, y + h), (255, 0, 0), 2)

def detect_person(frame):
    """Detects people in the frame using YOLOv8."""
    results = model(frame, classes=[0])  # Class 0 for detecting people
    detected = False
    person_boxes = []

    # Iterate through each result and get bounding boxes
    for result in results:
        for box in result.boxes:
            # Extract bounding box coordinates and add them to person_boxes
            person_box = (int(box.xyxy[0][0]), int(box.xyxy[0][1]), int(box.xyxy[0][2]), int(box.xyxy[0][3]))
            person_boxes.append(person_box)

            # Draw the bounding box on the frame
            cv2.rectangle(frame, (person_box[0], person_box[1]), (person_box[2], person_box[3]), (0, 255, 0), 2)

            # Set detected flag to True since a person was found
            detected = True

    return frame, detected, person_boxes

def get_piano_key(box, piano_keys):
    """Returns the index of the piano key that the person's bounding box overlaps with."""
    x1, y1, x2, y2 = box  # Person's bounding box coordinates

    for i, key in enumerate(piano_keys):
        key_x, key_y, key_w, key_h = key
        # Calculate the coordinates of the piano key's bounding box
        key_x2 = key_x + key_w
        key_y2 = key_y + key_h

        # Check for overlap between the person's bounding box and the piano key's bounding box
        if (x1 <= key_x2 and x2 >= key_x and y1 <= key_y2 and y2 >= key_y):
            return i  # Return the index of the piano key that overlaps with the bounding box

    return -1

# Initialize video capture
cap = cv2.VideoCapture(0)
cap.set(3, 1780)
cap.set(4, 720)

previous_key = -1
while cap.isOpened():
    # Read a frame from the camera
    ret, frame = cap.read()
    if not ret:
        break

    # Mirror the frame for a better user experience
    frame = cv2.flip(frame, 1)

    # Detect people and get their coordinates
    frame, detected, person_boxes = detect_person(frame)
    
    # Draw the piano keys on the frame
    draw_piano(frame, piano_keys)

    # Display the frame
    cv2.imshow('Person Detection and Piano Playing', frame)

    # Check if a person's bounding box overlaps with a piano key
    if detected:
        for box in person_boxes:
            # Iterate through detected bounding boxes
            key_index = get_piano_key(box, piano_keys)
            
            # Play the sound of the key that the person's bounding box overlaps with
            if key_index != -1 and key_index != previous_key:
                sounds[key_index].play()
                previous_key = key_index
    else:
        previous_key = -1
    
    # Break the loop if 'q' key is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the video capture object and close the OpenCV window
cap.release()
cv2.destroyAllWindows()
