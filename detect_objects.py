import cv2
import numpy as np

# --- 1. LOAD YOLO MODEL AND CLASS NAMES ---
# Load the pre-trained YOLO model from the downloaded files
# We use YOLOv3-tiny for better performance on standard computers
# NEW LINE
net = cv2.dnn.readNet("yolov4-tiny.weights", "yolov4-tiny.cfg")

# Load all the class names from the coco.names file
classes = []
with open("coco.names", "r") as f:
    classes = [line.strip() for line in f.readlines()]

# Get the names of the output layers in the YOLO network
layer_names = net.getLayerNames()
output_layers = [layer_names[i - 1] for i in net.getUnconnectedOutLayers()]

# Generate different colors for each class to draw bounding boxes
colors = np.random.uniform(0, 255, size=(len(classes), 3))

# --- 2. SETUP CAMERA ---
# Initialize video capture from the default webcam (index 0)
cap = cv2.VideoCapture(0)

while True:
    # Read a frame from the camera
    ret, frame = cap.read()
    if not ret:
        break
        
    height, width, channels = frame.shape

    # --- 3. PREPARE IMAGE FOR YOLO ---
    # Create a 'blob' from the image. A blob is the format YOLO expects.
    # It resizes the image to 416x416 and performs scaling.
    blob = cv2.dnn.blobFromImage(frame, 0.00392, (416, 416), (0, 0, 0), True, crop=False)
    net.setInput(blob)
    
    # --- 4. RUN OBJECT DETECTION ---
    # Perform a forward pass through the YOLO network to get the detections
    outs = net.forward(output_layers)

    # --- 5. PROCESS DETECTIONS ---
    class_ids = []
    confidences = []
    boxes = []

    # Loop through all the detections from the output layers
    for out in outs:
        for detection in out:
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]
            
            # Filter out weak detections by ensuring the confidence is above a threshold (e.g., 50%)
            if confidence > 0.5:
                # Calculate bounding box coordinates
                center_x = int(detection[0] * width)
                center_y = int(detection[1] * height)
                w = int(detection[2] * width)
                h = int(detection[3] * height)

                # Rectangle coordinates (top-left corner)
                x = int(center_x - w / 2)
                y = int(center_y - h / 2)

                boxes.append([x, y, w, h])
                confidences.append(float(confidence))
                class_ids.append(class_id)

    # Apply Non-Max Suppression to remove redundant, overlapping boxes
    indexes = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)

    # --- 6. DRAW BOXES AND LABELS ON THE FRAME ---
    font = cv2.FONT_HERSHEY_PLAIN
    for i in range(len(boxes)):
        if i in indexes:
            x, y, w, h = boxes[i]
            label = str(classes[class_ids[i]])
            confidence_score = confidences[i]
            color = colors[class_ids[i]]
            
            # Draw the bounding box rectangle
            cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
            
            # Prepare the label text with class name and confidence
            text = f"{label.upper()} {int(confidence_score * 100)}%"
            
            # Draw the label text above the rectangle
            cv2.putText(frame, text, (x, y - 5), font, 1.5, color, 2)

    # --- 7. DISPLAY THE RESULT ---
    # Show the final frame in a window
    cv2.imshow("Object Detector", frame)

    # Wait for 1 millisecond, and exit the loop if the 'q' key is pressed
    key = cv2.waitKey(1)
    if key == ord('q'):
        break

# --- 8. CLEANUP ---
# Release the camera and destroy all windows
cap.release()
cv2.destroyAllWindows()