import cv2
import numpy as np

# Load pre-trained YOLO model for person detection
# 'yolov3-tiny.weights' is the weights file and 'yolov3-tiny.cfg' is the configuration file
net = cv2.dnn.readNet("yolov3-tiny.weights", "yolov3-tiny.cfg")

# Get the names of all layers from the YOLO model
layer_names = net.getLayerNames()

# Get the output layers from the YOLO model
# These are the layers where YOLO makes its predictions
unconnected_layers = net.getUnconnectedOutLayers()
# Convert the indices of the unconnected layers to layer names
output_layers = [layer_names[i - 1] for i in unconnected_layers]

# Load COCO names file to label detected objects
# 'coco.names' contains the names of the classes that YOLO can detect (e.g., person, car, etc.)
with open("coco.names", "r") as f:
    classes = [line.strip() for line in f.readlines()]

# Initialize video capture from the input video file
cap = cv2.VideoCapture("video_20.mp4")  # Specify the path to your input video file

# Print video properties for reference
print("Frame width: ", cap.get(cv2.CAP_PROP_FRAME_WIDTH))  # Width of the video frames
print("Frame height: ", cap.get(cv2.CAP_PROP_FRAME_HEIGHT))  # Height of the video frames
print("Frame rate (FPS): ", cap.get(cv2.CAP_PROP_FPS))  # Frame rate (frames per second)
print("Total frames: ", cap.get(cv2.CAP_PROP_FRAME_COUNT))  # Total number of frames in the video

# Get the frame rate of the input video
fps = cap.get(cv2.CAP_PROP_FPS)
# Get the width and height of the video frames
frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

# Define the codec and create VideoWriter object for the output video
# 'XVID' is the codec used to compress the video
fourcc = cv2.VideoWriter_fourcc(*'XVID')  # You can change this to 'MJPG' or another codec if needed
# Create a VideoWriter object to save the output video
out = cv2.VideoWriter('video_20op.avi', fourcc, fps, (frame_width, frame_height))  # Specify the output file name

# Initialize a unique ID counter for detected persons
id_counter = 0
# Dictionary to store IDs for detected persons
person_dict = {}

# Function to detect persons in a frame and assign unique IDs
def detect_and_assign_id(frame, id_counter, person_dict):
    # Get the dimensions of the frame
    height, width = frame.shape[:2]
    
    # Prepare the image as input for YOLO
    # Convert the image to a blob (a format YOLO uses for input)
    blob = cv2.dnn.blobFromImage(frame, 0.00392, (416, 416), (0, 0, 0), True, crop=False)
    
    # Set the input to the YOLO network
    net.setInput(blob)
    
    # Forward pass through YOLO to get the predictions
    outs = net.forward(output_layers)

    # Initialize lists to hold the detected class IDs, confidences, and bounding boxes
    class_ids = []
    confidences = []
    boxes = []
    
    # Process each output from YOLO
    for out in outs:
        for detection in out:
            # Extract the class scores for each object detected
            scores = detection[5:]
            # Determine the class ID with the highest score
            class_id = np.argmax(scores)
            # Get the confidence (probability) for the highest score
            confidence = scores[class_id]
            # Filter only 'person' class detections with a confidence higher than 0.5
            if classes[class_id] == "person" and confidence > 0.5:
                # Calculate the center coordinates, width, and height of the bounding box
                center_x = int(detection[0] * width)
                center_y = int(detection[1] * height)
                w = int(detection[2] * width)
                h = int(detection[3] * height)
                # Calculate the top-left corner coordinates of the bounding box
                x = int(center_x - w / 2)
                y = int(center_y - h / 2)

                # Store the bounding box coordinates, confidence, and class ID
                boxes.append([x, y, w, h])
                confidences.append(float(confidence))
                class_ids.append(class_id)

    # Apply non-max suppression to remove redundant overlapping boxes with lower confidence
    indexes = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)
    
    # If any objects were detected
    if len(indexes) > 0:
        # Loop through the indices of the remaining boxes after non-max suppression
        for i in indexes.flatten():
            # Extract the coordinates of the bounding box
            x, y, w, h = boxes[i]
            # Get the label of the detected object (in this case, it should be 'person')
            label = str(classes[class_ids[i]])
            # Get the confidence score
            confidence = confidences[i]

            # If the person is detected for the first time, assign a unique ID
            if i not in person_dict:
                id_counter += 1
                person_dict[i] = id_counter

            # Draw the bounding box around the detected person
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
            # Display the assigned ID above the bounding box
            cv2.putText(frame, f"ID: {person_dict[i]}", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
    
    # Return the updated ID counter and dictionary
    return id_counter, person_dict

# Process the video frame by frame
while cap.isOpened():
    # Read a frame from the video
    ret, frame = cap.read()
    if not ret:
        print("End of video or error reading frame.")
        break

    # Detect persons in the frame and assign unique IDs
    id_counter, person_dict = detect_and_assign_id(frame, id_counter, person_dict)

    # Write the processed frame to the output video
    out.write(frame)

# Release the video capture and writer objects when done
cap.release()
out.release()
#cv2.destroyAllWindows()  # Uncomment if you need to close any open OpenCV windows
