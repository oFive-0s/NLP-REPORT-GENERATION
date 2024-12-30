import cv2
import pandas as pd
from ultralytics import YOLO
import cvzone

# Open output file for writing
output_file = open("output.txt", "w")

# Load the model
model = YOLO('best.pt')

def RGB(event, x, y, flags, param):
    if event == cv2.EVENT_MOUSEMOVE:  
        point = [x, y]
        print(point)

cv2.namedWindow('RGB')
cv2.setMouseCallback('RGB', RGB)

# Open the video capture
cap = cv2.VideoCapture('cr2.mp4')

# Read class list
my_file = open("data1.txt", "r")
data = my_file.read()
class_list = data.split("\n") 

count = 0

# Check if the video opened successfully
if not cap.isOpened():
    output_file.write("Error: Could not open video.\n")
    output_file.close()
    exit()

while True:    
    # Read a frame from the video
    ret, frame = cap.read()
    
    # If no frame is read (end of video), break the loop
    if not ret:
        output_file.write("Video ended. Stopping execution.\n")
        break

    count += 1
    if count % 3 != 0:
        continue
    
    frame = cv2.resize(frame, (1020, 500))
    results = model.predict(frame)
 
    a = results[0].boxes.data
    px = pd.DataFrame(a).astype("float")

    for index, row in px.iterrows():
        x1 = int(row[0])
        y1 = int(row[1])
        x2 = int(row[2])
        y2 = int(row[3])
        d = int(row[5])
        c = class_list[d]
        
        # Write detection information to output file
        output_line = f"Frame {count}: Object at ({x1},{y1},{x2},{y2}) - Class: {c}\n"
        output_file.write(output_line)
        
        if 'accident' in c:
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)
            cvzone.putTextRect(frame, f'{c}', (x1, y1), 1, 1)
        else:
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cvzone.putTextRect(frame, f'{c}', (x1, y1), 1, 1)
    
    cv2.imshow("RGB", frame)
    
    # Break the loop if 'ESC' key is pressed
    if cv2.waitKey(1) & 0xFF == 27:
        output_file.write("Execution stopped by user.\n")
        break

# Release resources
cap.release()  
cv2.destroyAllWindows()

# Close the output file
output_file.close()