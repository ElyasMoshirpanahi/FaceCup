import cv2
import torch
import torchvision.transforms as transforms

# Load the pre-trained face detector and open mouth classifier
face_detector = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
mouth_classifier = torch.hub.load('pytorch/vision', 'mobilenet_v2', pretrained=True)
mouth_classifier.classifier[1] = torch.nn.Linear(1280, 2)

# Define a function to detect open mouths
def detect_open_mouths(frame):
    # Convert the frame to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    # Detect faces in the frame
    faces = face_detector.detectMultiScale(gray, 1.3, 5)
    
    # Loop over the detected faces
    for (x,y,w,h) in faces:
        # Draw a rectangle around the face
        cv2.rectangle(frame,(x,y),(x+w,y+h),(255,0,0),2)
        
        # Get the region of interest (ROI) for the mouth
        roi_gray = gray[y+h//2:y+h, x:x+w]
        roi_color = frame[y+h//2:y+h, x:x+w]
        
        # Resize the mouth ROI and convert to tensor
        transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor()
        ])
        mouth_tensor = transform(cv2.cvtColor(roi_color, cv2.COLOR_BGR2RGB)).unsqueeze(0)
        
        # Pass the mouth tensor through the classifier
        with torch.no_grad():
            output = mouth_classifier(mouth_tensor)
            prediction = torch.argmax(output, dim=1)
        
        # Check if the mouth is open
        if prediction == 1:
            # Mouth is open, draw a green circle
            cv2.circle(frame, (x + w//2, y + h//2), 10, (0, 255, 0), -1)
    
    # Display the frame
    cv2.imshow('Open Mouth Detector', frame)

# Open the webcam
cap = cv2.VideoCapture(0)

# Loop over the frames from the webcam
while True:
    # Read a frame from the webcam
    ret, frame = cap.read()
    
    # Detect open mouths in the frame
    detect_open_mouths(frame)
    
    # Check for key presses
    if cv2.waitKey(1) == ord('q'):
        break

# Release the webcam and close the window
cap.release()
cv2.destroyAllWindows()
