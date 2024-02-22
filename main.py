import cv2

#Load pre-trained face detector
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Load pre-trained face recognition model
# face_recognizer = cv2.faces.EigenFaceRecognizer_create()
# face_recognizer.read('trained_model.yml')

# Initialize video capture
cap = cv2.VideoCapture("Video_recognition_T-2/video.mp4")

while True:
    # Capture frame-by-frame
    ret, frame = cap.read()
    
    # Convert frame to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    # Detect faces in the frame
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
    
    # Draw rectangle around each face and recognize
    for (x, y, w, h) in faces:
        roi_gray = gray[y:y+h, x:x+w]
        # id_, conf = face_recognizer.predict(roi_gray)
            # Recognized face
        print("Recognized")
        # Draw rectangle around recognized face
        color = (255, 0, 0) # BGR
        stroke = 2
        cv2.rectangle(frame, (x, y), (x+w, y+h), color, stroke)
        # Put text on recognized face
        font = cv2.FONT_HERSHEY_SIMPLEX
        name = "Person "
        cv2.putText(frame, name, (x,y), font, 1, (255,255,255), 2, cv2.LINE_AA)

    
    # Display the resulting frame
    cv2.imshow('frame', frame)
    if cv2.waitKey(20) & 0xFF == ord('q'):
        break

# Release video capture
cap.release()
cv2.destroyAllWindows()
