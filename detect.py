import cv2
import sys
import numpy as np

def detect_emotion(face_image):
    """
    Detect emotion from a face image using smile detection and facial analysis.
    Returns the dominant emotion as a string.
    """
    try:
        # Load smile cascade classifier
        smile_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_smile.xml')
        eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye.xml')
        
        # Convert to grayscale
        face_gray = cv2.cvtColor(face_image, cv2.COLOR_BGR2GRAY)
        
        # Detect smile
        smiles = smile_cascade.detectMultiScale(face_gray, scaleFactor=1.8, minNeighbors=20, minSize=(25, 25))
        
        # Detect eyes
        eyes = eye_cascade.detectMultiScale(face_gray, scaleFactor=1.1, minNeighbors=10, minSize=(15, 15))
        
        # Calculate image statistics for additional analysis
        mean_val = np.mean(face_gray)
        std_val = np.std(face_gray)
        
        # Emotion detection logic
        if len(smiles) > 0:
            # Smile detected
            return 'Happy 😊'
        elif len(eyes) >= 2 and std_val > 45:
            # Eyes open wide - could be surprise
            if mean_val > 110:
                return 'Surprise 😮'
            else:
                return 'Neutral 😐'
        elif mean_val < 90:
            # Darker face (shadows) - possibly sad or angry
            if std_val < 35:
                return 'Sad 😢'
            else:
                return 'Angry 😠'
        else:
            # Default to neutral
            return 'Neutral 😐'
    except Exception as e:
        return 'Neutral 😐'

def detect_faces(image_path=None, use_webcam=False, detect_emotions=False):
    """
    Detect faces in an image or from webcam feed using OpenCV's Haar Cascade classifier.
    
    Args:
        image_path: Path to the image file (if not using webcam)
        use_webcam: If True, uses webcam for real-time detection
        detect_emotions: If True, detects emotions on faces
    """
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    
    if use_webcam:
        cap = cv2.VideoCapture(0)
        
        print("Press 'q' to quit")
        
        while True:
            ret, frame = cap.read()
            if not ret:
                print("Failed to grab frame")
                break
          
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            
            
            faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
            
        
            for (x, y, w, h) in faces:
                cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
                
                # Detect emotion if enabled
                if detect_emotions:
                    face_roi = frame[y:y+h, x:x+w]
                    emotion = detect_emotion(face_roi)
                    cv2.putText(frame, emotion, (x, y-10), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
            
      
            cv2.putText(frame, f'Faces: {len(faces)}', (10, 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            
            
            cv2.imshow('Face Detection', frame)
            
            
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        
        cap.release()
        cv2.destroyAllWindows()
    
    elif image_path:
        image = cv2.imread(image_path)
        
        if image is None:
            print(f"Error: Could not load image from {image_path}")
            return
        
        
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
       
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
        
        print(f"Found {len(faces)} face(s)")
        
       
        for (x, y, w, h) in faces:
            cv2.rectangle(image, (x, y), (x+w, y+h), (0, 255, 0), 2)
            
            # Detect emotion if enabled
            if detect_emotions:
                face_roi = image[y:y+h, x:x+w]
                emotion = detect_emotion(face_roi)
                cv2.putText(image, emotion, (x, y-10), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
                print(f"Face at ({x}, {y}): {emotion}")
        
        cv2.imshow('Face Detection with Emotions' if detect_emotions else 'Face Detection', image)
        print("Press any key to close the window")
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    
    else:
        print("Error: Please provide either an image path or set use_webcam=True")


if __name__ == "__main__":
    # Run with emotion detection enabled
    detect_faces(use_webcam=True, detect_emotions=True)
    
