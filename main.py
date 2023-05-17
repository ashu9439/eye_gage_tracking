import cv2
import dlib
import numpy as np
import pyautogui



def shape_to_np(shape, dtype="int"):
	# initialize the list of (x, y)-coordinates
	coords = np.zeros((68, 2), dtype=dtype)
	# loop over the 68 facial landmarks and convert them
	# to a 2-tuple of (x, y)-coordinates
	for i in range(0, 68):
		coords[i] = (shape.part(i).x, shape.part(i).y)
	# return the list of (x, y)-coordinates
	return coords

#--------------------------------------------------

def eye_on_mask(mask, side):
    points = [shape[i] for i in side]
    points = np.array(points, dtype=np.int32)
    mask = cv2.fillConvexPoly(mask, points, 255)
    return mask

def contouring(thresh, mid, img, right=False):
    cnts, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_NONE)
    try:
        cnt = max(cnts, key = cv2.contourArea)
        M = cv2.moments(cnt)
        cx = int(M['m10']/M['m00'])
        cy = int(M['m01']/M['m00'])
        if right:
            cx += mid
        cv2.circle(img, (cx, cy), 4, (0, 0, 255), 2)
    except:
        pass

detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor(r'C:\Users\ashut\Downloads\shape_predictor_68_face_landmarks--281-29.dat')

left = [36, 37, 38, 39, 40, 41]
right = [42, 43, 44, 45, 46, 47]

cap = cv2.VideoCapture(0)
ret, img = cap.read()
thresh = img.copy()

cv2.namedWindow('image')
kernel = np.ones((9, 9), np.uint8)

def nothing(x):
    pass
cv2.createTrackbar('threshold', 'image', 0, 255, nothing)

#--------------------------------------------------


# Function to move the mouse cursor to the specified coordinates
def move_cursor(x, y):
    pyautogui.moveTo(x, y)


# Load the pre-trained face detector and facial landmarks predictor
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor(r'C:\Users\ashut\Downloads\shape_predictor_68_face_landmarks--281-29.dat')  
# Download the shape_predictor_68_face_landmarks.dat file

# Initialize video capture
cap = cv2.VideoCapture(0)

# Get screen size
screen_width, screen_height = pyautogui.size()
print("screen", screen_width, screen_height)

# # Create a named window
# cv2.namedWindow("Eye Tracking", cv2.WINDOW_NORMAL)

# # Set the window to full screen
# cv2.setWindowProperty("Eye Tracking", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)


while True:
    # Read frame from the camera
    ret, frame = cap.read()
    #print("yoyo", ret , frame)
    if not ret:
        break
    
    # Convert the frame to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    # Detect faces in the frame
    faces = detector(gray)
    
    for face in faces:
        # Detect facial landmarks
        landmarks = predictor(gray, face)
        
        
        #markout face shape
        shape = shape_to_np(landmarks)
#         print(shape)
#         for (x, y) in shape:
#             cv2.circle(frame, (x, y), 1, (32, 200, 0), -1)
        
        left_eye = np.array([(landmarks.part(36).x, landmarks.part(36).y),
                             (landmarks.part(37).x, landmarks.part(37).y),
                             (landmarks.part(38).x, landmarks.part(38).y),
                             (landmarks.part(39).x, landmarks.part(39).y),
                             (landmarks.part(40).x, landmarks.part(40).y),
                             (landmarks.part(41).x, landmarks.part(41).y)], dtype=np.int32)
        
        right_eye = np.array([(landmarks.part(42).x, landmarks.part(42).y),
                              (landmarks.part(43).x, landmarks.part(43).y),
                              (landmarks.part(44).x, landmarks.part(44).y),
                              (landmarks.part(45).x, landmarks.part(45).y),
                              (landmarks.part(46).x, landmarks.part(46).y),
                              (landmarks.part(47).x, landmarks.part(47).y)], dtype=np.int32)
        
        
       
        cv2.circle(frame, (int(screen_width//2), int(screen_height//2)), 4, (200, 200, 200), -1) 
        cv2.circle(frame, (300,300), 4, (200, 200, 200), -1) 
            
        for (x, y) in left_eye:
            cv2.circle(frame, (x, y), 1, (32, 200, 0), -1)
        
        for (x, y) in right_eye:
            cv2.circle(frame, (x, y), 1, (32, 200, 0), -1)
        
        # Calculate the centroid of each eye region
        left_centroid = np.mean(left_eye, axis=0)
        right_centroid = np.mean(right_eye, axis=0)
        
#         print("left_centroid" , left_centroid)
#         print("right_centroid" , right_centroid)
        
        cv2.circle(frame, tuple(left_centroid.astype(int)), 1, (0, 0, 200), -1)
        cv2.circle(frame, tuple(right_centroid.astype(int)), 1, (0, 0, 200), -1)
            
        # Determine the gaze direction based on the relative position of the eye centroids
        if left_centroid[0] < right_centroid[0]:
            gaze_x, gaze_y = left_centroid
        else:
            gaze_x, gaze_y = right_centroid
        
        # Normalize gaze coordinates
        gaze_x_norm = gaze_x / frame.shape[1]
        gaze_y_norm = gaze_y / frame.shape[0]
        
        # Map normalized gaze coordinates to screen coordinates
        screen_x = int(gaze_x_norm * screen_width)
        screen_y = int(gaze_y_norm * screen_height)
        
        # Draw a red dot at the gaze point on the frame
#         cv2.circle(frame, (int(gaze_x), int(gaze_y)), 5, (0, 0, 255), -1)
        
        # Draw a green dot at the gaze point on the screen
        cv2.circle(frame, (screen_x, screen_y), 5, (0, 255, 0), -1)
        
        # Display the screen coordinates on the frame
        cv2.putText(frame, f"Screen: ({screen_x}, {screen_y})", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        
        # Example usage
#         move_cursor(screen_x, screen_y)  # Move the cursor to (500, 500) coordinates
        
        # Display the frame
        # print(ret, frame)
        
        shape = predictor(gray, face)
        shape = shape_to_np(shape)
        mask = np.zeros(frame.shape[:2], dtype=np.uint8)
        mask = eye_on_mask(mask, left)
        mask = eye_on_mask(mask, right)
        mask = cv2.dilate(mask, kernel, 5)
        eyes = cv2.bitwise_and(frame, frame, mask=mask)
        mask = (eyes == [0, 0, 0]).all(axis=2)
        eyes[mask] = [255, 255, 255]
        mid = (shape[42][0] + shape[39][0]) // 2
        eyes_gray = cv2.cvtColor(eyes, cv2.COLOR_BGR2GRAY)
        threshold = cv2.getTrackbarPos('threshold', 'image')
        _, thresh = cv2.threshold(eyes_gray, threshold, 255, cv2.THRESH_BINARY)
        thresh = cv2.erode(thresh, None, iterations=2) #1
        thresh = cv2.dilate(thresh, None, iterations=4) #2
        thresh = cv2.medianBlur(thresh, 3) #3
        thresh = cv2.bitwise_not(thresh)
        contouring(thresh[:, 0:mid], mid, frame)
        contouring(thresh[:, mid:], mid, frame, True)
        # for (x, y) in shape[36:48]:
        #     cv2.circle(frame, (x, y), 2, (255, 0, 0), -1)
    # show the image with the face detections + facial landmarks
    cv2.imshow('eyes', frame)
    cv2.imshow("image", thresh)
        
        
        
    cv2.imshow("Eye Tracking", frame)

        # Break the loop when 'q' is pressed
    if cv2.waitKey(1) == ord('q'):
        break
        
cap.release()
cv2.destroyAllWindows()
