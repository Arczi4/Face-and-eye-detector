import cv2

face_classifier = cv2.CascadeClassifier('Haarcascades/haarcascade_frontalface_default.xml')
eye_classifier = cv2.CascadeClassifier('Haarcascades\haarcascade_eye.xml')

img = cv2.imread('test.jpg')
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

face = face_classifier.detectMultiScale(gray, 1.3, 5)
print(face)
if len(face) == 0:
    print("Face not found")

for (x, y, width, height) in face:
    # Face rectangle
    cv2.rectangle(img, (x, y), (x + width, y + height), (0, 255, 0), 2)
    
    # Region of interest
    face_roi_gray = gray[y: y+height, x: x+width]
    face_roi_color = img[y: y+height, x: x+width]

    eyes = eye_classifier.detectMultiScale(face_roi_gray)
    # Eyes rectangle
    for ex, ey, e_width, e_height in eyes:
        cv2.rectangle(face_roi_color, (ex, ey), (ex + e_width, ey + e_height), (0, 255, 0), 2)
    cv2.imshow('Face Detection', img)
    cv2.waitKey(0)

cv2.destroyAllWindows()