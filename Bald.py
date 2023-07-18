import cv2

# Load the cascade for detecting faces
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Load the image
image = cv2.imread("./images/Rock.jpg")

# Convert the image to grayscale
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# Detect faces in the image
faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

# Iterate over each detected face
for (x, y, w, h) in faces:
    # Extract the region of interest (ROI) for the face
    face_roi = image[y:y+h, x:x+w]

    # Apply hair region detection algorithm (example: Canny edge detection)
    edges = cv2.Canny(face_roi, 50, 150)

    # Count the number of hair pixels
    hair_pixel_count = cv2.countNonZero(edges)

    # Define a threshold to determine if the person is bald or not
    threshold = 5000  # Adjust this value based on your needs

    # Determine if the person is bald or not based on the hair pixel count
    is_bald = hair_pixel_count < threshold

    # Print the result
    if is_bald:
        print("The person is bald.")
    else:
        print("The person is not bald.")

    # Draw a rectangle around the face
    cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)

# Display the image with detected faces
cv2.imshow('Bald Detection', image)
cv2.waitKey(0)
cv2.destroyAllWindows()
