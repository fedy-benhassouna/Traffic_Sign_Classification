import os
import numpy as np
import cv2
import pickle

#############################################
frameWidth = 640         # DISPLAY RESOLUTION
frameHeight = 480
threshold = 0.75         # PROBABILITY THRESHOLD
font = cv2.FONT_HERSHEY_SIMPLEX
test_data_folder = "test_data"
output_image_size = (600, 400)  # Increase the output image size
font_scale = 0.5  # Smaller font scale
font_thickness = 1  # Adjust the thickness of the text
##############################################

# IMPORT THE TRAINED MODEL
pickle_in = open("model_trained.p", "rb")  # rb = READ BYTE
model = pickle.load(pickle_in)


def grayscale(img):
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return img


def equalize(img):
    img = cv2.equalizeHist(img)
    return img


def preprocessing(img):
    img = grayscale(img)
    img = equalize(img)
    img = img / 255
    return img


def getClassName(classNo):
    if classNo == 0:
        return 'Speed Limit 20 km/h'
    elif classNo == 1:
        return 'Speed Limit 30 km/h'
    elif classNo == 2:
        return 'Speed Limit 50 km/h'
    elif classNo == 3:
        return 'Speed Limit 60 km/h'
    elif classNo == 4:
        return 'Speed Limit 70 km/h'
    elif classNo == 5:
        return 'Speed Limit 80 km/h'
    elif classNo == 6:
        return 'End of Speed Limit 80 km/h'
    elif classNo == 7:
        return 'Speed Limit 100 km/h'
    elif classNo == 8:
        return 'Speed Limit 120 km/h'
    elif classNo == 9:
        return 'No passing'
    elif classNo == 10:
        return 'No passing for vehicles over 3.5 metric tons'
    elif classNo == 11:
        return 'Right-of-way at the next intersection'
    elif classNo == 12:
        return 'Priority road'
    elif classNo == 13:
        return 'Yield'
    elif classNo == 14:
        return 'Stop'
    elif classNo == 15:
        return 'No vehicles'
    elif classNo == 16:
        return 'Vehicles over 3.5 metric tons prohibited'
    elif classNo == 17:
        return 'No entry'
    elif classNo == 18:
        return 'General caution'
    elif classNo == 19:
        return 'Dangerous curve to the left'
    elif classNo == 20:
        return 'Dangerous curve to the right'
    elif classNo == 21:
        return 'Double curve'
    elif classNo == 22:
        return 'Bumpy road'
    elif classNo == 23:
        return 'Slippery road'
    elif classNo == 24:
        return 'Road narrows on the right'
    elif classNo == 25:
        return 'Road work'
    elif classNo == 26:
        return 'Traffic signals'
    elif classNo == 27:
        return 'Pedestrians'
    elif classNo == 28:
        return 'Children crossing'
    elif classNo == 29:
        return 'Bicycles crossing'
    elif classNo == 30:
        return 'Beware of ice/snow'
    elif classNo == 31:
        return 'Wild animals crossing'
    elif classNo == 32:
        return 'End of all speed and passing limits'
    elif classNo == 33:
        return 'Turn right ahead'
    elif classNo == 34:
        return 'Turn left ahead'
    elif classNo == 35:
        return 'Ahead only'
    elif classNo == 36:
        return 'Go straight or right'
    elif classNo == 37:
        return 'Go straight or left'
    elif classNo == 38:
        return 'Keep right'
    elif classNo == 39:
        return 'Keep left'
    elif classNo == 40:
        return 'Roundabout mandatory'
    elif classNo == 41:
        return 'End of no passing'
    elif classNo == 42:
        return 'End of no passing by vehicles over 3.5 metric tons'


for filename in os.listdir(test_data_folder):
    if filename.endswith(".jpg") or filename.endswith(".png"):
        imgOriginal = cv2.imread(os.path.join(test_data_folder, filename))

        if imgOriginal is not None:
            img = np.asarray(imgOriginal)
            img = cv2.resize(img, (32, 32))
            img = preprocessing(img)
            img = img.reshape(1, 32, 32, 1)

            # PREDICT IMAGE
            predictions = model.predict(img)
            classIndex = np.argmax(predictions)
            probabilityValue = np.amax(predictions)

            # Resize the original image for display
            imgOriginal = cv2.resize(imgOriginal, output_image_size)

            # Add text to the image
            cv2.putText(imgOriginal, "CLASS: ", (20, 35), font, font_scale, (0, 0, 255), font_thickness, cv2.LINE_AA)
            cv2.putText(imgOriginal, "PROBABILITY: ", (20, 75), font, font_scale, (0, 0, 255), font_thickness,
                        cv2.LINE_AA)

            if probabilityValue > threshold:
                cv2.putText(imgOriginal, str(classIndex) + " " + str(getClassName(classIndex)), (120, 35), font,
                            font_scale, (0, 0, 255), font_thickness, cv2.LINE_AA)
                cv2.putText(imgOriginal, str(round(probabilityValue * 100, 2)) + "%", (180, 75), font, font_scale,
                            (0, 0, 255), font_thickness, cv2.LINE_AA)

            # Display the image
            cv2.imshow("Result", imgOriginal)
            cv2.waitKey(0)  # Wait for a key press to move to the next image

cv2.destroyAllWindows()
