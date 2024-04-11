import cv2
import numpy as np
import os
import HandTrackingModule as htm
import keyboard
import pygame
import time
import pyttsx3
import re
from tensorflow.keras.models import load_model

engine = pyttsx3.init()

# Color Attributes
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
RED = (0, 0, 255)
YELLOW =  (0, 255, 255)
GREEN = (0, 255, 0)
BACKGROUND = (255, 255, 255)
FORGROUND = (0, 255, 0)
BORDER = (255, 0, 0)
lastdrawColor = (0, 0, 1)
drawColor = (0, 0, 255)
BOUNDRYINC = 5

color_dict = {
    ord('c'): (192, 192, 192),  # Clear (Gray color)
    ord('b'): (135, 206, 235),  # Skin
    ord('g'): (0, 255, 0),       # Green
    ord('r'): (0, 0, 255),       # Red
    ord('y'): (0, 255, 255),     # Yellow
    ord('w'): (255, 255, 255),   # White
    ord('e'): (0, 0, 0)          # Eraser (Black color)
}

# CV2 Attributes
cap = cv2.VideoCapture(0)
width, height = 1280, 720
cap.set(3, width)
cap.set(4, height)
imgCanvas = np.zeros((height, width, 3), np.uint8)

# PyGame Attributes
pygame.init()
FONT = pygame.font.SysFont('freesansbold.tff', 18)
DISPLAYSURF = pygame.display.set_mode((width, height), flags=pygame.HIDDEN)
pygame.display.set_caption("Digit Board")
number_xcord = []
number_ycord = []

# Prediction Model Attributes
label = ""
recognized_characters = ""
space_added = False
PREDICT = "off"
AlphaMODEL = load_model(r"bModel.h5")
NumMODEL = load_model(r"bestmodel.h5")
AlphaLABELS = {0: 'a', 1: 'b', 2: 'c', 3: 'd', 4: 'e', 5: 'f', 6: 'g', 7: 'h', 8: 'i', 9: 'j',
               10: 'k', 11: 'l', 12: 'm', 13: 'n', 14: 'o', 15: 'p', 16: 'q', 17: 'r', 18: 's', 19: 't',
               20: 'u', 21: 'v', 22: 'w', 23: 'x', 24: 'y', 25: 'z', 26: ''}
NumLABELS = {0: '0', 1: '1', 2: '2', 3: '3', 4: '4', 5: '5', 6: '6', 7: '7', 8: '8', 9: '9'}
rect_min_x, rect_max_x = 0, 0
rect_min_y, rect_max_y = 0, 0

# Assuming hand_detected is set to True when hand is detected for the first time
# if hand_detected:
#     engine.say("Hand detected!")
#     engine.runAndWait()
#     hand_detected = True

# HandDetection Attributes
detector = htm.handDetector(detectionCon=0.85)
x1, y1 = 0, 0
xp, yp = 0, 0
brushThickness = 15
eraserThickness = 30
modeValue = "OFF"
modeColor = RED

# hand_cascade=cv2.CascadeClassifier(r'C:/Users/Lenovo/Desktop/Air_writing_models/hand.xml')
hand_detected=False

while True:
    SUCCESS, img = cap.read()
    img = cv2.flip(img, 1)

    gray=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    # hands=hand_cascade.detectMultiScale(gray,scaleFactor=1.3,minNeighbors=5)

    
        
    img = detector.findHands(img)
    lmList = detector.findPosition(img, draw=False)

    cv2.putText(img, "Press A for Alphabet Recognition Mode ", (0, 145), 3, 0.5, (255, 255, 0), 1, cv2.LINE_AA)
    cv2.putText(img, "Press N for Digit Recognition Mode ", (0, 162), 3, 0.5, (255, 255, 0), 1, cv2.LINE_AA)
    cv2.putText(img, "Press O for Turn Off Recognition Mode ", (0, 179), 3, 0.5, (255, 255, 0), 1, cv2.LINE_AA)
    cv2.putText(img, f'{"RECOGNITION IS "}{modeValue}', (0, 196), 3, 0.5, modeColor, 1, cv2.LINE_AA)

    if keyboard.is_pressed('a'):
        if PREDICT != "alpha":
            PREDICT = "alpha"
            modeValue, modeColor = "ALPHABETS", GREEN
            engine.say("alphabet recognition mode is on")
            engine.runAndWait()

    if keyboard.is_pressed('n'):
        if PREDICT != "num":
            PREDICT = "num"
            modeValue, modeColor = "NUMBER", YELLOW
            engine.say("number recognition mode is on")
            engine.runAndWait()

    if keyboard.is_pressed('o'):
        if PREDICT != "off":
            PREDICT = "off"
            modeValue, modeColor = "OFF", RED
            engine.say("recognition mode is off")
            engine.runAndWait()

        xp, yp = 0, 0
        label = ""
        rect_min_x, rect_max_x = 0, 0
        rect_min_y, rect_max_y = 0, 0
        number_xcord = []
        number_ycord = []
        time.sleep(0.5)

    if len(lmList) > 0:


        x1, y1 = lmList[8][1:]
        x2, y2 = lmList[12][1:]

        if hand_detected==False:
            engine.say('hand  is detected')
            engine.runAndWait()
            hand_detected=True

        fingers = detector.fingersUp()

        if fingers[1] and fingers[2]:

            number_xcord = sorted(number_xcord)
            number_ycord = sorted(number_ycord)

            if len(number_xcord) > 0 and len(number_ycord) > 0 and PREDICT != "off":
                if drawColor != (0, 0, 0) and lastdrawColor != (0, 0, 0):
                    rect_min_x, rect_max_x = max(number_xcord[0] - BOUNDRYINC, 0), min(width, number_xcord[-1] + BOUNDRYINC)
                    rect_min_y, rect_max_y = max(0, number_ycord[0] - BOUNDRYINC), min(number_ycord[-1] + BOUNDRYINC, height)
                    number_xcord = []
                    number_ycord = []

                    img_arr = np.array(pygame.PixelArray(DISPLAYSURF))[rect_min_x:rect_max_x, rect_min_y:rect_max_y].T.astype(np.float32)

                    cv2.rectangle(imgCanvas, (rect_min_x, rect_min_y), (rect_max_x, rect_max_y), BORDER, 3)
                    image = cv2.resize(img_arr, (28, 28))
                    image = np.pad(image, (10, 10), 'constant', constant_values=0)
                    image = cv2.resize(image, (28, 28)) / 255
                    
                
                    

                    if PREDICT == "alpha":
                        label = str(AlphaLABELS[np.argmax(AlphaMODEL.predict(image.reshape(1, 28, 28, 1)))])
                        engine.say(label)
                        engine.runAndWait()
                        recognized_characters= recognized_characters+label
                    if PREDICT == "num":
                        label = str(NumLABELS[np.argmax(NumMODEL.predict(image.reshape(1, 28, 28, 1)))])
                        engine.say(label)
                        engine.runAndWait()
                        recognized_characters += label
                    pygame.draw.rect(DISPLAYSURF, BLACK, (0, 0, width, height))

                    cv2.rectangle(imgCanvas, (rect_min_x + 50, rect_min_y - 30), (rect_min_x, rect_min_y), BACKGROUND, -1)
                    cv2.putText(imgCanvas, label, (rect_min_x, rect_min_y - 5), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0, 255), 2, cv2.LINE_AA)
                    # # Calculate distances between fingers
                    # distance_pinky_index = ((lmList[20][1] - lmList[8][1]) ** 2 + (lmList[20][2] - lmList[8][2]) ** 2) ** 0.5
                    # distance_thumb_index = ((lmList[4][1] - lmList[8][1]) ** 2 + (lmList[4][2] - lmList[8][2]) ** 2) ** 0.5

                    # # If distance between pinky and index finger is greater than distance between thumb and index finger, add a space
                    # if distance_pinky_index > distance_thumb_index:
                    #     recognized_characters += " "
                    # Count the number of fingers raised (assuming you're using 21 landmarks)
                        # Inside the loop where you're checking for finger positions
    # Assuming you're using 21 landmarks
                    # Check for pinching gesture to insert a space
                    # thumb_tip = lmList[4]  # Thumb tip landmark
                    # index_tip = lmList[8]  # Index finger tip landmark
                    # if thumb_tip[0] and index_tip[0] and abs(thumb_tip[0] - index_tip[0]) < 20 and not space_added:
                    #     recognized_characters += " "
                    #     space_added = True  # Set flag  to True after adding space to avoid adding multiple spaces

                        
                    # Inside the part where you're checking for finger positions
                    # Assuming you're using 21 landmarks

                    
                else:
                    number_xcord = []
                    number_ycord = []

            xp, yp = 0, 0

            cv2.rectangle(img, (x1, y1 - 25), (x2, y2 + 25), drawColor, cv2.FILLED)

        elif fingers[1] and fingers[2] == False:

            number_xcord.append(x1)
            number_ycord.append(y1)

            cv2.circle(img, (x1, y1 - 15), 15, drawColor, cv2.FILLED)
            if xp == 0 and yp == 0:
                xp, yp = x1, y1

            if drawColor == (0, 0, 0):
                cv2.line(img, (xp, yp), (x1, y1), drawColor, eraserThickness)
                cv2.line(imgCanvas, (xp, yp), (x1, y1), drawColor, eraserThickness)
            else:
                cv2.line(img, (xp, yp), (x1, y1), drawColor, brushThickness)
                cv2.line(imgCanvas, (xp, yp), (x1, y1), drawColor, brushThickness)
                pygame.draw.line(DISPLAYSURF, WHITE, (xp, yp), (x1, y1), brushThickness)
            xp, yp = x1, y1
        else:
            xp, yp = 0, 0

    # Navigation bar logic
    if y1 < 50:
        selected_option_index = int(x1 / (width / len(color_dict)))
        if selected_option_index < len(color_dict):
            selected_option_key = list(color_dict.keys())[selected_option_index]
            if selected_option_key == ord('e'):
                drawColor = (0, 0, 0)
            elif selected_option_key == ord('c'):
                imgCanvas = np.zeros((height, width, 3), np.uint8)  # Clear the canvas
                recognized_characters = ""
            else:
                drawColor = color_dict[selected_option_key]

    # Calculate the total width of all color rectangles
            total_rect_width = width - 60  # Width minus space for close button

        # Calculate the width of each color rectangle
        color_rect_width = total_rect_width / len(color_dict)

    # # Draw color rectangles
    for i, (key, color) in enumerate(color_dict.items()):
        rect_start_x = i * color_rect_width + 10  # Adding 10 pixels of space between rectangles
        rect_end_x = (i + 1) * color_rect_width - 10  # Subtracting 10 pixels to maintain space
        cv2.rectangle(img, (int(rect_start_x), 0), (int(rect_end_x), 50), color, 2)  # Draw border only
        color_name = 'Eraser' if color == (0, 0, 0) else 'White' if color == (255, 255, 255) else 'Red' if color == (0, 0, 255) else 'Yellow' if color == (0, 255, 255) else 'Green' if color == (0, 255, 0) else 'Skin' if color == (135, 206, 235) else 'Clear all'
        text_color = color
        text_x = int(rect_start_x) + 10  # Example x-coordinate
        text_y = 30  
        cv2.putText(img, color_name, (text_x, text_y), cv2.FONT_HERSHEY_TRIPLEX, 0.75, text_color, 1)  # Draw text  # Draw text

        # Draw a gradient rectangle
        
        gradient_rect = np.zeros((50, int(rect_end_x - rect_start_x), 3), np.uint8)
        color_tuple = tuple(reversed(color))  # Reverse color for gradient effect
        for j in range(int(rect_end_x - rect_start_x)): 
            gradient_rect[:, j] = tuple(int(color_channel * (1 - j / (rect_end_x - rect_start_x))) for color_channel in color_tuple)

        # Adjust the placement to ensure the dimensions match precisely
        #img[int(y1):int(y1) + 50, int(rect_start_x):int(rect_end_x)] = gradient_rect[:, :int(rect_end_x - rect_start_x)]
        
        # Add border
        # cv2.rectangle(img, (int(rect_start_x), int(y1)), (int(rect_end_x), int(y1) + 50), (255, 255, 255), 2)
    

        font_scale = 0.8  # Example font scale
        font_thickness = 2  # Example font thickness
        # cv2.putText(img, color_name, (int(rect_start_x) + 10, int(y1) + 30), cv2.FONT_HERSHEY_TRIPLEX, font_scale, text_color, font_thickness)  # Draw text

        # Add border
        # cv2.rectangle(img, (int(rect_start_x), int(y1)), (int(rect_end_x), int(y1) + 50), (255, 255, 255), 2)
     # If add_space is True, add a space character to recognized_characters and set add_space to False
        # if add_space:
            
        #     add_space = False

        # Rest of your code...

        # Check if the space key is pressed
        
    # Draw close button
    close_rect_start_x = width - 60  # Distance from other rectangles
    close_rect_end_x = width - 10
    cv2.rectangle(img, (int(close_rect_start_x), 0), (int(close_rect_end_x), 50), (0, 0, 0), -1)  # Draw filled rectangle for close button
    cv2.putText(img, "X", (int(close_rect_start_x) + 10, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)  # Draw 'X' in white color
    if x1 > width - 50 and y1 < 50:
        break

    imgGray = cv2.cvtColor(imgCanvas, cv2.COLOR_BGR2GRAY)
    _, imgInv = cv2.threshold(imgGray, 50, 255, cv2.THRESH_BINARY_INV)
    imgInv = cv2.cvtColor(imgInv, cv2.COLOR_GRAY2BGR)
    img = cv2.bitwise_and(img, imgInv)
    img = cv2.bitwise_or(img, imgCanvas)

    newline_added = False

    # Calculate the position and size of the space button
    space_button_width = 90 # Width of the space button
    space_button_height = 50  # Height of the space button
    space_button_x = 10  # X-coordinate of the space button
    space_button_y = height // 3 - space_button_height // 2  # Y-coordinate of the space button

    # Draw the space button
    cv2.rectangle(img, (space_button_x, space_button_y), (space_button_x + space_button_width, space_button_y + space_button_height), WHITE, -1)
    cv2.rectangle(img, (space_button_x, space_button_y), (space_button_x + space_button_width, space_button_y + space_button_height), BLACK, 2)
    cv2.putText(img, "SPACE", (space_button_x + 10, space_button_y + 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, BLACK, 2)

    # Check if the hand is clicking on the space button
    if space_button_x < x1 < space_button_x + space_button_width and space_button_y < y1 < space_button_y + space_button_height:
        recognized_characters += "\n"
        newline_added = True

    if keyboard.is_pressed('up'):
        brushThickness += 1
    elif keyboard.is_pressed('down'):
        brushThickness = max(1, brushThickness - 1)

    # Get the width and height of the image
    img_height, img_width = img.shape[:2]

    # Define the position for the text
    text_position = (img_width - 200, img_height - 10)

    cv2.putText(img, f'Brush Thickness: {brushThickness}', text_position, cv2.FONT_HERSHEY_TRIPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
     
    c=1
    if keyboard.is_pressed('s'):
        filename = f'canvas_image{c}.png'
        cv2.imwrite(filename, imgCanvas)
        print(f"Canvas saved as {filename}")
        c=c+1

    if keyboard.is_pressed('space'):
            recognized_characters += "\n"

    pygame.display.update()
    cv2.imshow("Image", img)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# After the while loop
recognized_characters = re.sub(r'\s+', '\n', recognized_characters)
print("Recognized characters:",'\n',recognized_characters)

cap.release()
cv2.destroyAllWindows()
