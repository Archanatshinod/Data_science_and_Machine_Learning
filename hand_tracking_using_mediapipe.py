from django.shortcuts import render
import cv2
import mediapipe as mp
import time


# Create your views here.

def index(request):
    return render(request,'handtrack.html')

def handtracking(request):
    cap = cv2.VideoCapture(0)
    mp_hands = mp.solutions.hands
    hands = mp_hands.Hands()
    mp_draw = mp.solutions.drawing_utils
    p_time = 0
    c_time = 0
    while True:
        success, image = cap.read()
        img_RGB = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        results = hands.process(img_RGB)
        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                for id, lm in enumerate(hand_landmarks.landmark):
                #print(id,lm)
                    h, w, c = image.shape
                    cx, cy = int(lm.x*w), int(lm.y*h)
                    #print(cx, cy)
                    
                    # drawing circle
                    if id == 4:
                        cv2.circle(image, (cx, cy), 15, (255,255,0), cv2.FILLED)
                    if id == 8:
                        cv2.circle(image, (cx, cy), 15, (255,255,0), cv2.FILLED)
                    if id == 12:
                        cv2.circle(image, (cx, cy), 15, (255,255,0), cv2.FILLED)
                    if id == 16:
                        cv2.circle(image, (cx, cy), 15, (255,255,0), cv2.FILLED)
                    if id == 20:
                        cv2.circle(image, (cx, cy), 15, (255,255,0), cv2.FILLED)
                
                mp_draw.draw_landmarks(image, hand_landmarks, mp_hands.HAND_CONNECTIONS)
                
        # # current time        
        # c_time = time.time()
        # fps = 1/(c_time-p_time)
        # p_time = c_time
        # #display fps
        # cv2.putText(image, str(int(fps)),(10,70),cv2.FONT_HERSHEY_COMPLEX, 3, (255,0,255), 3)
        
        
        cv2.imshow("Hand Tracking", image)
        if cv2.waitKey(10) & 0xFF == ord('q'):
            break
        
    cap.release()
    cv2.destroyAllWindows()
    return render(request,'handtrack.html')
