import cv2
import mediapipe as mp
import numpy as np

mp_hands = mp.solutions.hands
hands = mp_hands.Hands(min_detection_confidence=0.7,
                       min_tracking_confidence=0.7)

mp_draw = mp.solutions.drawing_utils

cap = cv2.VideoCapture(0)

canvas = np.zeros((480,640,3),np.uint8)

prev_x, prev_y = 0,0

# 7 colors + eraser
colors = [
(255,0,0),      # blue
(0,255,0),      # green
(0,0,255),      # red
(255,255,0),    # cyan
(255,0,255),    # pink
(0,255,255),    # yellow
(128,0,128),    # purple
]

eraser_color = (0,0,0)

color = colors[0]

while True:

    success, frame = cap.read()
    frame = cv2.flip(frame,1)

    rgb = cv2.cvtColor(frame,cv2.COLOR_BGR2RGB)

    result = hands.process(rgb)

    if result.multi_hand_landmarks:

        for hand_landmarks in result.multi_hand_landmarks:

            mp_draw.draw_landmarks(frame,hand_landmarks,mp_hands.HAND_CONNECTIONS)

            lm = hand_landmarks.landmark
            h,w,c = frame.shape

            x = int(lm[8].x*w)
            y = int(lm[8].y*h)

            # finger states
            index_up = lm[8].y < lm[6].y
            middle_up = lm[12].y < lm[10].y
            ring_up = lm[16].y < lm[14].y
            pinky_up = lm[20].y < lm[18].y

            # -------- ALL FINGERS = ERASER --------
            if index_up and middle_up and ring_up and pinky_up:
                draw_color = eraser_color

                if prev_x == 0 and prev_y == 0:
                    prev_x, prev_y = x, y

                cv2.line(canvas,(prev_x,prev_y),(x,y),draw_color,20)

                prev_x, prev_y = x, y

            # -------- DRAW MODE --------
            elif index_up and not middle_up:

                draw_color = color

                if prev_x == 0 and prev_y == 0:
                    prev_x, prev_y = x, y

                cv2.line(canvas,(prev_x,prev_y),(x,y),draw_color,5)

                prev_x, prev_y = x, y

            # -------- COLOR SELECT --------
            elif index_up and middle_up:

                prev_x, prev_y = 0,0

                if y < 60:

                    section = x // 80

                    if section < len(colors):
                        color = colors[section]

                # ERASER BUTTON
                if x > 560 and y < 60:
                    color = eraser_color

    else:
        prev_x, prev_y = 0,0

    frame = cv2.add(frame,canvas)

    # draw color palette
    for i,c in enumerate(colors):
        cv2.rectangle(frame,(i*80,0),((i+1)*80,60),c,-1)

    # eraser button
    cv2.rectangle(frame,(560,0),(640,60),(50,50,50),-1)
    cv2.putText(frame,"E",(585,40),
                cv2.FONT_HERSHEY_SIMPLEX,1,(255,255,255),2)

    cv2.imshow("AI Air Drawing",frame)

    key = cv2.waitKey(1)

    if key == ord('c'):
        canvas = np.zeros((480,640,3),np.uint8)

    if key == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()