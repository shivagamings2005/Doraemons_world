import cv2
import numpy as np
def background(cap):
    bg=[]
    for i in range(30):
        ret,fr=cap.read()
        if ret:
            bg.append(fr)
        else:
            print("error in frame",i+1)
    return np.median(bg,axis=0).astype(np.uint8)
def make_mask(fr,l,u):
    hsv=cv2.cvtColor(fr,cv2.COLOR_BGR2HSV)
    mask=cv2.inRange(hsv,l,u)
    mask=cv2.morphologyEx(mask,cv2.MORPH_OPEN,np.ones((3,3),np.uint8),iterations=2)
    mask=cv2.morphologyEx(mask,cv2.MORPH_DILATE,np.ones((3,3),np.uint8),iterations=1)
    return mask
def make_cape(fr,mask,bgr):
    inv_mask=cv2.bitwise_not(mask)
    fg=cv2.bitwise_and(fr,fr,mask=inv_mask)
    bg=cv2.bitwise_and(bgr,bgr,mask=mask)
    return cv2.add(fg,bg)
cap=cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)
backgrd=background(cap)
lower_color=np.array([80, 50, 135]) #blue
upper_color=np.array([150, 255, 255])

#lower_color=np.array([35, 100, 100]) #green
#upper_color=np.array([85, 255, 255])
while True:
    _,fr=cap.read()
    mask=make_mask(fr,lower_color,upper_color)
    final=make_cape(fr,mask,backgrd)
    final = cv2.flip(final, 1)
    cv2.imshow("Invisible cape",final)
    if cv2.waitKey(1) & 0xFF==ord("q"):
        break
cap.release()
cv2.destroyAllWindows()
