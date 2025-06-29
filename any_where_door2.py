import cv2
import numpy as np
from time import sleep
import os

def background(cap):
    bg = []
    for _ in range(30):
        _, fr = cap.read()
        bg.append(fr)
    return np.median(bg, axis=0).astype(np.uint8)

def custom_diff(img1, img2):
    diff = np.zeros_like(img1)
    for c in range(3):
        diff[:,:,c] = np.abs(img1[:,:,c].astype(np.int16) - img2[:,:,c].astype(np.int16)).astype(np.uint8)
    return diff

def load_images(folder_path):
    door_images = []
    for i in range(1, 101):
        image_path = os.path.join(folder_path, f"{i:04}.png")
        image = cv2.imread(image_path, cv2.IMREAD_UNCHANGED)
        if image is not None:
            door_images.append(image)
    return door_images

def hide_transitions(bg):
    i = 1
    height, width = bg.shape[:2]
    while True:
        image = cv2.imread(f"D:/doraemons_world/any_where_door/door_images/{i:04}.png", cv2.IMREAD_UNCHANGED) #modify
        door_image = cv2.resize(image, (width, height))
        door_image_bgr = door_image[:, :, :3]
        door_mask = door_image[:, :, 3]

        door_images_inv = cv2.bitwise_not(door_mask)
        door_images_inv = cv2.resize(door_images_inv, (width, height))

        door_images_inv = door_images_inv.astype(np.uint8)

        mask = cv2.bitwise_and(bg, bg, mask=door_images_inv)
        door_image_bgr= cv2.normalize(door_image_bgr, None, 0, 255, cv2.NORM_MINMAX).astype('uint8')
        bg3 = cv2.add(mask, door_image_bgr)
        
        cv2.imshow("Frame", bg3)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            exit()
        if i < 100:
            i += 1
        else:
            cap.release()
            cv2.destroyAllWindows()
            exit()

def close_door(cap, bg, door_images):
    i = 99
    cv2.setMouseCallback("Frame", lambda event,x,y,flags,param: hide_transitions(bg) if event==cv2.EVENT_LBUTTONDOWN else None)
    
    points = np.array([(552, 185), (765, 185), (765, 550), (552, 550)], dtype=np.float32)
    
    while True:
        _, frame = cap.read()
        frame = cv2.flip(frame, 1)

        diff = custom_diff(frame, bg)
        diff_gray = cv2.cvtColor(diff, cv2.COLOR_BGR2GRAY)
        _, mask = cv2.threshold(diff_gray, 30, 255, cv2.THRESH_BINARY)
        kernel = np.ones((9,9), np.uint8)
        mask = cv2.erode(mask, kernel, iterations=1)
        mask = cv2.dilate(mask, kernel, iterations=1)
        
        rect = cv2.boundingRect(points)
        x, y, w, h = rect
        cropped_frame = frame[y:y+h, x:x+w].copy()
        cropped_mask = mask[y:y+h, x:x+w].copy()
        
        poly_mask = np.zeros((h, w), dtype=np.uint8)
        points_shifted = points - [x, y]
        cv2.fillConvexPoly(poly_mask, points_shifted.astype(np.int32), 255)
        
        cropped_frame = cv2.bitwise_and(cropped_frame, cropped_frame, mask=poly_mask)
        cropped_mask = cv2.bitwise_and(cropped_mask, cropped_mask, mask=poly_mask)
        
        mask_inv = cv2.bitwise_not(cropped_mask)
        
        bg_overlay_crop = bg_overlay[y:y+h, x:x+w]
        bg_part = cv2.bitwise_and(bg_overlay_crop, bg_overlay_crop, mask=mask_inv)
        frame_part = cv2.bitwise_and(cropped_frame, cropped_frame, mask=cropped_mask)
        blended = cv2.add(bg_part, frame_part)
        
        result = bg_overlay.copy()
        result[y:y+h, x:x+w] = blended
        
        door_image = cv2.normalize(door_images[i], None, 0, 255, cv2.NORM_MINMAX).astype('uint8')
        door_image_bgr = door_image[:, :, :3]
        door_mask = door_image[:, :, 3]
        door_mask_inv = cv2.bitwise_not(door_mask)
        
        bg_part = cv2.bitwise_and(result, result, mask=door_mask_inv)
        door_part = cv2.bitwise_and(door_image_bgr, door_image_bgr, mask=door_mask)
        result = cv2.add(bg_part, door_part)
        
        cv2.imshow("Frame", result)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
        if i > 0:
            i -= 1
        sleep(0.02)

def open_door(cap, bg, bg2, door_images): 
    i = 0
    cv2.setMouseCallback("Frame", lambda event,x,y,flags,param: close_door(cap, bg,door_images) if event==cv2.EVENT_LBUTTONDOWN else None)
    while True:
        _, frame = cap.read()
        frame = cv2.flip(frame, 1)

        diff = custom_diff(frame, bg)
        diff_gray = cv2.cvtColor(diff, cv2.COLOR_BGR2GRAY)
        _, mask = cv2.threshold(diff_gray, 30, 255, cv2.THRESH_BINARY)
        kernel = np.ones((9,9), np.uint8)
        mask = cv2.erode(mask, kernel, iterations=1)
        mask = cv2.dilate(mask, kernel, iterations=1)
        mask_inv = cv2.bitwise_not(mask)
        
        bg4 = cv2.bitwise_and(bg2[i], bg2[i], mask=mask_inv)
        bg3 = cv2.bitwise_and(frame, frame, mask=mask)
        bg3 = cv2.add(bg4, bg3)
        
        cv2.imshow("Frame", bg3)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
        if i < len(door_images) - 1:
            i += 1
        sleep(0.02)

def view_door(cap, bg, door_images, bg2):
    door_image = door_images[0]
    if door_image.dtype == 'uint16':
        door_image = cv2.normalize(door_image, None, 0, 255, cv2.NORM_MINMAX).astype('uint8')
    door_image = cv2.resize(door_image, (bg2[0].shape[1], bg2[0].shape[0]))
    door_image_bgr = door_image[:, :, :3]
    mask = door_image[:, :, 3]
    door_image_inv = cv2.bitwise_not(mask)
    bg2_initial = cv2.bitwise_and(bg, bg, mask=door_image_inv)
    bg2_initial = cv2.add(bg2_initial, door_image_bgr)
    cv2.setMouseCallback("Frame", lambda event,x,y,flags,param: open_door(cap, bg, bg2, door_images) if event==cv2.EVENT_LBUTTONDOWN else None)
    while True:
        _, frame = cap.read()
        frame = cv2.flip(frame, 1)

        diff = custom_diff(frame, bg)
        diff_gray = cv2.cvtColor(diff, cv2.COLOR_BGR2GRAY)
        _, mask = cv2.threshold(diff_gray, 30, 255, cv2.THRESH_BINARY)
        kernel = np.ones((9,9), np.uint8)
        mask = cv2.erode(mask, kernel, iterations=1)
        mask = cv2.dilate(mask, kernel, iterations=1)
        mask_inv = cv2.bitwise_not(mask)
        
        bg4 = cv2.bitwise_and(bg2_initial, bg2_initial, mask=mask_inv)
        bg3 = cv2.bitwise_and(frame, frame, mask=mask)
            
        bg3 = cv2.add(bg4, bg3)
        cv2.imshow("Frame", bg3)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1920) #modify
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080) #modify
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
bg = background(cap)
cv2.namedWindow("Frame")
door_images = load_images("D:/doraemons_world/any_where_door/door_images")
door_images = [cv2.resize(img, (width, height)) for img in door_images]
bg = cv2.flip(bg, 1)

overlay = cv2.imread('C:/Users/shiva/Downloads/download.jpeg')
overlay = cv2.resize(overlay, (width, height))
    
points = [(552, 185), (765, 185), (765, 550), (552, 550)]

pts1 = np.float32([[0, 0], [width, 0], [width, height], [0, height]])
pts2 = np.float32(points)
M = cv2.getPerspectiveTransform(pts1, pts2)
dst = cv2.warpPerspective(overlay, M, (width, height))
    
mask = np.zeros(bg.shape, dtype=np.uint8)
roi_corners = np.int32(points)
cv2.fillConvexPoly(mask, roi_corners, (255, 255, 255))
mask_inv = cv2.bitwise_not(mask)
img1_bg = cv2.bitwise_and(bg, mask_inv)
img2_fg = cv2.bitwise_and(dst, mask)
bg_overlay = cv2.add(img1_bg, img2_fg)

bg2 = []
for door_image in door_images:
    door_image = cv2.normalize(door_image, None, 0, 255, cv2.NORM_MINMAX).astype('uint8')
    door_image_bgr = door_image[:, :, :3]
    mask = door_image[:, :, 3]
    door_images_inv = cv2.bitwise_not(mask)
    mask = cv2.bitwise_and(bg_overlay, bg_overlay, mask=door_images_inv)
    bg2.append(cv2.add(mask, door_image_bgr))

view_door(cap, bg, door_images, bg2)
#close_door(cap,bg,door_images)