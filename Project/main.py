import cv2
import numpy as np

min_contour_width = 50 # 110 for first video and 50 for second video
min_contour_height = 50 # 110 for first video and 50 for second video
offset = 10
line_height = 200 # 600 for first video and 200 for second video
max_age = 5 
tracked_objects = {}
vehicles = 0
next_id = 0
distance_threshold = 70


def centroids(x, y, w, h):
    cx = x + w // 2
    cy = y + h // 2
    return cx, cy


cap = cv2.VideoCapture('Video2.mp4')
cap.set(3, 1920)
cap.set(4, 1080)

if cap.isOpened():
    ret, frame1 = cap.read()
else:
    ret = False
ret, frame1 = cap.read()
ret, frame2 = cap.read()

while ret:
    d = cv2.absdiff(frame1, frame2)
    grey = cv2.cvtColor(d, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(grey, (5, 5), 0)
    ret, th = cv2.threshold(blur, 20, 255, cv2.THRESH_BINARY)
    dilated = cv2.dilate(th, np.ones((3, 3)))
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2, 2))
    closing = cv2.morphologyEx(dilated, cv2.MORPH_CLOSE, kernel)
    contours, h = cv2.findContours(closing, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    new_tracks = []
    for (i, c) in enumerate(contours):
        (x, y, w, h) = cv2.boundingRect(c)
        contour_valid = (w >= min_contour_width) and (h >= min_contour_height)
        if not contour_valid:
            continue
        centroid = centroids(x, y, w, h)
        new_tracks.append(centroid)
        cv2.rectangle(frame1, (x, y), (x + w, y + h), (255, 0, 0), 2)
        cv2.circle(frame1, centroid, 5, (0, 255, 0), -1)

    # Update tracked objects
    for obj_id, track in list(tracked_objects.items()):
        match_found = False
        for centroid in new_tracks:
            if np.linalg.norm(
                    np.array(track['centroid']) - np.array(centroid)) < distance_threshold:  # Distance threshold
                tracked_objects[obj_id]['centroid'] = centroid
                tracked_objects[obj_id]['age'] = 0
                match_found = True
                new_tracks.remove(centroid)
                break
        if not match_found:
            tracked_objects[obj_id]['age'] += 1
        if tracked_objects[obj_id]['age'] > max_age:
            del tracked_objects[obj_id]

    # Add new tracks
    for centroid in new_tracks:
        tracked_objects[next_id] = {'centroid': centroid, 'age': 0}
        next_id += 1

    # Check for line crossing
    for obj_id, track in list(tracked_objects.items()):
        cx, cy = track['centroid']
        if track.get('counted', False):
            continue
        if cy < (line_height + offset) and cy > (line_height - offset):
            vehicles += 1
            tracked_objects[obj_id]['counted'] = True

    cv2.putText(frame1, "Total Vehicles Detected: " + str(vehicles), (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 170, 0),
                2)
    cv2.line(frame1, (0, line_height), (1920, line_height), (0, 255, 0), 2)
    cv2.imshow("Vehicle Detection", frame1)

    if cv2.waitKey(25) == 27:  # 100 ms delay to slow down the video
        break

    frame1 = frame2
    ret, frame2 = cap.read()

cv2.destroyAllWindows()
cap.release()
