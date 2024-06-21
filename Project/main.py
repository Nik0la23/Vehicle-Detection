# import cv2
# import numpy as np
#
# min_contour_width = 40
# min_contour_height = 40
# offset = 10
# line_height = 550
# matches = []
# vehicles = 0
#
#
# def get_centrolid(x, y, w, h):
#     x1 = int(w / 2)
#     y1 = int(h / 2)
#
#     cx = x + x1
#     cy = y + y1
#     return cx, cy
#
#
# cap = cv2.VideoCapture('Video.mp4')
# cap.set(3, 1920)
# cap.set(4, 1080)
#
# if cap.isOpened():
#     ret, frame1 = cap.read()
# else:
#     ret = False
# ret, frame1 = cap.read()
# ret, frame2 = cap.read()
# while ret:
#     d = cv2.absdiff(frame1, frame2)
#     grey = cv2.cvtColor(d, cv2.COLOR_BGR2GRAY)
#
#     blur = cv2.GaussianBlur(grey, (5, 5), 0)
#
#     ret, th = cv2.threshold(blur, 10, 255, cv2.THRESH_BINARY)
#     dilated = cv2.dilate(th, np.ones((3, 3)))
#     kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2, 2))
#
#     closing = cv2.morphologyEx(dilated, cv2.MORPH_CLOSE, kernel)
#     contours, h = cv2.findContours(
#         closing, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
#     for (i, c) in enumerate(contours):
#         (x, y, w, h) = cv2.boundingRect(c)
#         contour_valid = (w >= min_contour_width) and (
#                 h >= min_contour_height)
#
#         if not contour_valid:
#             continue
#         cv2.rectangle(frame1, (x - 10, y - 10), (x + w + 10, y + h + 10), (255, 0, 0), 2)
#
#         cv2.line(frame1, (0, line_height), (1200, line_height), (0, 255, 0), 2)
#         centrolid = get_centrolid(x, y, w, h)
#         matches.append(centrolid)
#         cv2.circle(frame1, centrolid, 5, (0, 255, 0), -1)
#         cx, cy = get_centrolid(x, y, w, h)
#         for (x, y) in matches:
#             if y < (line_height + offset) and y > (line_height - offset):
#                 vehicles = vehicles + 1
#                 matches.remove((x, y))
#
#     cv2.putText(frame1, "Total Vehicle Detected: " + str(vehicles), (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 1,
#                 (0, 170, 0), 2)
#
#     cv2.imshow("Vehicle Detection", frame1)
#     if cv2.waitKey(50) == 27:
#         break
#     frame1 = frame2
#     ret, frame2 = cap.read()
#
# cv2.destroyAllWindows()
# cap.release()

# import cv2
# import numpy as np
# from collections import deque
#
# min_contour_width = 40
# min_contour_height = 40
# offset = 10
# line_height = 220
# max_age = 10  # Maximum number of frames to keep a track without detection
# tracked_objects = {}
# vehicles = 0
# next_id = 0
#
#
# def get_centroid(x, y, w, h):
#     cx = x + w // 2
#     cy = y + h // 2
#     return cx, cy
#
#
# cap = cv2.VideoCapture('Video2.mp4')
# cap.set(3, 1920)
# cap.set(4, 1080)
#
# if cap.isOpened():
#     ret, frame1 = cap.read()
# else:
#     ret = False
# ret, frame1 = cap.read()
# ret, frame2 = cap.read()
#
# while ret:
#     d = cv2.absdiff(frame1, frame2)
#     grey = cv2.cvtColor(d, cv2.COLOR_BGR2GRAY)
#     blur = cv2.GaussianBlur(grey, (3, 3), 0)
#     ret, th = cv2.threshold(blur, 20, 255, cv2.THRESH_BINARY)
#     dilated = cv2.dilate(th, np.ones((3, 3)))
#     kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
#     closing = cv2.morphologyEx(dilated, cv2.MORPH_CLOSE, kernel)
#     contours, h = cv2.findContours(closing, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
#
#     new_tracks = []
#     for (i, c) in enumerate(contours):
#         (x, y, w, h) = cv2.boundingRect(c)
#         contour_valid = (w >= min_contour_width) and (h >= min_contour_height)
#         if not contour_valid:
#             continue
#         centroid = get_centroid(x, y, w, h)
#         new_tracks.append(centroid)
#         cv2.rectangle(frame1, (x + 5, y + 5), (x + w - 5, y + h - 5), (255, 0, 0), 2)
#         cv2.circle(frame1, centroid, 10, (0, 255, 0), -1)
#
#     # Update tracked objects
#     for obj_id, track in list(tracked_objects.items()):
#         match_found = False
#         for centroid in new_tracks:
#             if np.linalg.norm(np.array(track['centroid']) - np.array(centroid)) < 30:  # Distance threshold
#                 tracked_objects[obj_id]['centroid'] = centroid
#                 tracked_objects[obj_id]['age'] = 0
#                 match_found = True
#                 new_tracks.remove(centroid)
#                 break
#         if not match_found:
#             tracked_objects[obj_id]['age'] += 1
#         if tracked_objects[obj_id]['age'] > max_age:
#             del tracked_objects[obj_id]
#
#     # Add new tracks
#     for centroid in new_tracks:
#         tracked_objects[next_id] = {'centroid': centroid, 'age': 0}
#         next_id += 1
#
#     # Check for line crossing
#     for obj_id, track in list(tracked_objects.items()):
#         cx, cy = track['centroid']
#         if cy < (line_height + offset) and cy > (line_height - offset) and not track.get('counted', False):
#             vehicles += 1
#             tracked_objects[obj_id]['counted'] = True
#
#     cv2.putText(frame1, "Total Vehicle Detected: " + str(vehicles), (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 170, 0),
#                 2)
#     cv2.line(frame1, (0, line_height), (1920, line_height), (0, 255, 0), 2)
#     cv2.imshow("Vehicle Detection", frame1)
#
#     if cv2.waitKey(10) == 27:  # 100 ms delay to slow down the video
#         break
#
#     frame1 = frame2
#     ret, frame2 = cap.read()
#
# cv2.destroyAllWindows()
# cap.release()


import cv2
import numpy as np

min_contour_width = 50 # 110 for first video and 50 for second video
min_contour_height = 50 # 110 for first video and 50 for second video
offset = 10
line_height = 200 # 600 for first video and 200 for second video
max_age = 5  # Maximum number of frames to keep a track without detection
tracked_objects = {}
vehicles = 0
next_id = 0
distance_threshold = 70


def centroids(x, y, w, h):
    cx = x + w // 2
    cy = y + h // 2
    return cx, cy


cap = cv2.VideoCapture('Video2.mp4') #Video2
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

    cv2.putText(frame1, "Total Vehicle Detected: " + str(vehicles), (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 170, 0),
                2)
    cv2.line(frame1, (0, line_height), (1920, line_height), (0, 255, 0), 2)
    cv2.imshow("Vehicle Detection", frame1)

    if cv2.waitKey(25) == 27:  # 100 ms delay to slow down the video
        break

    frame1 = frame2
    ret, frame2 = cap.read()

cv2.destroyAllWindows()
cap.release()
