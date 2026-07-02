#!/usr/bin/env python3
"""
Visualize the needle detection for gauge 3 across all images.
Save cropped images showing what the algorithm sees.
"""
import cv2
import numpy as np
from gauge_reader import detect_gauges, read_needle_angle

def visualize_gauge3(img_path, idx):
    img = cv2.imread(img_path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gauges = detect_gauges(gray, n_gauges=3)
    
    if len(gauges) < 3:
        print(f"Image {idx}: only {len(gauges)} gauges found")
        return
    
    cx, cy, r = gauges[2]  # gauge 3
    h, w = img.shape[:2]
    y0, y1 = max(0, int(cy - r * 1.3)), min(h, int(cy + r * 1.3))
    x0, x1 = max(0, int(cx - r * 1.3)), min(w, int(cx + r * 1.3))
    
    crop = img[y0:y1, x0:x1].copy()
    lcx, lcy = cx - x0, cy - y0
    
    # Get the angle
    angle = read_needle_angle(img, cx, cy, r)
    
    # Draw the detected needle direction
    if angle is not None:
        tip_x = lcx + r * 0.6 * np.cos(np.radians(angle))
        tip_y = lcy + r * 0.6 * np.sin(np.radians(angle))
        cv2.arrowedLine(crop, (int(lcx), int(lcy)), (int(tip_x), int(tip_y)),
                        (0, 0, 255), 3, tipLength=0.2)
        cv2.putText(crop, f"angle={angle:.1f}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
    
    cv2.imwrite(f"gauge3_img{idx}.jpg", crop)
    print(f"Image {idx}: gauge3 angle = {angle}")

for i in range(1, 6):
    visualize_gauge3(f"dataset/{i}.jpg", i)
