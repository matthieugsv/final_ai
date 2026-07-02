import argparse
import sys
import numpy as np
import cv2

# Calibration constants for the 0-1 MPa gauge scale
ANGLE_ZERO_DEG = 183.0
ANGLE_ONE_DEG = 9.0


def _refine_gauge_circle(gray, cx, cy, r):
    # Refine gauge center and radius in a localized ROI to avoid neighboring artifacts
    h, w = gray.shape[:2]
    pad = 1.3
    x0 = max(0, int(cx - r * pad))
    y0 = max(0, int(cy - r * pad))
    x1 = min(w, int(cx + r * pad))
    y1 = min(h, int(cy + r * pad))
    sub = gray[y0:y1, x0:x1]
    if sub.size == 0:
        return cx, cy, r
    blur = cv2.medianBlur(sub, 7)

    best = None
    for param2 in (60, 50, 40, 30, 20):
        circles = cv2.HoughCircles(
            blur, cv2.HOUGH_GRADIENT, dp=1, minDist=int(r),
            param1=100, param2=param2,
            minRadius=int(r * 0.75), maxRadius=int(r * 1.15),
        )
        if circles is None:
            continue
        local_cx, local_cy = cx - x0, cy - y0
        best = min(
            circles[0],
            key=lambda c: (c[0] - local_cx) ** 2 + (c[1] - local_cy) ** 2,
        )
        break

    if best is None:
        return cx, cy, r
    return float(best[0] + x0), float(best[1] + y0), float(best[2])


def detect_gauges(gray, n_gauges=3):
    # Detect the circular gauges using Hough Circle Transform
    h, w = gray.shape[:2]
    blur = cv2.medianBlur(gray, 7)

    min_r = int(h * 0.10)
    max_r = int(h * 0.42)

    candidates = []
    for param2 in (95, 85, 75, 65, 55, 45):
        circles = cv2.HoughCircles(
            blur, cv2.HOUGH_GRADIENT, dp=1.2, minDist=int(h * 0.22),
            param1=120, param2=param2, minRadius=min_r, maxRadius=max_r,
        )
        if circles is None:
            continue
        candidates = []
        for cx, cy, r in circles[0]:
            cx, cy, r = float(cx), float(cy), float(r)
            Y, X = np.ogrid[:h, :w]
            mask = (X - cx) ** 2 + (Y - cy) ** 2 <= (r * 0.6) ** 2
            if mask.sum() == 0:
                continue
            mean_val = gray[mask].mean()
            if mean_val > 140:
                candidates.append((cx, cy, r))
            if len(candidates) >= n_gauges:
                break
        if len(candidates) >= n_gauges:
            break

    candidates = [_refine_gauge_circle(gray, cx, cy, r) for cx, cy, r in candidates]
    candidates.sort(key=lambda c: c[0])
    return candidates[:n_gauges]


def read_needle_angle(gray, cx, cy, r):
    # Isolate the gauge area and threshold the image to find the dark needle
    h, w = gray.shape[:2]
    y0, y1 = max(0, int(cy - r * 1.05)), min(h, int(cy + r * 1.05))
    x0, x1 = max(0, int(cx - r * 1.05)), min(w, int(cx + r * 1.05))
    sub = gray[y0:y1, x0:x1]
    lcx, lcy = cx - x0, cy - y0

    Y, X = np.ogrid[:sub.shape[0], :sub.shape[1]]
    dist = np.hypot(X - lcx, Y - lcy)
    disk_mask = dist < 0.65 * r

    disk_pixels = sub[disk_mask]
    if disk_pixels.size == 0:
        return None
    thresh_val, _ = cv2.threshold(disk_pixels, 0, 255,
                                   cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    dark = (sub.astype(np.float64) < thresh_val).astype(np.uint8) * 255
    dark[~disk_mask] = 0

    # Find the largest connected component connected to the gauge center
    n, labels, stats, _ = cv2.connectedComponentsWithStats(dark, connectivity=8)
    if n <= 1:
        return None

    near_mask = dist < 0.18 * r
    labels_near = labels[near_mask]
    labels_near = labels_near[labels_near > 0]
    if labels_near.size == 0:
        return None
    uniq = np.unique(labels_near)
    areas_uniq = stats[uniq, cv2.CC_STAT_AREA]
    needle_label = int(uniq[np.argmax(areas_uniq)])

    ys, xs = np.where(labels == needle_label)
    pts = np.stack([xs - lcx, ys - lcy], axis=1).astype(np.float64)
    if len(pts) < 5:
        return None

    # Compute the principal axis of the needle using PCA
    cov = np.cov(pts.T)
    eigvals, eigvecs = np.linalg.eigh(cov)
    principal = eigvecs[:, np.argmax(eigvals)]
    perp = np.array([-principal[1], principal[0]])

    proj = pts @ principal
    perp_proj = pts @ perp

    # Identify the needle tip by assuming it's narrower than the counterweight
    def extreme_stats(sign):
        side = proj * sign > 0
        if side.sum() < 3:
            return np.inf, None
        p = np.abs(proj[side])
        cutoff = np.percentile(p, 70)
        extreme = p >= cutoff
        if extreme.sum() < 2:
            return np.inf, None
        width = perp_proj[side][extreme].std()
        centroid = pts[side][extreme].mean(axis=0)
        return width, centroid

    width_pos, centroid_pos = extreme_stats(1.0)
    width_neg, centroid_neg = extreme_stats(-1.0)
    tip_centroid = centroid_pos if width_pos <= width_neg else centroid_neg
    if tip_centroid is None:
        return None

    angle = np.degrees(np.arctan2(tip_centroid[1], tip_centroid[0])) % 360
    return angle


def angle_to_value(angle_deg):
    # Convert the measured angle to a value based on the gauge's sweep
    sweep = (ANGLE_ONE_DEG - ANGLE_ZERO_DEG) % 360
    delta = (angle_deg - ANGLE_ZERO_DEG) % 360
    if delta <= sweep:
        value = delta / sweep
    else:
        dist_to_one = delta - sweep
        dist_to_zero = 360.0 - delta
        value = 1.0 if dist_to_one < dist_to_zero else 0.0
    return float(np.clip(value, 0.0, 1.0))


def read_gauges(image_path):
    img = cv2.imread(image_path)
    if img is None:
        raise FileNotFoundError(image_path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    gauges = detect_gauges(gray, n_gauges=3)

    vis = img.copy()
    values = []
    for i, (cx, cy, r) in enumerate(gauges):
        angle = read_needle_angle(gray, cx, cy, r)
        if angle is None:
            values.append(None)
            continue
        value = angle_to_value(angle)
        values.append(value)

        # Draw annotations on the image
        cv2.circle(vis, (int(cx), int(cy)), int(r), (0, 255, 0), 4)
        cv2.circle(vis, (int(cx), int(cy)), 6, (0, 0, 255), -1)
        tip_x = cx + r * 0.7 * np.cos(np.radians(angle))
        tip_y = cy + r * 0.7 * np.sin(np.radians(angle))
        cv2.line(vis, (int(cx), int(cy)), (int(tip_x), int(tip_y)),
                 (0, 0, 255), 4)
        label = "%.2f" % value
        cv2.putText(vis, label, (int(cx - r), int(cy - r - 15)),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.6, (0, 0, 0), 8, cv2.LINE_AA)
        cv2.putText(vis, label, (int(cx - r), int(cy - r - 15)),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.6, (0, 255, 255), 3, cv2.LINE_AA)

    return values, vis


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("image")
    parser.add_argument("--out", default=None)
    args = parser.parse_args()

    values, vis = read_gauges(args.image)

    for i, v in enumerate(values):
        if v is not None:
            print(v)

    out_path = args.out
    if out_path is None:
        if "." in args.image:
            base, ext = args.image.rsplit(".", 1)
            out_path = base + "_annotated." + ext
        else:
            out_path = args.image + "_annotated.jpg"
    cv2.imwrite(out_path, vis)


if __name__ == "__main__":
    main()
