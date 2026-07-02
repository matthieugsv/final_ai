#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
gauge_reader.py

Lecture automatique de la valeur (entre 0 et 1) affichee par l'aiguille de 3
manometres analogiques (capteurs de pression SMC, echelle 0-1 MPa) a partir
d'une photo.

Notions du cours mobilisees :
- Segmentation par seuillage (threshold-based segmentation) pour isoler
  l'aiguille noire sur le cadran blanc.
- Detection d'objets (localisation des 3 cadrans circulaires dans l'image,
  cf. object detection) via la transformee de Hough (contours + gradients).
- Regression lineaire simple (Part 4 - Supervised learning) pour convertir
  l'angle de l'aiguille en valeur : value = beta0 + beta1 * angle.

Usage:
    python gauge_reader.py chemin/vers/image.jpg [--out annotated.jpg]

Le script affiche les 3 valeurs (dans l'ordre gauche -> droite) et enregistre
une image annotee montrant les cadrans detectes et l'angle lu pour chaque
aiguille (utile pour verifier visuellement le resultat).
"""

import argparse
import sys
import numpy as np
import cv2

# --------------------------------------------------------------------------
# Calibration de l'echelle (modele SMC, echelle 0 -> 1 MPa).
# Chaque capteur est monte avec une legere rotation differente, donc on
# calibre individuellement l'angle du zero et du un pour chacun des 3
# cadrans (gauche -> droite). Les angles sont determines par regression
# lineaire sur les images du dataset de reference.
#
# Angles exprimes en degres, convention image (0=droite, 90=bas, 180=gauche,
# 270=haut) ; l'aiguille balaie en augmentant l'angle depuis angle_zero
# jusqu'a angle_one (en "deroulant" le tour, cf. angle_to_value).
#
# Format : (angle_zero, angle_one) pour chaque capteur.
GAUGE_CALIBRATIONS = [
    (177.8, 20.6),   # Capteur 1 (gauche)
    (182.6, 25.4),   # Capteur 2 (centre)
    (169.8, 12.7),   # Capteur 3 (droite)
]


def _refine_gauge_circle(gray, cx, cy, r):
    """Affine la position/rayon d'un cadran deja localise approximativement.

    La detection grossiere (sur l'image entiere) peut se faire piéger par un
    objet voisin (ex: le raccord noir au-dessus du cadran). On recadre donc
    une petite zone autour de l'estimation grossiere et on relance Hough
    Circle avec un rayon tres contraint (proche du rayon grossier) : dans
    cette zone reduite, le seul grand cercle net est la lunette du cadran.
    """
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
        # Parmi les cercles trouves dans cette zone reduite, on garde celui
        # le plus proche de l'estimation grossiere (evite un objet voisin).
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
    """Detecte les cadrans circulaires (object detection) via Hough Circle.

    Detection grossiere sur l'image entiere puis affinage local (voir
    _refine_gauge_circle) pour obtenir un centre/rayon precis meme si un
    objet voisin (raccord, vanne) perturbe la detection globale.

    Retourne une liste de (cx, cy, r) triee de gauche a droite.
    """
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


def read_needle_angle(img, cx, cy, r):
    """
    Isole l'aiguille par seuillage adaptatif (Otsu) et operations morphologiques,
    puis determine l'angle de la pointe via le centre de masse pondere par la
    distance des pixels de l'aiguille dans l'anneau 0.30r-0.55r. La ponderation
    par la distance donne plus de poids aux pixels eloignes du centre (la pointe)
    et reduit l'influence du texte et de la contre-masse.
    """
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if len(img.shape) == 3 else img
    h, w = gray.shape[:2]
    y0, y1 = max(0, int(cy - r * 1.05)), min(h, int(cy + r * 1.05))
    x0, x1 = max(0, int(cx - r * 1.05)), min(w, int(cx + r * 1.05))
    sub = gray[y0:y1, x0:x1]
    sub_img = img[y0:y1, x0:x1]
    lcx, lcy = cx - x0, cy - y0

    Y, X = np.ogrid[:sub.shape[0], :sub.shape[1]]
    dist = np.hypot(X - lcx, Y - lcy)
    disk_mask = dist < 0.65 * r

    # Filtrer les pixels verts (marqueurs de consigne)
    if len(img.shape) == 3:
        b, g, r_ch = cv2.split(sub_img.astype(np.int16))
        greenness = g - np.maximum(r_ch, b)
        sub_mod = sub.copy()
        sub_mod[greenness > 10] = 255
    else:
        sub_mod = sub.copy()

    disk_pixels = sub_mod[disk_mask]
    if disk_pixels.size == 0:
        return None

    thresh_val, _ = cv2.threshold(disk_pixels, 0, 255,
                                   cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    dark = (sub_mod.astype(np.float64) < thresh_val).astype(np.uint8) * 255
    dark[~disk_mask] = 0

    # Combler le trou cause par le reflet a la base de l'aiguille
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (11, 11))
    dark = cv2.morphologyEx(dark, cv2.MORPH_CLOSE, kernel)

    # Remplissage du moyeu pour connecter l'aiguille au centre
    hub_radius = 0.25 * r
    dark[dist < hub_radius] = 255

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

    # -- Determiner la direction de la pointe ------------------------------
    # On utilise les pixels de l'aiguille dans l'anneau 0.30r-0.55r pour
    # eviter le moyeu (< 0.30r) et le texte/graduations (> 0.55r).
    needle_full = labels == needle_label
    ring_mask = needle_full & (dist > 0.30 * r) & (dist < 0.55 * r)
    ys, xs = np.where(ring_mask)

    if len(ys) < 3:
        # Si trop peu de pixels dans l'anneau restreint, elargir a 0.65r
        ring_mask = needle_full & (dist > 0.30 * r)
        ys, xs = np.where(ring_mask)
        if len(ys) < 3:
            # Dernier recours : tous les pixels
            ys, xs = np.where(needle_full)
            if len(ys) < 3:
                return None

    dx = xs.astype(np.float64) - lcx
    dy = ys.astype(np.float64) - lcy
    pixel_dists = np.hypot(dx, dy)

    # Balayage angulaire pondere par la distance : on divise le cercle en
    # secteurs de 5 degres et on somme les distances dans chaque secteur.
    # Le secteur avec le score le plus eleve correspond a la pointe (longue,
    # avec beaucoup de pixels loin du centre). Le texte, etant court et
    # isole, aura un score plus faible.
    n_sectors = 72  # 360 / 5
    sector_score = np.zeros(n_sectors)
    pixel_angles = np.degrees(np.arctan2(dy, dx)) % 360
    sector_idx = (pixel_angles / 5.0).astype(int) % n_sectors

    for i in range(len(pixel_dists)):
        sector_score[sector_idx[i]] += pixel_dists[i]

    # Lissage circulaire (fenetre de 3 secteurs = 15 degres)
    padded = np.concatenate([sector_score[-2:], sector_score, sector_score[:2]])
    smoothed = np.convolve(padded, np.ones(5) / 5.0, mode='valid')

    best_sector = int(np.argmax(smoothed))
    angle = (best_sector * 5.0 + 2.5) % 360
    return angle


def angle_to_value(angle_deg, gauge_index=0):
    """Regression lineaire simple : value = (angle - angle_zero) / sweep.

    Utilise la calibration propre a chaque capteur (gauge_index). L'angle
    mesure est "deroule" par rapport a angle_zero. S'il tombe dans la zone
    morte (l'arc sans graduations, en bas du cadran), on le ramene a la
    borne (0 ou 1) la plus proche.
    """
    if gauge_index < len(GAUGE_CALIBRATIONS):
        az, ao = GAUGE_CALIBRATIONS[gauge_index]
    else:
        az, ao = GAUGE_CALIBRATIONS[0]
    sweep = (ao - az) % 360
    delta = (angle_deg - az) % 360
    if delta <= sweep:
        value = delta / sweep
    else:
        dist_to_one = delta - sweep
        dist_to_zero = 360.0 - delta
        value = 1.0 if dist_to_one < dist_to_zero else 0.0
    return float(np.clip(value, 0.0, 1.0))


def read_gauges(image_path):
    """Pipeline complet : renvoie la liste des 3 valeurs (gauche -> droite)
    et l'image annotee (numpy array BGR)."""
    img = cv2.imread(image_path)
    if img is None:
        raise FileNotFoundError("Impossible de lire l'image : " + str(image_path))
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    gauges = detect_gauges(gray, n_gauges=3)
    if len(gauges) < 3:
        print("Attention : seulement " + str(len(gauges)) +
              " cadran(s) detecte(s) au lieu de 3.", file=sys.stderr)

    vis = img.copy()
    values = []
    for i, (cx, cy, r) in enumerate(gauges):
        angle = read_needle_angle(img, cx, cy, r)
        if angle is None:
            values.append(None)
            continue
        value = angle_to_value(angle, gauge_index=i)
        values.append(value)

        cv2.circle(vis, (int(cx), int(cy)), int(r), (0, 255, 0), 4)
        cv2.circle(vis, (int(cx), int(cy)), 6, (0, 0, 255), -1)
        tip_x = cx + r * 0.7 * np.cos(np.radians(angle))
        tip_y = cy + r * 0.7 * np.sin(np.radians(angle))
        cv2.line(vis, (int(cx), int(cy)), (int(tip_x), int(tip_y)),
                 (0, 0, 255), 4)
        label = "#%d: %.2f" % (i + 1, value)
        cv2.putText(vis, label, (int(cx - r), int(cy - r - 15)),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.6, (0, 0, 0), 8, cv2.LINE_AA)
        cv2.putText(vis, label, (int(cx - r), int(cy - r - 15)),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.6, (0, 255, 255), 3, cv2.LINE_AA)

    return values, vis


def main():
    parser = argparse.ArgumentParser(
        description="Lit la valeur (0-1) affichee par l'aiguille de 3 "
                    "manometres sur une photo.")
    parser.add_argument("image", help="Chemin de l'image en entree")
    parser.add_argument("--out", default=None,
                         help="Chemin de l'image annotee de sortie "
                              "(par defaut: <image>_annotated.jpg)")
    args = parser.parse_args()

    values, vis = read_gauges(args.image)

    print("Valeurs lues (gauche -> droite) :")
    for i, v in enumerate(values):
        txt = ("%.3f" % v) if v is not None else "non detectee"
        print("  Capteur " + str(i + 1) + ": " + txt)

    out_path = args.out
    if out_path is None:
        if "." in args.image:
            base, ext = args.image.rsplit(".", 1)
            out_path = base + "_annotated." + ext
        else:
            out_path = args.image + "_annotated.jpg"
    cv2.imwrite(out_path, vis)
    print("")
    print("Image annotee enregistree : " + out_path)


if __name__ == "__main__":
    main()
