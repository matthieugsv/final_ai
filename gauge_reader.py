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
# Calibration de l'echelle (meme modele SMC, echelle 0 -> 1 MPa, partagee par
# les 3 capteurs). Mesuree une fois par inspection d'une image de reference,
# puis reutilisee pour tous les cadrans.
# Angles exprimes en degres, convention image (0=droite, 90=bas, 180=gauche,
# 270=haut) ; l'aiguille balaie en augmentant l'angle depuis ANGLE_ZERO_DEG
# jusqu'a ANGLE_ONE_DEG (en "deroulant" le tour, cf. angle_to_value).
#
# LIMITE CONNUE : chaque capteur est une unite physique montee separement et
# peut presenter une legere rotation de montage differente (quelques degres)
# par rapport a la photo de reference utilisee pour cette calibration. Une
# detection automatique du repere "0" propre a chaque capteur a ete testee
# mais s'est averee peu fiable (traits de graduation fins/peu contrastes,
# facilement confondus avec le texte ou la lunette) : mieux vaut une
# calibration globale stable qu'une auto-calibration bruitee. Sur les cas
# testes, l'erreur induite reste de l'ordre de quelques centiemes de MPa.
ANGLE_ZERO_DEG = 183.0   # angle de l'aiguille quand la valeur = 0
ANGLE_ONE_DEG = 9.0      # angle de l'aiguille quand la valeur = 1


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


def read_needle_angle(gray, cx, cy, r):
    """Segmente l'aiguille (seuillage) dans le cadran et regresse son angle.

    1. Seuillage (Otsu) restreint a un disque interne (evite le texte et les
       graduations proches du bord).
    2. Parmi les composantes connexes touchant le voisinage immediat du
       pivot (le centre du cadran, autour duquel l'aiguille tourne), on
       garde la plus grande en aire (evite un pixel de bruit isole).
    3. Analyse en composantes principales (PCA) pour trouver l'axe de
       l'aiguille, puis on distingue la pointe fine (cote a lire) de la
       contre-masse large a l'oppose du pivot par la largeur transverse des
       pixels les plus eloignes de chaque cote (cf. extreme_stats) : le cote
       le plus fin est la pointe. L'angle final est celui du centre de masse
       des pixels extremes de ce cote.
    """
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

    cov = np.cov(pts.T)
    eigvals, eigvecs = np.linalg.eigh(cov)
    principal = eigvecs[:, np.argmax(eigvals)]
    perp = np.array([-principal[1], principal[0]])

    proj = pts @ principal
    perp_proj = pts @ perp

    # L'aiguille a une forme asymetrique : une pointe fine (la valeur lue)
    # et une contre-masse courte et large a l'oppose. On distingue les deux
    # cotes de l'axe principal par la largeur (ecart-type transverse) de
    # leurs pixels les plus eloignes du centre : le cote "pointe" est fin,
    # le cote "contre-masse" est large.
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

    # La direction de la pointe est donnee par le centre de masse des
    # pixels extremes du cote "pointe" (plus fiable que le simple signe du
    # vecteur propre, qui peut etre biaise par l'asymetrie du moyeu).
    angle = np.degrees(np.arctan2(tip_centroid[1], tip_centroid[0])) % 360
    return angle


def angle_to_value(angle_deg):
    """Regression lineaire simple : value = beta0 + beta1 * angle.

    L'angle mesure est "deroule" par rapport a ANGLE_ZERO_DEG. S'il tombe
    dans la zone morte (l'arc sans graduations, en bas du cadran), on le
    ramene a la borne (0 ou 1) la plus proche plutot que de laisser le
    modulo 360 produire un resultat aberrant.
    """
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
        angle = read_needle_angle(gray, cx, cy, r)
        if angle is None:
            values.append(None)
            continue
        value = angle_to_value(angle)
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
