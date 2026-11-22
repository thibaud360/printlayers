#!/usr/bin/env python3
"""printlayers: turn a photo or drawing into flat ink layers for printmaking.

Modes: linocut, screenprint, stencil, riso. Writes one SVG (black = ink) and one
PNG per layer, a proof that overprints the layers on paper, and a contact sheet.
Needs: pillow, numpy, opencv-python, scipy, scikit-learn, scikit-image.
"""
import argparse, os, sys, math, glob
import numpy as np
import cv2
from PIL import Image, ImageDraw, ImageFont, ImageOps
from scipy import ndimage as ndi

PAPER_DEFAULT = "#F3EEE2"

RISO = {  # name: hex (common risograph ink colours)
    "Fluorescent Pink": "#FF48B0", "Blue": "#0078BF", "Yellow": "#FFE800",
    "Teal": "#00838A", "Orange": "#FF6C2F", "Green": "#00A95C",
    "Purple": "#765BA7", "Red": "#FF665E", "Sunflower": "#FFB511",
    "Mint": "#82D8D5", "Black": "#2A2A2E", "Bright Red": "#F15060",
    "Medium Blue": "#3255A4", "Aqua": "#5EC8E5", "Burgundy": "#914E72",
    "Marine Red": "#D2515E",
}

# detail 1..5 -> (smoothing sigma at 1600 px, minimum feature in mm)
DETAIL = {1: (4.0, 1.6), 2: (2.8, 1.2), 3: (2.0, 0.9), 4: (1.3, 0.6), 5: (0.8, 0.4)}


# ----------------------------------------------------------------- helpers
def hex2rgb(h):
    h = h.strip().lstrip("#")
    if len(h) == 3:
        h = "".join(c * 2 for c in h)
    return tuple(int(h[i:i + 2], 16) for i in (0, 2, 4))


def rgb2hex(c):
    return "#%02X%02X%02X" % tuple(int(round(max(0, min(255, v)))) for v in c)


def lum(c):
    r, g, b = [v / 255.0 for v in c]
    return 0.2126 * r + 0.7152 * g + 0.0722 * b


def odd(n, lo=3):
    n = int(round(n))
    n = max(lo, n)
    return n if n % 2 else n + 1


def ellipse(n):
    return cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (odd(n), odd(n)))


def load_image(path, width):
    im = Image.open(path)
    im = ImageOps.exif_transpose(im)
    if im.mode in ("RGBA", "LA"):
        bg = Image.new("RGB", im.size, (255, 255, 255))
        bg.paste(im, mask=im.getchannel("A"))
        im = bg
    im = im.convert("RGB")
    w, h = im.size
    nh = int(round(h * width / w))
    interp = cv2.INTER_CUBIC if width > w else cv2.INTER_AREA
    a = cv2.resize(np.asarray(im), (width, nh), interpolation=interp)
    return a


def gblur(a, s):
    if s <= 0:
        return a
    return cv2.GaussianBlur(a, (0, 0), s)


def remove_small(mask, minarea):
    m = mask.astype(np.uint8)
    n, lab, stats, _ = cv2.connectedComponentsWithStats(m, connectivity=8)
    keep = stats[:, cv2.CC_STAT_AREA] >= minarea
    keep[0] = False
    return keep[lab]


def clean_binary(ink, minpx, minarea):
    """Drop ink and gaps thinner than the tool can carve, remove specks."""
    ink = ink.astype(np.uint8)
    if minpx >= 3:
        k = ellipse(minpx)
        ink = cv2.morphologyEx(ink, cv2.MORPH_OPEN, k)
        ink = cv2.morphologyEx(ink, cv2.MORPH_CLOSE, k)
    ink = gblur(ink.astype(np.float32), max(1.5, minpx * 0.3)) > 0.5
    ink = remove_small(ink, minarea)
    holes = remove_small(~ink, minarea)
    return ~holes


# ------------------------------------------------------------------ vector
def catmull_path(pts):
    n = len(pts)
    p = pts
    d = ["M%.1f %.1f" % (p[0][0], p[0][1])]
    for i in range(n):
        p0, p1, p2, p3 = p[(i - 1) % n], p[i], p[(i + 1) % n], p[(i + 2) % n]
        c1 = p1 + (p2 - p0) / 6.0
        c2 = p2 - (p3 - p1) / 6.0
        d.append("C%.1f %.1f %.1f %.1f %.1f %.1f" % (c1[0], c1[1], c2[0], c2[1], p2[0], p2[1]))
    d.append("Z")
    return "".join(d)


def mask_to_path(mask, mirror=False, eps=0.6, smooth=2.0):
    """Binary mask -> one SVG path string (fill-rule evenodd), in mask pixels."""
    H, W = mask.shape
    pad = 3
    m = np.pad(mask.astype(np.uint8), pad, mode="edge")
    cnts, _ = cv2.findContours(m, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
    parts = []
    for c in cnts:
        pts = c[:, 0, :].astype(np.float64) - pad + 0.5
        n = len(pts)
        if n < 6:
            continue
        s = min(smooth, n / 10.0)
        if s > 0.4:
            pts = ndi.gaussian_filter1d(pts, s, axis=0, mode="wrap")
        pts[:, 0] = np.clip(pts[:, 0], 0, W)
        pts[:, 1] = np.clip(pts[:, 1], 0, H)
        ap = cv2.approxPolyDP(pts.astype(np.float32).reshape(-1, 1, 2), eps, True)[:, 0, :].astype(np.float64)
        if len(ap) < 3 or abs(cv2.contourArea(ap.astype(np.float32))) < 3:
            continue
        if mirror:
            ap[:, 0] = W - ap[:, 0]
        parts.append(catmull_path(ap))
    return "".join(parts)


def write_svg(path, mask, width_mm, mirror=False, eps=0.6, smooth=2.0, title=""):
    H, W = mask.shape
    h_mm = width_mm * H / W
    d = mask_to_path(mask, mirror, eps, smooth)
    svg = ('<?xml version="1.0" encoding="UTF-8"?>\n'
           '<svg xmlns="http://www.w3.org/2000/svg" width="%.2fmm" height="%.2fmm" viewBox="0 0 %d %d">\n'
           '<title>%s</title>\n'
           '<path fill="#000000" fill-rule="evenodd" stroke="none" d="%s"/>\n</svg>\n'
           % (width_mm, h_mm, W, H, title, d))
    with open(path, "w") as f:
        f.write(svg)
    return len(svg)


def reg_marks(H, W, m, size):
    """Registration marks (ring + cross) in the margin of a padded canvas."""
    mk = np.zeros((H + 2 * m, W + 2 * m), np.uint8)
    c = m // 2
    pos = [(c, c), (W + m + c, c), (c, H + m + c), (W + m + c, H + m + c)]
    t = max(2, int(size * 0.12))
    for (x, y) in pos:
        cv2.circle(mk, (x, y), size // 2, 1, t, cv2.LINE_AA)
        cv2.line(mk, (x - size, y), (x + size, y), 1, t, cv2.LINE_AA)
        cv2.line(mk, (x, y - size), (x, y + size), 1, t, cv2.LINE_AA)
    return mk > 0


# ---------------------------------------------------------------- stencils
def add_bridges(ink, bridge_px, min_island):
    """Connect every island of stencil material to the sheet with a thin bridge."""
    ink = ink.copy()
    H, W = ink.shape
    n_bridges = 0
    for it in range(40):
        mat = (~ink).astype(np.uint8)
        n, lab, stats, _ = cv2.connectedComponentsWithStats(mat, connectivity=4)
        border = set(np.unique(np.concatenate([lab[0], lab[-1], lab[:, 0], lab[:, -1]]))) - {0}
        islands = [i for i in range(1, n) if i not in border]
        if not islands:
            break
        tiny = [i for i in islands if stats[i, cv2.CC_STAT_AREA] < min_island]
        if tiny:
            lut = np.zeros(n, bool)
            lut[tiny] = True
            ink |= lut[lab]
        islands = [i for i in islands if i not in set(tiny)]
        if not islands:
            continue
        main = np.isin(lab, list(border))
        dist, (iy, ix) = ndi.distance_transform_edt(~main, return_indices=True)
        objs = ndi.find_objects(lab)
        br = np.zeros((H, W), np.uint8)
        info = []
        for i in islands:
            sl = objs[i - 1]
            sub = lab[sl] == i
            info.append((float(np.where(sub, dist[sl], np.inf).min()), i))
        dmin = min(d for d, _ in info)
        lim = max(2.0 * dmin, 10 * bridge_px) if it < 39 else 1e9
        for dd, i in info:
            if dd > lim:
                continue
            sl = objs[i - 1]
            sub = lab[sl] == i
            d = np.where(sub, dist[sl], np.inf)
            j = np.unravel_index(np.argmin(d), d.shape)
            y, x = j[0] + sl[0].start, j[1] + sl[1].start
            qy, qx = iy[y, x], ix[y, x]
            cv2.line(br, (int(x), int(y)), (int(qx), int(qy)), 1, int(bridge_px))
            n_bridges += 1
        ink &= ~(br > 0)
    return ink, n_bridges


# ------------------------------------------------------------- separations
def lab_of(rgb):
    return cv2.cvtColor(rgb.astype(np.float32) / 255.0, cv2.COLOR_RGB2Lab)


def sat_boost(rgb_c, k=1.12):
    px = np.uint8([[rgb_c]])
    hsv = cv2.cvtColor(px, cv2.COLOR_RGB2HSV).astype(np.float32)
    hsv[..., 1] = np.clip(hsv[..., 1] * k, 0, 255)
    return tuple(int(v) for v in cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2RGB)[0, 0])


def smooth_labels(dist_stack, sigma):
    """Majority-style smoothing of a nearest-colour label map."""
    K = dist_stack.shape[0]
    lab = np.argmin(dist_stack, axis=0)
    if sigma <= 0:
        return lab
    acc = np.empty(dist_stack.shape, np.float32)
    for k in range(K):
        acc[k] = gblur((lab == k).astype(np.float32), sigma)
    return np.argmax(acc, axis=0)


def tidy_labels(lab, minarea, K):
    """Remove tiny regions by giving their pixels to the nearest surviving label."""
    out = lab.copy()
    bad = np.zeros(lab.shape, bool)
    for k in range(K):
        m = (lab == k)
        bad |= m & ~remove_small(m, minarea)
    if bad.any() and (~bad).any():
        _, (iy, ix) = ndi.distance_transform_edt(bad, return_indices=True)
        out = lab[iy, ix]
    return out


def colour_separation(rgb, n, inks, paper, sigma, minarea, seed=1):
    """Return (labels, centres_rgb, paper_index or -1). Label K-1.. = paper if present."""
    from sklearn.cluster import KMeans
    small = cv2.resize(rgb, (400, max(8, int(400 * rgb.shape[0] / rgb.shape[1]))), interpolation=cv2.INTER_AREA)
    small = cv2.bilateralFilter(small, 7, 30, 7)
    lab_s = lab_of(small).reshape(-1, 3)
    full = cv2.bilateralFilter(rgb, 9, 35, 9)
    lab_f = lab_of(full)
    paper_lab = lab_of(np.uint8([[paper]]))[0, 0]
    if inks:
        cents_rgb = [hex2rgb(h) for h in inks]
        cents_lab = [lab_of(np.uint8([[c]]))[0, 0] for c in cents_rgb]
        cents_lab.append(paper_lab)
        paper_idx = len(cents_lab) - 1
        cents_lab = np.array(cents_lab)
    else:
        def fit(k):
            km = KMeans(n_clusters=k, n_init=6, random_state=seed).fit(lab_s)
            return km.cluster_centers_
        c = fit(n + 1)
        light = int(np.argmax(c[:, 0]))
        if c[light, 0] > 74:
            cents_lab = c
            paper_idx = light
        else:
            cents_lab = fit(n)
            paper_idx = -1
    d = np.stack([np.linalg.norm(lab_f - c.reshape(1, 1, 3), axis=2) for c in cents_lab]).astype(np.float32)
    lab = smooth_labels(d, sigma)
    lab = tidy_labels(lab, minarea, len(cents_lab))
    return lab, paper_idx, len(cents_lab)


def mean_colours(rgb, lab, K):
    out = []
    for k in range(K):
        m = lab == k
        out.append(tuple(rgb[m].mean(axis=0)) if m.any() else (128, 128, 128))
    return out


def separate_screenprint(rgb, n, inks, paper, sigma, minarea, trap_px):
    lab, paper_idx, K = colour_separation(rgb, n, inks, paper, sigma, minarea)
    cols = mean_colours(rgb, lab, K)
    if inks:
        cols = [hex2rgb(h) for h in inks] + [paper]
    else:
        cols = [sat_boost(tuple(int(v) for v in c)) for c in cols]
    order = [k for k in range(K) if k != paper_idx]
    order.sort(key=lambda k: -lum(cols[k]))  # print light inks first
    layers = []
    for k in order:
        m = (lab == k)
        layers.append(dict(mask=m, ink=cols[k]))
    # trap: lighter inks run slightly under darker ones
    if trap_px >= 1:
        ker = ellipse(2 * trap_px + 1)
        for i in range(len(layers) - 1):
            layers[i]["mask"] = cv2.dilate(layers[i]["mask"].astype(np.uint8), ker) > 0
    return layers


def separate_stencil(rgb, n, inks, sigma, minpx, minarea, bridge_px, minmm_=1.0):
    g8 = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    g8 = cv2.createCLAHE(clipLimit=1.6, tileGridSize=(4, 4)).apply(g8)
    g8 = cv2.bilateralFilter(g8, 9, 40, 9)
    g = gblur(g8.astype(np.float32), sigma) / 255.0
    th = None
    try:
        from skimage.filters import threshold_multiotsu
        th = list(threshold_multiotsu((g * 255).astype(np.uint8), classes=n + 1) / 255.0)
    except Exception:
        pass
    if th is None or len(th) != n:
        th = list(np.quantile(g, [(i + 1) / (n + 1) for i in range(n)]))
    th = sorted(th)
    layers = []
    for k, t in enumerate(th):
        ink = g < t  # cumulative: lighter stencils cover the darker ones too
        lo = th[k - 1] if k else -1
        band = (g >= lo) & (g < t)
        if inks and k < len(inks):
            col = hex2rgb(inks[len(th) - 1 - k]) if len(inks) >= len(th) else hex2rgb(inks[-1])
        else:
            col = tuple(int(v * 0.9) for v in (rgb[band].mean(axis=0) if band.any() else (40, 40, 40)))
        ink = clean_binary(ink, minpx, minarea)
        # thin strips of material are too fragile: let the ink swallow them
        mat = cv2.morphologyEx((~ink).astype(np.uint8), cv2.MORPH_OPEN, ellipse(minpx))
        ink = ~(mat > 0)
        ink, nb = add_bridges(ink, bridge_px, max(minarea, int((2.2 * minpx / max(minmm_, 0.1)) ** 2)))
        layers.append(dict(mask=ink, ink=col, bridges=nb))
    layers.sort(key=lambda L: -lum(L["ink"]))
    if inks and len(inks) >= len(layers):
        pass
    return layers


def separate_linocut(rgb, sigma, minpx, minarea, texture, line_px, ink_hex, angle, width_px):
    H, W = rgb.shape[:2]
    g8 = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    cl = cv2.createCLAHE(clipLimit=2.2, tileGridSize=(6, 6)).apply(g8)
    g8 = cv2.addWeighted(g8, 0.45, cl, 0.55, 0)
    g8 = cv2.bilateralFilter(g8, 9, 45, 9)
    g = gblur(g8.astype(np.float32), sigma) / 255.0
    # local contrast so shadows read as shapes
    loc = gblur(g, W / 9.0)
    gn = np.clip(g - 0.55 * (loc - np.median(g)), 0, 1)
    lo_p, hi_p = np.percentile(gn, [2, 98])
    gn = np.clip((gn - lo_p) / max(hi_p - lo_p, 1e-3), 0, 1)
    try:
        from skimage.filters import threshold_otsu
        T = float(threshold_otsu(gn))
    except Exception:
        T = float(np.median(gn))
    if texture == "none":
        ink = gn < T
    else:
        t1, t2 = T * 0.38, min(0.98, T * 2.0 + 0.12)
        warp = gblur(gn, W / 30.0)
        yy, xx = np.mgrid[0:H, 0:W].astype(np.float32)
        if angle is None:
            gs = gblur(gn, W / 25.0)
            gy_, gx_ = np.gradient(gs)
            a = 0.5 * math.atan2(2 * (gx_ * gy_).sum(), (gx_ ** 2).sum() - (gy_ ** 2).sum())
        else:
            a = math.radians(angle)
        phi = (xx * math.cos(a) + yy * math.sin(a)) / line_px + 2.5 * warp
        frac = phi - np.floor(phi)
        w = np.clip((gn - t1) / (t2 - t1), 0, 1)  # 0 = solid ink, 1 = all carved
        carved = frac < w
        ink = (gn < t2) & ~carved
        ink |= gn < t1
    ink = clean_binary(ink, minpx, minarea)
    return [dict(mask=ink, ink=hex2rgb(ink_hex))]


# -------------------------------------------------------------------- riso
def pick_riso_pair(rgb, paper):
    small = cv2.resize(rgb, (96, max(8, int(96 * rgb.shape[0] / rgb.shape[1]))), interpolation=cv2.INTER_AREA)
    c = np.clip(small.reshape(-1, 3) / 255.0, 0.02, None)
    p = np.array(paper) / 255.0
    y = -np.log(np.clip(c / p, 0.02, 1.0))
    names = list(RISO)
    A = {k: -np.log(np.clip(np.array(hex2rgb(RISO[k])) / 255.0, 0.05, 1)) for k in names}
    best = None
    for i in range(len(names)):
        for j in range(i + 1, len(names)):
            U = np.stack([A[names[i]], A[names[j]]], 1)
            ab = np.clip(np.linalg.lstsq(U, y.T, rcond=None)[0], 0, 1)
            err = np.mean((U @ ab - y.T) ** 2)
            if best is None or err < best[0]:
                best = (err, names[i], names[j])
    return RISO[best[1]], RISO[best[2]]


def riso_coverages(rgb, inks, paper, sigma):
    p = np.array(paper) / 255.0
    img = cv2.bilateralFilter(rgb, 9, 40, 9)
    c = np.clip(img.astype(np.float32) / 255.0, 0.02, None)
    y = -np.log(np.clip(c / p, 0.02, 1.0))
    U = np.stack([-np.log(np.clip(np.array(hex2rgb(h)) / 255.0, 0.05, 1)) for h in inks], 1).astype(np.float32)
    M = np.linalg.pinv(U)  # 2x3
    ab = np.einsum("kc,hwc->hwk", M, y)
    ab = np.clip(ab, 0, 1)
    out = []
    for k in range(2):
        a = gblur(ab[..., k].astype(np.float32), sigma)
        a = np.clip((a - 0.03) / 0.9, 0, 1) ** 0.9
        out.append(a)
    return out


def halftone(cov, cell_px, angle_deg, grain=0.0, seed=0):
    H, W = cov.shape
    yy, xx = np.mgrid[0:H, 0:W].astype(np.float32)
    a = math.radians(angle_deg)
    u = (xx * math.cos(a) + yy * math.sin(a)) / cell_px
    v = (-xx * math.sin(a) + yy * math.cos(a)) / cell_px
    T = 0.5 + 0.25 * (np.cos(2 * math.pi * u) + np.cos(2 * math.pi * v))
    c = gblur(cov, cell_px * 0.35)
    thr = 1.0 - T
    if grain:
        rng = np.random.default_rng(seed)
        thr = thr + grain * (gblur(rng.standard_normal((H, W)).astype(np.float32), 1.0) * 0.5)
    m = c > thr
    m[c < 0.04] = False
    return m


def separate_riso(rgb, inks, paper, cell_px, sigma, minarea):
    if not inks:
        inks = list(pick_riso_pair(rgb, paper))
    inks = inks[:2]
    if len(inks) < 2:
        inks = inks + [RISO["Black"]]
    covs = riso_coverages(rgb, inks, paper, sigma)
    angles = [15, 75]
    layers = []
    for k in range(2):
        m = halftone(covs[k], cell_px, angles[k], 0.0)
        m = remove_small(m, max(4, minarea // 8))
        layers.append(dict(mask=m, ink=hex2rgb(inks[k])))
    layers.sort(key=lambda L: -lum(L["ink"]))
    return layers


# ------------------------------------------------------------------- proof
def noise(rng, shape, s):
    n = rng.standard_normal(shape).astype(np.float32)
    n = gblur(n, s)
    return n / (n.std() + 1e-6)


def make_paper(H, W, paper, rng, scale):
    base = np.array(paper, np.float32) / 255.0
    fine = noise(rng, (H, W), 0.7 * scale)
    mid = noise(rng, (H, W), 2.5 * scale)
    blot = noise(rng, (H, W), 40 * scale)
    fib = noise(rng, (H, W), 1.0)
    fib = cv2.GaussianBlur(fib, (0, 0), sigmaX=6 * scale, sigmaY=0.6)
    fib = fib / (fib.std() + 1e-6)
    t = 0.022 * fine + 0.014 * mid + 0.018 * blot + 0.010 * fib
    img = base[None, None, :] * (1 + t[..., None])
    yy, xx = np.mgrid[0:H, 0:W].astype(np.float32)
    r = np.sqrt(((xx - W / 2) / (W / 2)) ** 2 + ((yy - H / 2) / (H / 2)) ** 2)
    img *= (1 - 0.05 * np.clip(r - 0.6, 0, 1) ** 1.5)[..., None]
    return np.clip(img, 0, 1)


def render_proof(layers, paper, style="press", misreg=0.0035, seed=7, margin=0.06):
    H, W = layers[0]["mask"].shape
    sc = W / 1600.0
    rng = np.random.default_rng(seed)
    pad = int(W * margin)
    PH, PW = H + 2 * pad, W + 2 * pad
    canvas = make_paper(PH, PW, paper, rng, sc)
    relief = np.zeros((PH, PW), np.float32)
    for i, L in enumerate(layers):
        cov = L["mask"].astype(np.float32)
        if i > 0 and misreg > 0:
            dx = rng.uniform(-1, 1) * misreg * W
            dy = rng.uniform(-1, 1) * misreg * W
            ang = rng.uniform(-1, 1) * 0.12
            Mx = cv2.getRotationMatrix2D((W / 2, H / 2), ang, 1.0)
            Mx[:, 2] += (dx, dy)
            cov = cv2.warpAffine(cov, Mx, (W, H), flags=cv2.INTER_LINEAR, borderValue=0)
        full = np.zeros((PH, PW), np.float32)
        full[pad:pad + H, pad:pad + W] = cov
        edge = 0.75 * sc
        cov = gblur(full, edge)
        n_hi = noise(rng, (PH, PW), 0.8 * sc)
        n_lo = noise(rng, (PH, PW), 12 * sc)
        if style == "spray":
            halo = gblur(full, 5 * sc)
            stip = noise(rng, (PH, PW), 0.6 * sc)
            cov = np.clip(cov + 0.5 * halo * (stip > 1.35), 0, 1)
            cov = np.clip(gblur(full, 1.6 * sc) + 0.35 * halo * (stip > 0.9) * 0.5, 0, 1)
            dens = 0.94 + 0.04 * n_lo
        elif style == "riso":
            cov = np.clip(cov + 0.10 * n_hi * (cov > 0.05), 0, 1)
            dens = 0.90 + 0.06 * n_lo
            cov = cov * (1 - 0.35 * (noise(rng, (PH, PW), 0.9 * sc) > 1.5))
        else:
            cov = np.clip(cov + 0.12 * n_hi * ((cov > 0.05) & (cov < 0.95)) , 0, 1)
            dens = 0.975 + 0.012 * n_lo
            cov = cov * (1 - 0.6 * (noise(rng, (PH, PW), 0.7 * sc) > 2.6))
        # mottle in solid areas
        cov = np.clip(cov * dens, 0, 1)
        tx = np.clip(np.array(L["ink"], np.float32) / 255.0, 0.02, 1.0)
        canvas = canvas * (1 - cov[..., None]) + canvas * tx[None, None, :] * cov[..., None]
        relief += cov
    if style != "spray":
        e = gblur(relief, 2.2 * sc)
        gx = cv2.Sobel(e, cv2.CV_32F, 1, 0, ksize=3)
        gy = cv2.Sobel(e, cv2.CV_32F, 0, 1, ksize=3)
        sh = np.clip((gx + gy) * 0.10, -0.045, 0.045)
        canvas = canvas * (1 + sh[..., None])
    out = (np.clip(canvas, 0, 1) * 255 + 0.5).astype(np.uint8)
    return out


# ------------------------------------------------------------------ sheets
def get_font(size):
    for p in ["/usr/share/fonts/opentype/inter/Inter-Regular.otf",
              "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
              "/Library/Fonts/Arial.ttf", "/System/Library/Fonts/Helvetica.ttc",
              "C:/Windows/Fonts/arial.ttf", "DejaVuSans.ttf", "Arial.ttf"]:
        try:
            return ImageFont.truetype(p, size)
        except Exception:
            continue
    try:
        return ImageFont.load_default(size)
    except Exception:
        return ImageFont.load_default()


def layer_panel(mask, ink, h):
    H, W = mask.shape
    w = max(1, int(W * h / H))
    m = cv2.resize(mask.astype(np.float32), (w, h), interpolation=cv2.INTER_AREA)
    shown = np.array(ink, np.float32)
    # light inks vanish on white: show them a touch deeper
    L = lum(ink)
    if L > 0.72:
        shown = shown * (0.72 / L) ** 0.6
    bg = np.array([250, 248, 243], np.float32)
    img = bg[None, None, :] * (1 - m[..., None]) + shown[None, None, :] * m[..., None]
    out = Image.fromarray(np.clip(img, 0, 255).astype(np.uint8))
    ImageDraw.Draw(out).rectangle([0, 0, w - 1, h - 1], outline=(212, 208, 198))
    return out


def contact_sheet(items, title, panel_h=620):
    """items: list of (mask, ink, caption)."""
    gap, pad, cap_h, top = 24, 36, 70, 70
    panels = [layer_panel(m, c, panel_h) for m, c, _ in items]
    total_w = sum(p.width for p in panels) + gap * (len(panels) - 1) + 2 * pad
    sheet = Image.new("RGB", (total_w, top + panel_h + cap_h + pad // 2), (236, 233, 225))
    d = ImageDraw.Draw(sheet)
    f_t, f_c, f_s = get_font(26), get_font(20), get_font(17)
    d.text((pad, 22), title, fill=(60, 57, 50), font=f_t)
    x = pad
    for p, (m, c, cap) in zip(panels, items):
        sheet.paste(p, (x, top))
        d.rounded_rectangle([x, top + panel_h + 18, x + 26, top + panel_h + 44], 5, fill=tuple(int(v) for v in c), outline=(150, 146, 136))
        d.text((x + 38, top + panel_h + 17), cap[0], fill=(50, 47, 42), font=f_c)
        d.text((x + 38, top + panel_h + 43), cap[1], fill=(120, 116, 106), font=f_s)
        x += p.width + gap
    return sheet


def make_strip(orig, proof, layer_items, h=700):
    def fit(im):
        w = max(1, int(im.width * h / im.height))
        return im.resize((w, h), Image.LANCZOS)
    panels = [fit(Image.fromarray(orig)), fit(Image.fromarray(proof))]
    panels += [layer_panel(m, c, h) for m, c in layer_items]
    gap = 14
    W = sum(p.width for p in panels) + gap * (len(panels) + 1)
    s = Image.new("RGB", (W, h + 2 * gap), (236, 233, 225))
    x = gap
    for p in panels:
        s.paste(p, (x, gap))
        x += p.width + gap
    return s


# -------------------------------------------------------------------- main
def process(path, a):
    stem = os.path.splitext(os.path.basename(path))[0]
    rgb = load_image(path, a.px)
    H, W = rgb.shape[:2]
    src_w = Image.open(path).size[0]
    up = max(1.0, W / float(src_w))
    pxmm = W / a.width_mm
    sig_base, minmm = DETAIL[a.detail]
    sigma = sig_base * W / 1600.0 * max(1.0, up * 0.7)
    if a.mode in ("stencil",):
        minmm = max(minmm, 1.0)
    minpx = max(3, int(round(minmm * pxmm)))
    minarea = max(12, int(minpx * minpx * 1.2))
    inks = a.inks
    paper = hex2rgb(a.paper)
    style = "press"
    note = ""

    if a.mode == "linocut":
        ink_hex = inks[0] if inks else "#17130F"
        line_px = a.line_mm * pxmm
        layers = separate_linocut(rgb, sigma, minpx, minarea, a.texture, line_px, ink_hex, a.angle, W)
        misreg = 0.0
    elif a.mode == "screenprint":
        n = int(np.clip(a.layers if a.layers else (len(inks) if inks else 3), 2, 4))
        if inks:
            inks = inks[:4]
            n = len(inks)
        trap = max(1, int(round(0.15 * pxmm)))
        layers = separate_screenprint(rgb, n, inks, paper, sigma * 0.8, minarea * 2, trap)
        misreg = a.misreg
    elif a.mode == "stencil":
        n = int(np.clip(a.layers if a.layers else (len(inks) if inks else 2), 1, 4))
        bridge_px = max(3, int(round(a.bridge_mm * pxmm)))
        layers = separate_stencil(rgb, n, inks, sigma * 1.1, minpx, minarea, bridge_px, minmm)
        style = "spray"
        misreg = a.misreg
        note = "bridges: " + ", ".join(str(L.get("bridges", 0)) for L in layers)
    elif a.mode == "riso":
        cell_px = max(5.0, a.cell_mm * pxmm)
        layers = separate_riso(rgb, inks, paper, cell_px, sigma * 0.6, minarea)
        style = "riso"
        misreg = a.misreg * 1.3
    else:
        sys.exit("unknown mode")

    # empty layers are useless
    layers = [L for L in layers if L["mask"].any()]
    if not layers:
        sys.exit("No ink found for %s; try --detail 4 or other --inks." % path)

    os.makedirs(a.out, exist_ok=True)
    base = os.path.join(a.out, "%s_%s" % (stem, a.mode))
    multi = len(layers) > 1
    marg = int(round(6 * pxmm)) if (multi and not a.no_marks) else 0
    mirror_all = a.mirror == "on" or (a.mirror == "auto" and a.mode == "linocut")
    eps = 0.45 if a.mode == "riso" else 0.6
    smooth = 1.4 if a.mode == "riso" else 2.0
    written = []
    for i, L in enumerate(layers, 1):
        m = L["mask"]
        if marg:
            pm = np.zeros((H + 2 * marg, W + 2 * marg), bool)
            pm[marg:marg + H, marg:marg + W] = m
            pm |= reg_marks(H, W, marg, int(round(2.6 * pxmm)))
        else:
            pm = m
        tag = "%s_layer%d" % (base, i)
        title = "%s layer %d ink %s" % (a.mode, i, rgb2hex(L["ink"]))
        size = write_svg(tag + ".svg", pm, a.width_mm + 2 * marg / pxmm, False, eps, smooth, title)
        Image.fromarray(np.where(pm, 0, 255).astype(np.uint8), "L").save(tag + ".png", optimize=True)
        written += [tag + ".svg", tag + ".png"]
        if mirror_all:
            write_svg(tag + "_mirror.svg", pm, a.width_mm + 2 * marg / pxmm, True, eps, smooth, title + " (mirrored)")
            Image.fromarray(np.where(pm[:, ::-1], 0, 255).astype(np.uint8), "L").save(tag + "_mirror.png", optimize=True)
            written += [tag + "_mirror.svg", tag + "_mirror.png"]

    proof = render_proof(layers, paper, style, misreg=misreg)
    Image.fromarray(proof).save(base + "_proof.png", optimize=True)
    written.append(base + "_proof.png")

    items = []
    for i, L in enumerate(layers, 1):
        cap2 = ("print first" if i == 1 else "then") if multi else "black = ink"
        items.append((L["mask"], L["ink"], ("Layer %d  %s" % (i, rgb2hex(L["ink"])), cap2)))
    if mirror_all:
        for i, L in enumerate(layers, 1):
            items.append((L["mask"][:, ::-1], L["ink"], ("Layer %d mirrored" % i, "ready to carve")))
    sheet = contact_sheet(items, "printlayers · %s · %s · %d mm wide" % (stem, a.mode, a.width_mm))
    sheet.save(base + "_sheet.png", optimize=True)
    written.append(base + "_sheet.png")

    if a.strip:
        s = make_strip(rgb, proof, [(L["mask"], L["ink"]) for L in layers] +
                       ([(layers[0]["mask"][:, ::-1], layers[0]["ink"])] if mirror_all and not multi else []))
        s.convert("RGB").save(base + "_strip.jpg", quality=88)
        written.append(base + "_strip.jpg")

    print("%s [%s] %d layer(s): %s%s" % (os.path.basename(path), a.mode, len(layers),
          ", ".join(rgb2hex(L["ink"]) for L in layers), ("  " + note) if note else ""))
    return written


def main():
    ap = argparse.ArgumentParser(description="Turn a photo or drawing into flat ink layers for linocut, screen print, stencil or riso.")
    ap.add_argument("images", nargs="+", help="input photo(s) or drawing(s)")
    ap.add_argument("--mode", choices=["linocut", "screenprint", "stencil", "riso"], default="linocut")
    ap.add_argument("--layers", type=int, default=0, help="number of ink layers (screenprint 2-4, stencil 1-4; default 3 / 2)")
    ap.add_argument("--inks", nargs="+", metavar="HEX", help='ink colours, e.g. --inks "#1B2A49" "#E4572E" (default: picked from the photo)')
    ap.add_argument("--detail", type=int, default=3, choices=[1, 2, 3, 4, 5], help="1 = bold and simple, 5 = fine detail (default 3)")
    ap.add_argument("--width-mm", type=float, default=150, help="physical width of the print in mm (SVG size), default 150")
    ap.add_argument("--paper", default=PAPER_DEFAULT, help="paper colour for the proof")
    ap.add_argument("--texture", choices=["lines", "none"], default="lines", help="linocut: carved-line texture in the midtones")
    ap.add_argument("--line-mm", type=float, default=2.4, help="linocut: spacing of carved lines in mm")
    ap.add_argument("--angle", type=float, default=None, help="linocut: direction of carved lines in degrees (default: follow the tones)")
    ap.add_argument("--bridge-mm", type=float, default=0.9, help="stencil: width of the bridges in mm")
    ap.add_argument("--cell-mm", type=float, default=1.5, help="riso: halftone dot spacing in mm")
    ap.add_argument("--mirror", choices=["auto", "on", "off"], default="auto", help="also write mirrored SVG/PNG (auto = linocut only)")
    ap.add_argument("--misreg", type=float, default=0.0035, help="proof: misregistration as a fraction of the width (0 = perfect)")
    ap.add_argument("--no-marks", action="store_true", help="no registration marks on multi-layer SVGs")
    ap.add_argument("--px", type=int, default=1600, help="working width in pixels")
    ap.add_argument("--strip", action="store_true", help="also write original | proof | layers strip (JPG)")
    ap.add_argument("-o", "--out", default="printlayers_out")
    a = ap.parse_args()
    files = []
    for p in a.images:
        files += sorted(glob.glob(p)) or [p]
    for p in files:
        if not os.path.exists(p):
            print("missing:", p, file=sys.stderr)
            continue
        process(p, a)
    print("done ->", os.path.abspath(a.out))


if __name__ == "__main__":
    main()
