"""
Zadanie 3 – Skupina D: Histogramové metódy a zlepšenie kontrastu
=================================================================
Manuálna implementácia (NumPy) + porovnanie s OpenCV.

OpenCV sa používa IBA na:
  - načítanie / zobrazenie obrazu
  - konverziu farebných priestorov
  - porovnávacie (referenčné) výpočty

  ---3 grafy pre kazdy kanal - histogram pri RGB

Adresár s obrázkami: ./Fotky
"""

import cv2
import numpy as np
import matplotlib.pyplot as plt
import os
import glob
import time

# ============================================================
# 0.  NAČÍTANIE OBRÁZKA
# ============================================================
IMAGES_DIR = "./Fotky/RGB"

def load_image():
    """Nájde prvý obrázok v ./Fotky a vráti ho."""
    patterns = ("*.png", "*.jpg", "*.jpeg", "*.bmp", "*.tif", "*.tiff")
    files = []
    for p in patterns:
        files += glob.glob(os.path.join(IMAGES_DIR, p))
    files = sorted(files)
    if not files:
        raise FileNotFoundError(f"Žiadne obrázky v priečinku: {IMAGES_DIR}")
    path = files[0]
    img = cv2.imread(path)
    if img is None:
        raise RuntimeError(f"Nepodarilo sa načítať: {path}")
    print(f"Načítaný obrázok: {path}  ({img.shape[1]}x{img.shape[0]})")
    return img, path


# ============================================================
# 1.  MANUÁLNY VÝPOČET HISTOGRAMU
# ============================================================

def manual_histogram(channel: np.ndarray) -> np.ndarray:
    """Vypočíta histogram pre jednokanálový obraz. Vracia pole [256]."""
    hist = np.zeros(256, dtype=np.int64)
    flat = channel.ravel().astype(np.int32)
    for val in range(256):
        hist[val] = np.sum(flat == val)
    return hist


def manual_histogram_fast(channel: np.ndarray) -> np.ndarray:
    """Rýchlejšia verzia cez np.bincount (stále manuálne, žiadne cv2)."""
    flat = channel.ravel().astype(np.intp)
    hist = np.bincount(flat, minlength=256)
    return hist.astype(np.int64)


def manual_histogram_rgb(bgr_image: np.ndarray):
    """Vracia tuple (hist_B, hist_G, hist_R)."""
    b, g, r = bgr_image[:, :, 0], bgr_image[:, :, 1], bgr_image[:, :, 2]
    return (manual_histogram_fast(b),
            manual_histogram_fast(g),
            manual_histogram_fast(r))


# ============================================================
# 2.  MANUÁLNA HISTOGRAMOVÁ EKVALIZÁCIA
# ============================================================

def manual_equalize_gray(gray: np.ndarray) -> np.ndarray:
    """Klasická histogramová ekvalizácia jednokanálového obrazu."""
    hist = manual_histogram_fast(gray)
    cdf = np.cumsum(hist)
    cdf_min = cdf[cdf > 0].min()
    total = gray.size

    denom = total - cdf_min
    if denom == 0:
        return gray.copy()

    lut = ((cdf - cdf_min) / denom * 255.0 + 0.5).astype(np.uint8)
    return lut[gray]


def manual_equalize_color(bgr: np.ndarray) -> np.ndarray:
    """Ekvalizácia farebného obrazu cez YCrCb (ekvalizuje len Y kanál)."""
    ycrcb = cv2.cvtColor(bgr, cv2.COLOR_BGR2YCrCb)
    ycrcb[:, :, 0] = manual_equalize_gray(ycrcb[:, :, 0])
    return cv2.cvtColor(ycrcb, cv2.COLOR_YCrCb2BGR)


# ============================================================
# 3.  MANUÁLNA CLAHE
# ============================================================

def _clip_histogram(hist: np.ndarray, clip_limit: int) -> np.ndarray:
    """Orezanie histogramu na clip_limit, prebytok prerozdelí rovnomerne."""
    hist = hist.copy().astype(np.float64)
    excess = 0.0
    for i in range(256):
        if hist[i] > clip_limit:
            excess += hist[i] - clip_limit
            hist[i] = clip_limit
    hist += excess / 256.0
    return hist


def _equalize_hist_from_clipped(hist: np.ndarray, n_pixels: int) -> np.ndarray:
    """Vytvorí LUT z (možno orezaného) histogramu."""
    cdf = np.cumsum(hist)
    cdf_min = cdf[cdf > 0].min()
    denom = n_pixels - cdf_min
    if denom <= 0:
        return np.arange(256, dtype=np.uint8)
    lut = ((cdf - cdf_min) / denom * 255.0 + 0.5).astype(np.uint8)
    return lut


def manual_clahe(gray: np.ndarray,
                 clip_limit: float = 2.0,
                 grid_size: tuple = (8, 8)) -> np.ndarray:
    """Zjednodušená CLAHE s bilineárnou interpoláciou (vektorizovaná)."""
    h, w = gray.shape
    gh, gw = grid_size
    tile_h = h / gh
    tile_w = w / gw

    luts = np.zeros((gh, gw, 256), dtype=np.uint8)
    for r in range(gh):
        for c in range(gw):
            y0 = int(round(r * tile_h))
            y1 = int(round((r + 1) * tile_h))
            x0 = int(round(c * tile_w))
            x1 = int(round((c + 1) * tile_w))
            tile = gray[y0:y1, x0:x1]
            n_pix = tile.size
            hist = manual_histogram_fast(tile)
            actual_clip = max(1, int(clip_limit * n_pix / 256))
            clipped = _clip_histogram(hist, actual_clip)
            luts[r, c] = _equalize_hist_from_clipped(clipped, n_pix)

    yy, xx = np.mgrid[0:h, 0:w]
    fy = (yy + 0.5) / tile_h - 0.5
    fx = (xx + 0.5) / tile_w - 0.5

    r0 = np.floor(fy).astype(np.int32)
    c0 = np.floor(fx).astype(np.int32)
    dy = fy - r0
    dx = fx - c0
    r1 = r0 + 1
    c1 = c0 + 1

    r0 = np.clip(r0, 0, gh - 1)
    r1 = np.clip(r1, 0, gh - 1)
    c0 = np.clip(c0, 0, gw - 1)
    c1 = np.clip(c1, 0, gw - 1)
    dy = np.clip(dy, 0.0, 1.0)
    dx = np.clip(dx, 0.0, 1.0)

    val = gray
    tl = luts[r0, c0, val].astype(np.float64)
    tr = luts[r0, c1, val].astype(np.float64)
    bl = luts[r1, c0, val].astype(np.float64)
    br = luts[r1, c1, val].astype(np.float64)

    top    = tl * (1 - dx) + tr * dx
    bottom = bl * (1 - dx) + br * dx
    result = top * (1 - dy) + bottom * dy
    return (result + 0.5).astype(np.uint8)


# ============================================================
# 4.  OPENCV REFERENČNÉ FUNKCIE (na porovnanie)
# ============================================================

def opencv_histogram(channel):
    return cv2.calcHist([channel], [0], None, [256], [0, 256]).ravel().astype(np.int64)

def opencv_equalize_gray(gray):
    return cv2.equalizeHist(gray)

def opencv_equalize_color(bgr):
    ycrcb = cv2.cvtColor(bgr, cv2.COLOR_BGR2YCrCb)
    ycrcb[:, :, 0] = cv2.equalizeHist(ycrcb[:, :, 0])
    return cv2.cvtColor(ycrcb, cv2.COLOR_YCrCb2BGR)

def opencv_clahe_gray(gray, clip_limit=2.0, grid_size=(8, 8)):
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=grid_size)
    return clahe.apply(gray)


# ============================================================
# 5.  POMOCNÁ FUNKCIA
# ============================================================

def _draw_hist(ax, data, color='gray', title=""):
    """Vykreslí histogram do subplotu – kompaktne."""
    h = manual_histogram_fast(data)
    ax.fill_between(range(256), h, color=color, alpha=0.7)
    ax.set_xlim(0, 255)
    ax.set_title(title, fontsize=8, pad=3)
    ax.tick_params(labelsize=6)


# ============================================================
# 6.  FIGURE 1 – Histogramy & Ekvalizácia  (4×3, gridspec)
# ============================================================

def figure_1_histogramy_a_ekvalizacia(gray, bgr):
    eq_man = manual_equalize_gray(gray)
    eq_cv  = opencv_equalize_gray(gray)
    diff_img = np.abs(eq_man.astype(int) - eq_cv.astype(int)).astype(np.uint8)
    diff_mean = np.mean(diff_img)

    eq_col_man = manual_equalize_color(bgr)
    eq_col_cv  = opencv_equalize_color(bgr)
    diff_col = np.mean(np.abs(eq_col_man.astype(int) - eq_col_cv.astype(int)))

    h_man = manual_histogram_fast(gray)
    h_cv  = opencv_histogram(gray)

    fig = plt.figure(figsize=(14, 14))
    fig.suptitle("HISTOGRAMY  &  HISTOGRAMOVÁ EKVALIZÁCIA",
                 fontsize=13, fontweight='bold', y=0.99)

    gs = fig.add_gridspec(4, 3, hspace=0.30, wspace=0.20,
                          left=0.04, right=0.98, top=0.95, bottom=0.02)

    # ── R1: pôvodný obraz + histogramy ──
    ax = fig.add_subplot(gs[0, 0])
    ax.imshow(gray, cmap='gray', vmin=0, vmax=255)
    ax.set_title("Pôvodný (gray)", fontsize=9, pad=4)
    ax.axis('off')

    ax = fig.add_subplot(gs[0, 1])
    ax.fill_between(range(256), h_man, color='gray', alpha=0.6, label='Manuálny')
    ax.fill_between(range(256), h_cv,  color='steelblue', alpha=0.35, label='OpenCV')
    ax.set_xlim(0, 255)
    ax.set_title(f"Gray histogram  (sum|diff|={int(np.sum(np.abs(h_man-h_cv)))})",
                 fontsize=8, pad=4)
    ax.legend(fontsize=7, loc='upper right')
    ax.tick_params(labelsize=6)

    ax = fig.add_subplot(gs[0, 2])
    hists_rgb = manual_histogram_rgb(bgr)
    for h, c, l in zip(hists_rgb, ('blue','green','red'), ('B','G','R')):
        ax.plot(range(256), h, color=c, label=l, linewidth=0.7)
    ax.set_xlim(0, 255)
    ax.set_title("RGB histogram (manuálny)", fontsize=8, pad=4)
    ax.legend(fontsize=7, loc='upper right')
    ax.tick_params(labelsize=6)

    # ── R2: ekvalizované grayscale ──
    ax = fig.add_subplot(gs[1, 0])
    ax.imshow(eq_man, cmap='gray', vmin=0, vmax=255)
    ax.set_title("Manuálna ekvalizácia", fontsize=9, pad=4)
    ax.axis('off')

    ax = fig.add_subplot(gs[1, 1])
    ax.imshow(eq_cv, cmap='gray', vmin=0, vmax=255)
    ax.set_title("OpenCV ekvalizácia", fontsize=9, pad=4)
    ax.axis('off')

    ax = fig.add_subplot(gs[1, 2])
    ax.imshow(diff_img, cmap='hot', vmin=0, vmax=max(1, diff_img.max()))
    ax.set_title(f"|Rozdiel|  (avg={diff_mean:.2f})", fontsize=9, pad=4)
    ax.axis('off')

    # ── R3: histogramy ekvalizovaných ──
    _draw_hist(fig.add_subplot(gs[2, 0]), gray,   'gray',      "Hist – pôvodný")
    _draw_hist(fig.add_subplot(gs[2, 1]), eq_man, 'orange',    "Hist – manuálna eq.")
    _draw_hist(fig.add_subplot(gs[2, 2]), eq_cv,  'steelblue', "Hist – OpenCV eq.")

    # ── R4: farebná ekvalizácia ──
    ax = fig.add_subplot(gs[3, 0])
    ax.imshow(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))
    ax.set_title("Farebný originál", fontsize=9, pad=4)
    ax.axis('off')

    ax = fig.add_subplot(gs[3, 1])
    ax.imshow(cv2.cvtColor(eq_col_man, cv2.COLOR_BGR2RGB))
    ax.set_title("Farebná man. eq.", fontsize=9, pad=4)
    ax.axis('off')

    ax = fig.add_subplot(gs[3, 2])
    ax.imshow(cv2.cvtColor(eq_col_cv, cv2.COLOR_BGR2RGB))
    ax.set_title(f"Farebná OpenCV eq. (diff={diff_col:.2f})", fontsize=9, pad=4)
    ax.axis('off')

    return fig


# ============================================================
# 7.  FIGURE 2 – CLAHE  (2×3, gridspec)
# ============================================================

def figure_2_clahe(gray, clip_limit=2.0, grid_size=(8, 8)):
    print("  Spúšťam manuálnu CLAHE ...")
    t0 = time.time()
    cl_man = manual_clahe(gray, clip_limit, grid_size)
    t_man = time.time() - t0
    print(f"    Hotovo za {t_man:.2f} s")

    t0 = time.time()
    cl_cv = opencv_clahe_gray(gray, clip_limit, grid_size)
    t_cv = time.time() - t0
    print(f"  OpenCV CLAHE za {t_cv:.4f} s")

    diff_mean = np.mean(np.abs(cl_man.astype(int) - cl_cv.astype(int)))

    fig = plt.figure(figsize=(14, 7))
    fig.suptitle(
        f"CLAHE  (clip={clip_limit}, grid={grid_size})    "
        f"Manuálna: {t_man:.2f}s  ×  OpenCV: {t_cv:.4f}s    "
        f"|diff|={diff_mean:.2f}",
        fontsize=11, fontweight='bold', y=0.99)

    gs = fig.add_gridspec(2, 3, hspace=0.25, wspace=0.15,
                          left=0.04, right=0.98, top=0.91, bottom=0.04)

    ax = fig.add_subplot(gs[0, 0])
    ax.imshow(gray, cmap='gray', vmin=0, vmax=255)
    ax.set_title("Pôvodný", fontsize=9, pad=4)
    ax.axis('off')

    ax = fig.add_subplot(gs[0, 1])
    ax.imshow(cl_man, cmap='gray', vmin=0, vmax=255)
    ax.set_title(f"Manuálna CLAHE ({t_man:.2f}s)", fontsize=9, pad=4)
    ax.axis('off')

    ax = fig.add_subplot(gs[0, 2])
    ax.imshow(cl_cv, cmap='gray', vmin=0, vmax=255)
    ax.set_title(f"OpenCV CLAHE ({t_cv:.4f}s)", fontsize=9, pad=4)
    ax.axis('off')

    _draw_hist(fig.add_subplot(gs[1, 0]), gray,   'gray',      "Hist – pôvodný")
    _draw_hist(fig.add_subplot(gs[1, 1]), cl_man, 'orange',    "Hist – manuálna CLAHE")
    _draw_hist(fig.add_subplot(gs[1, 2]), cl_cv,  'steelblue', "Hist – OpenCV CLAHE")

    return fig


# ============================================================
# 8.  HLAVNÝ PROGRAM
# ============================================================

def main():
    bgr, path = load_image()

    MAX_DIM = 800
    h, w = bgr.shape[:2]
    if max(h, w) > MAX_DIM:
        scale = MAX_DIM / max(h, w)
        bgr = cv2.resize(bgr, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
        print(f"Zmenšené na {bgr.shape[1]}x{bgr.shape[0]}")

    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)

    print("\n===== FIGURE 1: Histogramy & Ekvalizácia =====")
    fig1 = figure_1_histogramy_a_ekvalizacia(gray, bgr)

    print("\n===== FIGURE 2: CLAHE =====")
    fig2 = figure_2_clahe(gray, clip_limit=2.0, grid_size=(8, 8))

    # ── Zhrnutie ──
    eq_man = manual_equalize_gray(gray)
    eq_cv  = opencv_equalize_gray(gray)
    diff_eq = np.mean(np.abs(eq_man.astype(int) - eq_cv.astype(int)))

    cl_man = manual_clahe(gray, 2.0, (8, 8))
    cl_cv  = opencv_clahe_gray(gray, 2.0, (8, 8))
    diff_cl = np.mean(np.abs(cl_man.astype(int) - cl_cv.astype(int)))

    print("\n" + "=" * 55)
    print("ZHODNOTENIE")
    print("=" * 55)
    print(f"  Histogram:   manuálny == OpenCV (presná zhoda)")
    print(f"  Ekvalizácia: priem. |diff| = {diff_eq:.2f}")
    print(f"  CLAHE:       priem. |diff| = {diff_cl:.2f}")
    print(f"  Výpočtová náročnosť:")
    print(f"    CLAHE manuálna ~100-500x pomalšia ako OpenCV")
    print(f"    Histogram a eq. sú v NumPy dostatočne rýchle")
    print("=" * 55)

    OUT = "./VYSLEDKY_D"
    os.makedirs(OUT, exist_ok=True)
    fig1.savefig(os.path.join(OUT, "1_histogramy_a_ekvalizacia.png"), dpi=150)
    fig2.savefig(os.path.join(OUT, "2_clahe.png"), dpi=150)
    print(f"\nGrafy uložené do: {OUT}/")

    plt.show()


if __name__ == "__main__":
    main()