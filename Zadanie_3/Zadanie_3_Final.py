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
IMAGES_DIR = "./Fotky/Farebne_jablka"

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
# 5.  POMOCNÉ FUNKCIE
# ============================================================

def _gaussian_curve(data: np.ndarray) -> tuple:
    """Vráti (x, y) fitovanej Gaussovej krivky škálovanej na veľkosť histogramu."""
    flat  = data.ravel().astype(np.float64)
    mu    = flat.mean()
    sigma = flat.std()
    if sigma < 1e-6:
        return np.array([]), np.array([])
    x = np.arange(256, dtype=np.float64)
    y = (flat.size / (sigma * np.sqrt(2 * np.pi))) * np.exp(-0.5 * ((x - mu) / sigma) ** 2)
    return x, y


def _draw_hist(ax, data, color='gray', title="", gauss=False):
    """Vykreslí histogram do subplotu. gauss=True pridá fitovanú Gaussovu krivku."""
    h = manual_histogram_fast(data)
    ax.fill_between(range(256), h, color=color, alpha=0.7)
    if gauss:
        gx, gy = _gaussian_curve(data)
        if len(gx):
            mu    = data.ravel().astype(np.float64).mean()
            sigma = data.ravel().astype(np.float64).std()
            ax.plot(gx, gy, color='black', linewidth=1.4, linestyle='--', label='Gauss fit')
            ax.axvline(mu, color='black', linewidth=0.8, linestyle=':')
            ax.legend(fontsize=6, loc='upper right')
            ax.set_xlabel(f"μ={mu:.1f}  σ={sigma:.1f}", fontsize=6, labelpad=1)
    ax.set_xlim(0, 255)
    ax.set_title(title, fontsize=8, pad=3)
    ax.tick_params(labelsize=6)


# ============================================================
# 6.  FIGURE 1 – RGB Histogramy  (2×4, gridspec)
#     Originál  |  R kanál  |  G kanál  |  B kanál
#     Gray hist |  R hist   |  G hist   |  B hist
# ============================================================

def figure_1_rgb_histogramy(bgr):
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    r, g, b = rgb[:, :, 0], rgb[:, :, 1], rgb[:, :, 2]

    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)

    # R/G/B kanály zobrazené v ich skutočnej farbe
    r_img = np.zeros_like(rgb); r_img[:, :, 0] = r
    g_img = np.zeros_like(rgb); g_img[:, :, 1] = g
    b_img = np.zeros_like(rgb); b_img[:, :, 2] = b

    fig = plt.figure(figsize=(16, 7))
    fig.suptitle("RGB HISTOGRAMY – každý kanál samostatne",
                 fontsize=13, fontweight='bold', y=0.99)

    gs = fig.add_gridspec(2, 4, hspace=0.30, wspace=0.18,
                          left=0.04, right=0.98, top=0.93, bottom=0.04)

    # ── Riadok 0: obrazy ──
    for col, (img, title) in enumerate([
        (rgb,   "Originál (RGB)"),
        (r_img, "R kanál"),
        (g_img, "G kanál"),
        (b_img, "B kanál"),
    ]):
        ax = fig.add_subplot(gs[0, col])
        ax.imshow(img)
        ax.set_title(title, fontsize=9, pad=4)
        ax.axis('off')

    # ── Riadok 1: histogramy s Gaussovou krivkou ──
    _draw_hist(fig.add_subplot(gs[1, 0]), gray, 'dimgray',  "Grayscale histogram", gauss=True)

    for col, (channel, color, title) in enumerate([
        (r, '#cc2222', "Red (R) histogram"),
        (g, '#22aa22', "Green (G) histogram"),
        (b, '#2255cc', "Blue (B) histogram"),
    ], start=1):
        _draw_hist(fig.add_subplot(gs[1, col]), channel, color, title, gauss=True)

    return fig


# ============================================================
# 7.  FIGURE 2 – Grayscale Ekvalizácia  (2×4, gridspec)
#     Originál | Manuálna eq | OpenCV eq | |Rozdiel|
#     Hist orig| Hist man.   | Hist OCV  | (prázdny)
# ============================================================

def figure_2_ekvalizacia_gray(gray):
    eq_man   = manual_equalize_gray(gray)
    eq_cv    = opencv_equalize_gray(gray)
    diff_img = np.abs(eq_man.astype(int) - eq_cv.astype(int)).astype(np.uint8)
    diff_mean = np.mean(diff_img)

    h_man = manual_histogram_fast(gray)
    h_cv  = opencv_histogram(gray)
    hist_diff = int(np.sum(np.abs(h_man - h_cv)))

    fig = plt.figure(figsize=(16, 7))
    fig.suptitle("GRAYSCALE – HISTOGRAMOVÁ EKVALIZÁCIA",
                 fontsize=13, fontweight='bold', y=0.99)

    gs = fig.add_gridspec(2, 4, hspace=0.30, wspace=0.18,
                          left=0.04, right=0.98, top=0.93, bottom=0.04)

    # ── Riadok 0: obrazy ──
    for col, (img, cmap, title) in enumerate([
        (gray,     'gray', "Pôvodný (gray)"),
        (eq_man,   'gray', "Manuálna ekvalizácia"),
        (eq_cv,    'gray', "OpenCV ekvalizácia"),
        (diff_img, 'hot',  f"|Rozdiel|  avg={diff_mean:.2f}"),
    ]):
        ax = fig.add_subplot(gs[0, col])
        ax.imshow(img, cmap=cmap, vmin=0, vmax=max(1, img.max()) if cmap == 'hot' else 255)
        ax.set_title(title, fontsize=9, pad=4)
        ax.axis('off')

    # ── Riadok 1: histogramy ──
    _draw_hist(fig.add_subplot(gs[1, 0]), gray,   'dimgray',    "Hist – pôvodný",      gauss=True)
    _draw_hist(fig.add_subplot(gs[1, 1]), eq_man, 'darkorange', "Hist – manuálna eq.")
    _draw_hist(fig.add_subplot(gs[1, 2]), eq_cv,  'steelblue',  "Hist – OpenCV eq.")

    # Posledný subplot: porovnanie histogramov orig (man vs cv)
    ax = fig.add_subplot(gs[1, 3])
    ax.fill_between(range(256), h_man, color='dimgray',   alpha=0.6, label='Manuálny')
    ax.fill_between(range(256), h_cv,  color='steelblue', alpha=0.4, label='OpenCV')
    ax.set_xlim(0, 255)
    ax.set_title(f"Man. vs OpenCV hist  (Δ={hist_diff})", fontsize=8, pad=3)
    ax.legend(fontsize=6, loc='upper right')
    ax.tick_params(labelsize=6)

    return fig


# ============================================================
# 8.  FIGURE 3 – Farebná Ekvalizácia  (2×3, gridspec)
#     Originál  | Man. eq (farba) | OpenCV eq (farba)
#     R/G/B hist orig | R/G/B hist man | R/G/B hist cv
# ============================================================

def figure_3_ekvalizacia_farba(bgr):
    eq_col_man = manual_equalize_color(bgr)
    eq_col_cv  = opencv_equalize_color(bgr)
    diff_col   = np.mean(np.abs(eq_col_man.astype(int) - eq_col_cv.astype(int)))

    def rgb_hists(img_bgr):
        rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        return [manual_histogram_fast(rgb[:, :, i]) for i in range(3)]

    fig = plt.figure(figsize=(14, 8))
    fig.suptitle(f"FAREBNÁ EKVALIZÁCIA  (manuálna vs OpenCV  |diff|={diff_col:.2f})",
                 fontsize=13, fontweight='bold', y=0.99)

    gs = fig.add_gridspec(2, 3, hspace=0.35, wspace=0.18,
                          left=0.04, right=0.98, top=0.93, bottom=0.04)

    # ── Riadok 0: farebné obrazy ──
    for col, (img_bgr, title) in enumerate([
        (bgr,        "Farebný originál"),
        (eq_col_man, "Manuálna farebná eq."),
        (eq_col_cv,  "OpenCV farebná eq."),
    ]):
        ax = fig.add_subplot(gs[0, col])
        ax.imshow(cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB))
        ax.set_title(title, fontsize=9, pad=4)
        ax.axis('off')

    # ── Riadok 1: RGB histogramy pre každú verziu ──
    colors  = ('#cc2222', '#22aa22', '#2255cc')
    labels  = ('R', 'G', 'B')
    imgs    = [bgr, eq_col_man, eq_col_cv]
    subtitles = ["RGB hist – originál", "RGB hist – man. eq.", "RGB hist – OpenCV eq."]

    for col, (img_bgr, subtitle) in enumerate(zip(imgs, subtitles)):
        ax = fig.add_subplot(gs[1, col])
        hists = rgb_hists(img_bgr)
        for h, c, l in zip(hists, colors, labels):
            ax.fill_between(range(256), h, color=c, alpha=0.35, label=l)
            ax.plot(range(256), h, color=c, linewidth=0.6)
        ax.set_xlim(0, 255)
        ax.set_title(subtitle, fontsize=8, pad=3)
        ax.legend(fontsize=6, loc='upper right')
        ax.tick_params(labelsize=6)

    return fig


# ============================================================
# 9.  FIGURE 4 – CLAHE  (2×3, gridspec)
# ============================================================

def figure_4_clahe(gray, clip_limit=2.0, grid_size=(8, 8)):
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
        f"Manuálna: {t_man:.2f}s  |  OpenCV: {t_cv:.4f}s    "
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

    _draw_hist(fig.add_subplot(gs[1, 0]), gray,   'dimgray',    "Hist – pôvodný",       gauss=True)
    _draw_hist(fig.add_subplot(gs[1, 1]), cl_man, 'darkorange', "Hist – manuálna CLAHE")
    _draw_hist(fig.add_subplot(gs[1, 2]), cl_cv,  'steelblue',  "Hist – OpenCV CLAHE")

    return fig, cl_man, cl_cv, t_man, t_cv


# ============================================================
# 10.  HLAVNÝ PROGRAM
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

    print("\n===== FIGURE 1: RGB Histogramy =====")
    fig1 = figure_1_rgb_histogramy(bgr)

    print("\n===== FIGURE 2: Grayscale Ekvalizácia =====")
    fig2 = figure_2_ekvalizacia_gray(gray)

    print("\n===== FIGURE 3: Farebná Ekvalizácia =====")
    fig3 = figure_3_ekvalizacia_farba(bgr)

    print("\n===== FIGURE 4: CLAHE =====")
    fig4, cl_man, cl_cv, t_cl_man, t_cl_cv = figure_4_clahe(gray, clip_limit=2.0, grid_size=(8, 8))

    # ── Zhrnutie ──
    t0 = time.time(); eq_man = manual_equalize_gray(gray); t_eq_man = time.time() - t0
    t0 = time.time(); eq_cv  = opencv_equalize_gray(gray); t_eq_cv  = time.time() - t0
    diff_eq = np.mean(np.abs(eq_man.astype(int) - eq_cv.astype(int)))
    diff_cl = np.mean(np.abs(cl_man.astype(int) - cl_cv.astype(int)))

    h_man = manual_histogram_fast(gray)
    h_cv  = opencv_histogram(gray)
    hist_diff = int(np.sum(np.abs(h_man - h_cv)))

    eq_ratio    = t_eq_man  / t_eq_cv  if t_eq_cv  > 0 else float('inf')
    clahe_ratio = t_cl_man  / t_cl_cv  if t_cl_cv  > 0 else float('inf')

    lines = [
        "=" * 60,
        "STRUČNÉ ZHODNOTENIE ROZDIELOV (Manuálna implementácia vs OpenCV)",
        "=" * 60,
        "",
        "--- PRESNOSŤ ---",
        f"  Histogram (gray):   súčet |diff| hodnôt = {hist_diff}  (0 = presná zhoda)",
        f"  Ekvalizácia (gray): priemerný |diff| pixelu = {diff_eq:.4f}",
        f"  CLAHE (gray):       priemerný |diff| pixelu = {diff_cl:.4f}",
        "",
        "--- ŠUM / ARTEFAKTY ---",
        "  Histogram:   žiadne rozdiely – bincount je deterministický.",
        f"  Ekvalizácia: odchýlka {diff_eq:.4f} pix je spôsobená zaokrúhľovaním",
        "               v LUT; vizuálne nerozoznateľné.",
        f"  CLAHE:       odchýlka {diff_cl:.4f} pix pochádza z iného spôsobu",
        "               interpolácie dlaždíc; jemné hrany sa môžu mierne líšiť.",
        "",
        "--- VÝPOČTOVÁ NÁROČNOSŤ ---",
        f"  Ekvalizácia – manuálna: {t_eq_man*1000:.2f} ms   OpenCV: {t_eq_cv*1000:.2f} ms   "
        f"(pomer {eq_ratio:.1f}×)",
        f"  CLAHE        – manuálna: {t_cl_man:.3f} s     OpenCV: {t_cl_cv*1000:.2f} ms   "
        f"(pomer {clahe_ratio:.0f}×)",
        "  Histogram a ekvalizácia sú v NumPy dostatočne rýchle.",
        "  Manuálna CLAHE je výrazne pomalšia kvôli Python-level slučkám",
        "  nad dlaždicami; OpenCV je implementovaná v C++.",
        "",
        "=" * 60,
    ]

    summary = "\n".join(lines)
    print("\n" + summary)

    ROOT_OUT = "./Vysledky"
    os.makedirs(ROOT_OUT, exist_ok=True)
    i = 1
    while os.path.exists(os.path.join(ROOT_OUT, f"Vysledky{i}")):
        i += 1
    OUT = os.path.join(ROOT_OUT, f"Vysledky{i}")
    os.makedirs(OUT)
    fig1.savefig(os.path.join(OUT, "1_rgb_histogramy.png"),    dpi=150)
    fig2.savefig(os.path.join(OUT, "2_ekvalizacia_gray.png"),  dpi=150)
    fig3.savefig(os.path.join(OUT, "3_ekvalizacia_farba.png"), dpi=150)
    fig4.savefig(os.path.join(OUT, "4_clahe.png"),             dpi=150)

    txt_path = os.path.join(OUT, "zhodnotenie.txt")
    with open(txt_path, "w", encoding="utf-8") as f:
        f.write(summary + "\n")
    print(f"\nGrafy a zhodnotenie uložené do: {OUT}/")

    plt.show()


if __name__ == "__main__":
    main()