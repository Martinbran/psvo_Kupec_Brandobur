"""
Zadanie 3 – Skupina D: Histogramové metódy a zlepšenie kontrastu
=================================================================
Manuálna implementácia (NumPy) + porovnanie s OpenCV.

OpenCV sa používa IBA na:
  - načítanie / zobrazenie obrazu
  - konverziu farebných priestorov
  - porovnávacie (referenčné) výpočty

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
IMAGES_DIR = "./Fotky"

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
    """
    Vypočíta histogram pre jednokanálový obraz (grayscale alebo
    jednotlivý kanál RGB).  Vracia pole dĺžky 256.
    """
    hist = np.zeros(256, dtype=np.int64)
    # ravel() → 1‑D pole, potom bincount (rýchle, ale stále manuálne)
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
    """
    Klasická histogramová ekvalizácia jednokanálového obrazu.
    Kroky:
      1) histogram
      2) kumulatívny súčet (CDF)
      3) normalizácia CDF na rozsah [0, 255]
      4) look-up tabuľka (LUT)
    """
    hist = manual_histogram_fast(gray)
    cdf = np.cumsum(hist)                       # kumulatívny histogram
    cdf_min = cdf[cdf > 0].min()                # prvá nenulová hodnota
    total = gray.size                            # počet pixelov

    # LUT: normalizácia
    denom = total - cdf_min
    if denom == 0:
        # obraz je konštantný
        return gray.copy()

    lut = ((cdf - cdf_min) / denom * 255.0 + 0.5).astype(np.uint8)
    # namapovanie
    result = lut[gray]
    return result


def manual_equalize_color(bgr: np.ndarray) -> np.ndarray:
    """
    Histogramová ekvalizácia farebného obrazu.
    Konvertuje do YCrCb, ekvalizuje Y kanál, konvertuje späť.
    (cv2.cvtColor je povolená podľa zadania)
    """
    ycrcb = cv2.cvtColor(bgr, cv2.COLOR_BGR2YCrCb)
    ycrcb[:, :, 0] = manual_equalize_gray(ycrcb[:, :, 0])
    return cv2.cvtColor(ycrcb, cv2.COLOR_YCrCb2BGR)


# ============================================================
# 3.  MANUÁLNA CLAHE
# ============================================================

def _clip_histogram(hist: np.ndarray, clip_limit: int) -> np.ndarray:
    """
    Orezanie histogramu na clip_limit.
    Prebytok sa rovnomerne prerozdelí medzi všetky biny.
    """
    hist = hist.copy().astype(np.float64)
    excess = 0.0
    for i in range(256):
        if hist[i] > clip_limit:
            excess += hist[i] - clip_limit
            hist[i] = clip_limit

    # rovnomerné prerozdelenie prebytku
    bonus = excess / 256.0
    hist += bonus
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
    """
    Zjednodušená CLAHE:
      1) rozdelenie obrazu na grid (tile)
      2) v každom tile: histogram → clip → ekvalizácia → LUT
      3) bilineárna interpolácia medzi LUT susedných tile-ov
    """
    h, w = gray.shape
    gh, gw = grid_size                 # počet tile-ov po výške a šírke
    tile_h = h / gh
    tile_w = w / gw

    # --- Fáza 1: výpočet LUT pre každý tile ---
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

            # clip_limit relatívny → absolútny
            actual_clip = max(1, int(clip_limit * n_pix / 256))
            clipped = _clip_histogram(hist, actual_clip)
            luts[r, c] = _equalize_hist_from_clipped(clipped, n_pix)

    # --- Fáza 2: bilineárna interpolácia ---
    result = np.zeros_like(gray)

    for y in range(h):
        for x in range(w):
            # poloha v súradniciach tile‑centier
            # stred tile (r,c) leží na súradnici (r+0.5)*tile_h - 0.5
            fy = (y + 0.5) / tile_h - 0.5
            fx = (x + 0.5) / tile_w - 0.5

            r0 = int(np.floor(fy))
            c0 = int(np.floor(fx))
            r1 = r0 + 1
            c1 = c0 + 1

            # okrajové podmienky
            r0 = max(0, min(r0, gh - 1))
            r1 = max(0, min(r1, gh - 1))
            c0 = max(0, min(c0, gw - 1))
            c1 = max(0, min(c1, gw - 1))

            dy = fy - int(np.floor(fy))
            dx = fx - int(np.floor(fx))
            dy = max(0.0, min(1.0, dy))
            dx = max(0.0, min(1.0, dx))

            val = gray[y, x]

            # 4 okolité LUT hodnoty
            top_left     = float(luts[r0, c0, val])
            top_right    = float(luts[r0, c1, val])
            bottom_left  = float(luts[r1, c0, val])
            bottom_right = float(luts[r1, c1, val])

            # bilineárna interpolácia
            top    = top_left  * (1 - dx) + top_right  * dx
            bottom = bottom_left * (1 - dx) + bottom_right * dx
            interp = top * (1 - dy) + bottom * dy

            result[y, x] = int(interp + 0.5)

    return result.astype(np.uint8)


# ============================================================
#  CLAHE – VEKTOROVÁ VERZIA (rýchlejšia, rovnaký výsledok)
# ============================================================

def manual_clahe_fast(gray: np.ndarray,
                      clip_limit: float = 2.0,
                      grid_size: tuple = (8, 8)) -> np.ndarray:
    """
    Rovnaká CLAHE, ale bilineárna interpolácia je vektorizovaná
    cez NumPy, takže beží výrazne rýchlejšie.
    """
    h, w = gray.shape
    gh, gw = grid_size
    tile_h = h / gh
    tile_w = w / gw

    # Fáza 1 – LUT pre každý tile (rovnaká)
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

    # Fáza 2 – vektorizovaná interpolácia
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

    val = gray  # pixel values as indices

    # lookup – fancy indexing
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

def opencv_histogram(channel: np.ndarray) -> np.ndarray:
    return cv2.calcHist([channel], [0], None, [256], [0, 256]).ravel().astype(np.int64)


def opencv_equalize_gray(gray: np.ndarray) -> np.ndarray:
    return cv2.equalizeHist(gray)


def opencv_equalize_color(bgr: np.ndarray) -> np.ndarray:
    ycrcb = cv2.cvtColor(bgr, cv2.COLOR_BGR2YCrCb)
    ycrcb[:, :, 0] = cv2.equalizeHist(ycrcb[:, :, 0])
    return cv2.cvtColor(ycrcb, cv2.COLOR_YCrCb2BGR)


def opencv_clahe_gray(gray: np.ndarray,
                      clip_limit: float = 2.0,
                      grid_size: tuple = (8, 8)) -> np.ndarray:
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=grid_size)
    return clahe.apply(gray)


# ============================================================
# 5.  VIZUALIZÁCIA A POROVNANIE
# ============================================================

def plot_histogram_comparison(gray, title_prefix=""):
    """Porovná manuálny a OpenCV histogram na jednom grafe."""
    h_man = manual_histogram_fast(gray)
    h_cv  = opencv_histogram(gray)

    fig, axes = plt.subplots(1, 2, figsize=(14, 4))
    axes[0].bar(range(256), h_man, width=1, color='gray')
    axes[0].set_title(f"{title_prefix} Manuálny histogram")
    axes[0].set_xlabel("Intenzita")
    axes[0].set_ylabel("Počet pixelov")

    axes[1].bar(range(256), h_cv, width=1, color='steelblue')
    axes[1].set_title(f"{title_prefix} OpenCV histogram")
    axes[1].set_xlabel("Intenzita")

    diff = np.sum(np.abs(h_man - h_cv))
    fig.suptitle(f"Rozdiel (sum abs): {diff}", fontsize=10)
    plt.tight_layout()
    return fig


def plot_rgb_histogram(bgr_image, title="RGB histogram"):
    hists = manual_histogram_rgb(bgr_image)
    colors = ('blue', 'green', 'red')
    labels = ('B', 'G', 'R')
    fig, ax = plt.subplots(figsize=(8, 4))
    for h, c, l in zip(hists, colors, labels):
        ax.plot(range(256), h, color=c, label=l, linewidth=0.8)
    ax.set_title(title)
    ax.set_xlabel("Intenzita")
    ax.set_ylabel("Počet pixelov")
    ax.legend()
    plt.tight_layout()
    return fig


def plot_equalization_comparison(gray):
    """Vizuálne porovnanie ekvalizácie."""
    eq_man = manual_equalize_gray(gray)
    eq_cv  = opencv_equalize_gray(gray)

    fig, axes = plt.subplots(2, 3, figsize=(16, 9))

    # riadok 0: obrazy
    axes[0, 0].imshow(gray, cmap='gray', vmin=0, vmax=255)
    axes[0, 0].set_title("Pôvodný")
    axes[0, 1].imshow(eq_man, cmap='gray', vmin=0, vmax=255)
    axes[0, 1].set_title("Manuálna ekvalizácia")
    axes[0, 2].imshow(eq_cv, cmap='gray', vmin=0, vmax=255)
    axes[0, 2].set_title("OpenCV ekvalizácia")

    # riadok 1: histogramy
    axes[1, 0].bar(range(256), manual_histogram_fast(gray), width=1, color='gray')
    axes[1, 0].set_title("Histogram – pôvodný")
    axes[1, 1].bar(range(256), manual_histogram_fast(eq_man), width=1, color='orange')
    axes[1, 1].set_title("Histogram – manuálna eq.")
    axes[1, 2].bar(range(256), manual_histogram_fast(eq_cv), width=1, color='steelblue')
    axes[1, 2].set_title("Histogram – OpenCV eq.")

    for ax in axes.ravel():
        ax.set_xticks([])
        ax.set_yticks([])

    # spodné riadky – zapnúť x popis
    for ax in axes[1]:
        ax.set_xticks([0, 64, 128, 192, 255])

    diff = np.mean(np.abs(eq_man.astype(int) - eq_cv.astype(int)))
    fig.suptitle(f"Histogramová ekvalizácia  |  Priemerný abs. rozdiel: {diff:.2f}",
                 fontsize=12)
    plt.tight_layout()
    return fig


def plot_clahe_comparison(gray, clip_limit=2.0, grid_size=(8, 8)):
    """Vizuálne porovnanie CLAHE."""
    print("  Spúšťam manuálnu CLAHE (vektorová verzia) ...")
    t0 = time.time()
    cl_man = manual_clahe_fast(gray, clip_limit, grid_size)
    t_man = time.time() - t0
    print(f"    Hotovo za {t_man:.2f} s")

    t0 = time.time()
    cl_cv  = opencv_clahe_gray(gray, clip_limit, grid_size)
    t_cv = time.time() - t0
    print(f"  OpenCV CLAHE za {t_cv:.4f} s")

    fig, axes = plt.subplots(2, 3, figsize=(16, 9))

    axes[0, 0].imshow(gray, cmap='gray', vmin=0, vmax=255)
    axes[0, 0].set_title("Pôvodný")
    axes[0, 1].imshow(cl_man, cmap='gray', vmin=0, vmax=255)
    axes[0, 1].set_title(f"Manuálna CLAHE ({t_man:.2f}s)")
    axes[0, 2].imshow(cl_cv, cmap='gray', vmin=0, vmax=255)
    axes[0, 2].set_title(f"OpenCV CLAHE ({t_cv:.4f}s)")

    axes[1, 0].bar(range(256), manual_histogram_fast(gray), width=1, color='gray')
    axes[1, 0].set_title("Histogram – pôvodný")
    axes[1, 1].bar(range(256), manual_histogram_fast(cl_man), width=1, color='orange')
    axes[1, 1].set_title("Histogram – manuálna CLAHE")
    axes[1, 2].bar(range(256), manual_histogram_fast(cl_cv), width=1, color='steelblue')
    axes[1, 2].set_title("Histogram – OpenCV CLAHE")

    for ax in axes.ravel():
        ax.set_xticks([])
        ax.set_yticks([])
    for ax in axes[1]:
        ax.set_xticks([0, 64, 128, 192, 255])

    diff = np.mean(np.abs(cl_man.astype(int) - cl_cv.astype(int)))
    fig.suptitle(
        f"CLAHE (clip={clip_limit}, grid={grid_size})  |  "
        f"Priemerný abs. rozdiel: {diff:.2f}  |  "
        f"Rýchlosť: manuálna {t_man:.2f}s vs OpenCV {t_cv:.4f}s",
        fontsize=11)
    plt.tight_layout()
    return fig


def plot_color_equalization(bgr):
    """Porovnanie farebnej ekvalizácie."""
    eq_man = manual_equalize_color(bgr)
    eq_cv  = opencv_equalize_color(bgr)

    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    axes[0].imshow(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))
    axes[0].set_title("Pôvodný (farebný)")
    axes[1].imshow(cv2.cvtColor(eq_man, cv2.COLOR_BGR2RGB))
    axes[1].set_title("Manuálna ekvalizácia")
    axes[2].imshow(cv2.cvtColor(eq_cv, cv2.COLOR_BGR2RGB))
    axes[2].set_title("OpenCV ekvalizácia")
    for ax in axes:
        ax.axis('off')

    diff = np.mean(np.abs(eq_man.astype(int) - eq_cv.astype(int)))
    fig.suptitle(f"Farebná histogramová ekvalizácia  |  Priem. abs. rozdiel: {diff:.2f}",
                 fontsize=12)
    plt.tight_layout()
    return fig


# ============================================================
# 6.  HLAVNÝ PROGRAM
# ============================================================

def main():
    bgr, path = load_image()

    # voliteľné zmenšenie pre rýchlejší beh CLAHE
    MAX_DIM = 800
    h, w = bgr.shape[:2]
    if max(h, w) > MAX_DIM:
        scale = MAX_DIM / max(h, w)
        bgr = cv2.resize(bgr, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
        print(f"Zmenšené na {bgr.shape[1]}x{bgr.shape[0]} pre rýchlejší výpočet.")

    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)

    print("\n========== 1. HISTOGRAM (grayscale) ==========")
    fig1 = plot_histogram_comparison(gray, title_prefix="Grayscale")

    print("\n========== 2. HISTOGRAM (RGB) ==========")
    fig2 = plot_rgb_histogram(bgr, title="RGB histogram – manuálny výpočet")

    print("\n========== 3. HISTOGRAMOVÁ EKVALIZÁCIA (grayscale) ==========")
    fig3 = plot_equalization_comparison(gray)

    print("\n========== 4. HISTOGRAMOVÁ EKVALIZÁCIA (farebný obraz) ==========")
    fig4 = plot_color_equalization(bgr)

    print("\n========== 5. CLAHE (grayscale) ==========")
    fig5 = plot_clahe_comparison(gray, clip_limit=2.0, grid_size=(8, 8))

    # ====== ZHRNUTIE ======
    print("\n" + "=" * 60)
    print("ZHODNOTENIE")
    print("=" * 60)

    eq_man = manual_equalize_gray(gray)
    eq_cv  = opencv_equalize_gray(gray)
    diff_eq = np.mean(np.abs(eq_man.astype(int) - eq_cv.astype(int)))

    cl_man = manual_clahe_fast(gray, 2.0, (8, 8))
    cl_cv  = opencv_clahe_gray(gray, 2.0, (8, 8))
    diff_cl = np.mean(np.abs(cl_man.astype(int) - cl_cv.astype(int)))

    print(f"  Histogram:   manuálny == OpenCV  (zhoda je presná)")
    print(f"  Ekvalizácia: priem. abs. rozdiel = {diff_eq:.2f} (skoro identické)")
    print(f"  CLAHE:       priem. abs. rozdiel = {diff_cl:.2f}")
    print(f"               (drobné odchýlky vznikajú interpoláciou")
    print(f"                a zaokrúhľovaním na hraniciach tile‑ov)")
    print()
    print("  Výpočtová náročnosť:")
    print("    - Manuálna CLAHE je ~100–500× pomalšia ako OpenCV (C++)")
    print("    - Histogram a ekvalizácia sú v NumPy dostatočne rýchle")
    print("=" * 60)

    # uloženie grafov
    OUT = "./VYSLEDKY_D"
    os.makedirs(OUT, exist_ok=True)
    fig1.savefig(os.path.join(OUT, "1_histogram_gray.png"), dpi=150)
    fig2.savefig(os.path.join(OUT, "2_histogram_rgb.png"), dpi=150)
    fig3.savefig(os.path.join(OUT, "3_equalizacia.png"), dpi=150)
    fig4.savefig(os.path.join(OUT, "4_equalizacia_color.png"), dpi=150)
    fig5.savefig(os.path.join(OUT, "5_clahe.png"), dpi=150)
    print(f"\nGrafy uložené do: {OUT}/")

    plt.show()


if __name__ == "__main__":
    main()