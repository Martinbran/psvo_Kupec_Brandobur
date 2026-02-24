import cv2
import numpy as np
import os

IMG_DIR = "./FOTKY_TEST"
RESIZE_TO = (240, 240)  # (W, H)
OUT_PATH = os.path.join(IMG_DIR, "x_mosaic_final.png")

def rotate90_clockwise_for(src: np.ndarray) -> np.ndarray:
    # src musí byť (H,W,C)
    h, w = src.shape[:2]
    ch = src.shape[2]
    dst = np.zeros((w, h, ch), dtype=src.dtype)  # po otočení (W,H,C)

    for y in range(h):
        for x in range(w):
            dst[x, h - 1 - y, :] = src[y, x, :]

    return dst

def print_image_info(img: np.ndarray) -> None:
    print("dtype:", img.dtype)
    print("shape:", img.shape)
    print("size (elements):", img.size)
    print("bytes:", img.nbytes)

def ensure_3ch(img: np.ndarray) -> np.ndarray:
    # ak je grayscale, spravíme 3-kanál
    if img is None:
        return None
    if img.ndim == 2:
        return cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    if img.ndim == 3 and img.shape[2] == 4:
        return img[:, :, :3]  # drop alpha
    return img

def list_images(folder: str):
    exts = (".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".webp")
    files = [f for f in os.listdir(folder) if f.lower().endswith(exts)]
    files.sort()  # stabilné poradie (abecedne)
    return [os.path.join(folder, f) for f in files]

def main():
    if not os.path.isdir(IMG_DIR):
        raise FileNotFoundError(f"Priečinok neexistuje: {IMG_DIR}")

    paths = list_images(IMG_DIR)
    if len(paths) < 4:
        raise RuntimeError(f"V priečinku je len {len(paths)} obrázkov, potrebujem aspoň 4.")

    # zoberieme prvé 4 (podľa sort)
    paths = paths[:4]
    print("Použité obrázky:")
    for p in paths:
        print(" -", p)

    captured = []
    for p in paths:
        img = cv2.imread(p, cv2.IMREAD_UNCHANGED)
        img = ensure_3ch(img)
        if img is None:
            raise RuntimeError(f"Nepodarilo sa načítať: {p}")

        img = cv2.resize(img, RESIZE_TO, interpolation=cv2.INTER_AREA)
        captured.append(img)

    # 2x2 mozaika
    h, w = captured[0].shape[:2]
    mosaic = np.zeros((2*h, 2*w, 3), dtype=captured[0].dtype)

    mosaic[0:h,   0:w]   = captured[0]
    mosaic[0:h,   w:2*w] = captured[1]
    mosaic[h:2*h, 0:w]   = captured[2]
    mosaic[h:2*h, w:2*w] = captured[3]

    # selektory casti mozaiky
    part1 = mosaic[0:h,   0:w]
    part2 = mosaic[0:h,   w:2*w]
    part3 = mosaic[h:2*h, 0:w]

    # 1) sharpen na cast 1
    kernel = np.array([[0, -1,  0],
                       [-1, 5, -1],
                       [0, -1,  0]], dtype=np.float32)
    part1[:, :] = cv2.filter2D(part1, -1, kernel, borderType=cv2.BORDER_REPLICATE)

    # 2) rotacia cast 2 o 90° doprava cez for-cyklus
    rotated = rotate90_clockwise_for(part2)
    # rotated má tvar (W,H,3), ale part2 je (H,W,3) -> RESIZE_TO je štvorcové,
    # takže H==W a sedí to. Pre istotu:
    if rotated.shape != part2.shape:
        rotated = cv2.resize(rotated, (part2.shape[1], part2.shape[0]), interpolation=cv2.INTER_NEAREST)
    part2[:, :] = rotated

    # 3) cast 3: nechaj iba cerveny kanal (OpenCV BGR -> R je index 2)
    part3[:, :, 0] = 0  # B
    part3[:, :, 1] = 0  # G
    #part3[:, :, 2] = 0  # R

    print("\nINFO o vyslednej mozaike:")
    print_image_info(mosaic)

    cv2.imwrite(OUT_PATH, mosaic)
    print("Saved:", OUT_PATH)

    cv2.namedWindow("Mosaic", cv2.WINDOW_NORMAL)
    cv2.imshow("Mosaic", mosaic)
    print("Stlac q v okne Mosaic na koniec.")

    while True:
        if (cv2.waitKey(10) & 0xFF) == ord('q'):
            break

    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
