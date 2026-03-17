import cv2
import numpy as np
import os
import glob

CHESSBOARD_SIZE = (7, 5)


SQUARE_SIZE = 0.024

IMAGES_DIR = "./FOTKY_SACHOVNICA_bez_resize"   
OUT_DIR = "./CALIB_OUT5"
os.makedirs(OUT_DIR, exist_ok=True)

objp = np.zeros((CHESSBOARD_SIZE[0] * CHESSBOARD_SIZE[1], 3), np.float32)
objp[:, :2] = np.mgrid[0:CHESSBOARD_SIZE[0], 0:CHESSBOARD_SIZE[1]].T.reshape(-1, 2)
objp *= float(SQUARE_SIZE)

objpoints = []  
imgpoints = []  

imgs = []
for ext in ("*.png", "*.jpg", "*.jpeg", "*.bmp", "*.tif", "*.tiff"):
    imgs += glob.glob(os.path.join(IMAGES_DIR, ext))
imgs = sorted(imgs)

if len(imgs) == 0:
    raise RuntimeError(f"Nenašiel som žiadne obrázky v: {IMAGES_DIR}")

print(f"Našiel som {len(imgs)} obrázkov.")

img_size = None
good = 0
bad = 0

criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
flags = cv2.CALIB_CB_ADAPTIVE_THRESH | cv2.CALIB_CB_NORMALIZE_IMAGE

for path in imgs:
    img = cv2.imread(path)
    if img is None:
        print("SKIP (neviem načítať):", path)
        bad += 1
        continue

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    if img_size is None:
        img_size = (gray.shape[1], gray.shape[0])  # (w,h)

    found, corners = cv2.findChessboardCorners(gray, CHESSBOARD_SIZE, flags)

    if not found:
        print("NOT FOUND:", os.path.basename(path))
        bad += 1
        continue

    corners2 = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)

    objpoints.append(objp.copy())
    imgpoints.append(corners2)

    vis = img.copy()
    cv2.drawChessboardCorners(vis, CHESSBOARD_SIZE, corners2, found)
    cv2.imwrite(os.path.join(OUT_DIR, "corners_" + os.path.basename(path)), vis)

    print("OK:", os.path.basename(path))
    good += 1

print(f"\nPoužiteľné snímky: {good}, neúspešné: {bad}")
if good < 3:
    raise RuntimeError("Na kalibráciu potrebuješ aspoň 3 úspešné snímky (ideálne 10-20).")

# ====== KALIBRÁCIA ======
ret, K, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, img_size, None, None)

fx, fy = float(K[0, 0]), float(K[1, 1])
cx, cy = float(K[0, 2]), float(K[1, 2])

print("\n===== VÝSLEDKY =====")
print("RMS reprojection error:", ret)
print("Camera matrix K:\n", K)
print("Distortion coeffs:\n", dist.ravel())
print(f"fx={fx:.3f}, fy={fy:.3f}, cx={cx:.3f}, cy={cy:.3f}")

np.savez(os.path.join(OUT_DIR, "calibration.npz"),
         K=K, dist=dist, img_size=np.array(img_size))

fs = cv2.FileStorage(os.path.join(OUT_DIR, "calibration.yaml"), cv2.FILE_STORAGE_WRITE)
fs.write("image_width", int(img_size[0]))
fs.write("image_height", int(img_size[1]))
fs.write("camera_matrix", K)
fs.write("distortion_coefficients", dist)
fs.release()

print("\nUložené:")
print(" -", os.path.join(OUT_DIR, "calibration.npz"))
print(" -", os.path.join(OUT_DIR, "calibration.yaml"))

# ====== UNDISTORT ====== 
sample_path = imgs[0]
img = cv2.imread(sample_path)
und = cv2.undistort(img, K, dist)

combo = np.hstack([img, und])
out_demo = os.path.join(OUT_DIR, "undistort_demo.png")
cv2.imwrite(out_demo, combo)
print("Undistortion demo uložené:", out_demo)

cv2.namedWindow("Undistort demo (orig | undist)", cv2.WINDOW_NORMAL)
cv2.imshow("Undistort demo (orig | undist)", combo)
cv2.waitKey(0)
cv2.destroyAllWindows()