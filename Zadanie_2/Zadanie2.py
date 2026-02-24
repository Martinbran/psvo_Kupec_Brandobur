#!/usr/bin/env python3
"""
Tento skript vie:
1) Získať kalibračné snímky šachovnice, vykonať kalibráciu (OpenCV),
   uložiť camera matrix + dist coef do .npz/.yaml a vypísať fx, fy, cx, cy.
2) V reálnom čase:
   - odstrániť distorziu (undistort),
   - detegovať základné geometrické tvary a vykresliť názov + ťažisko,
   - spraviť farebný filter (napr. červená -> zelená).
3) Bonus: Ak je v zábere šachovnica, odhadne rozmery objektov v cm (na rovine šachovnice).

Pozn.: Primárne je to pre Ximea (xiapi). Ak xiapi nie je dostupné, použije sa klasická webkamera (cv2.VideoCapture).
"""

from __future__ import annotations

import argparse
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Tuple, List, Dict

import numpy as np
import cv2

# -------------------------
# Kamera wrapper (Ximea / fallback)
# -------------------------

class FrameSource:
    def read(self) -> Optional[np.ndarray]:
        raise NotImplementedError

    def close(self) -> None:
        pass


class OpenCVCamera(FrameSource):
    def __init__(self, index: int = 0, width: Optional[int] = None, height: Optional[int] = None):
        self.cap = cv2.VideoCapture(index)
        if width is not None:
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, float(width))
        if height is not None:
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, float(height))

        if not self.cap.isOpened():
            raise RuntimeError("Nepodarilo sa otvoriť cv2.VideoCapture(). Skontroluj index kamery a prístup.")

    def read(self) -> Optional[np.ndarray]:
        ok, frame = self.cap.read()
        if not ok:
            return None
        return frame

    def close(self) -> None:
        try:
            self.cap.release()
        except Exception:
            pass


class XimeaCamera(FrameSource):
    def __init__(self, exposure_us: int = 20000, imgdataformat: str = "XI_RGB32"):
        try:
            from ximea import xiapi  # type: ignore
        except Exception as e:
            raise RuntimeError(f"xiapi nie je dostupné: {e}")

        self.xiapi = xiapi
        self.cam = xiapi.Camera()
        self.cam.open_device()
        self.cam.set_exposure(int(exposure_us))
        self.cam.set_param("imgdataformat", imgdataformat)
        self.cam.set_param("auto_wb", 1)
        self.img = xiapi.Image()
        self.cam.start_acquisition()

    def read(self) -> Optional[np.ndarray]:
        try:
            self.cam.get_image(self.img)
            frame = self.img.get_image_data_numpy()
            # XI_RGB32 -> BGR (zhod alpha)
            if frame.ndim == 3 and frame.shape[2] >= 3:
                frame = frame[:, :, :3]
            # xiapi dáva RGB, OpenCV chce BGR
            frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
            return frame
        except Exception:
            return None

    def close(self) -> None:
        try:
            self.cam.stop_acquisition()
            self.cam.close_device()
        except Exception:
            pass


def create_source(args) -> FrameSource:
    if args.source == "ximea":
        return XimeaCamera(exposure_us=args.exposure_us, imgdataformat=args.xi_format)
    else:
        return OpenCVCamera(index=args.cam_index, width=args.width, height=args.height)


# -------------------------
# Kalibrácia
# -------------------------

@dataclass
class CalibrationResult:
    camera_matrix: np.ndarray
    dist_coeffs: np.ndarray
    rvecs: List[np.ndarray]
    tvecs: List[np.ndarray]
    rms: float
    image_size: Tuple[int, int]  # (w, h)

    @property
    def fx(self) -> float:
        return float(self.camera_matrix[0, 0])

    @property
    def fy(self) -> float:
        return float(self.camera_matrix[1, 1])

    @property
    def cx(self) -> float:
        return float(self.camera_matrix[0, 2])

    @property
    def cy(self) -> float:
        return float(self.camera_matrix[1, 2])


def chessboard_object_points(pattern_size: Tuple[int, int], square_size_cm: float) -> np.ndarray:
    """
    Vytvorí 3D body šachovnice v rovine Z=0.
    pattern_size = (cols, rows) = počet vnútorných rohov.
    square_size_cm = veľkosť jedného štvorca v cm.
    """
    cols, rows = pattern_size
    objp = np.zeros((rows * cols, 3), np.float32)
    grid = np.mgrid[0:cols, 0:rows].T.reshape(-1, 2).astype(np.float32)
    objp[:, :2] = grid * float(square_size_cm)
    return objp


def find_corners(gray: np.ndarray, pattern_size: Tuple[int, int]) -> Tuple[bool, Optional[np.ndarray]]:
    flags = cv2.CALIB_CB_ADAPTIVE_THRESH + cv2.CALIB_CB_NORMALIZE_IMAGE
    ok, corners = cv2.findChessboardCorners(gray, pattern_size, flags)
    if not ok:
        return False, None

    # spresnenie
    term = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
    corners2 = cv2.cornerSubPix(
        gray, corners, winSize=(11, 11), zeroZone=(-1, -1), criteria=term
    )
    return True, corners2


def calibrate_from_images(
    image_paths: List[Path],
    pattern_size: Tuple[int, int],
    square_size_cm: float,
) -> CalibrationResult:
    objp = chessboard_object_points(pattern_size, square_size_cm)

    objpoints: List[np.ndarray] = []
    imgpoints: List[np.ndarray] = []
    img_size: Optional[Tuple[int, int]] = None

    for p in image_paths:
        img = cv2.imread(str(p))
        if img is None:
            continue
        h, w = img.shape[:2]
        img_size = (w, h)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        ok, corners = find_corners(gray, pattern_size)
        if ok and corners is not None:
            objpoints.append(objp.copy())
            imgpoints.append(corners)

    if not objpoints or img_size is None:
        raise RuntimeError("Nenašli sa platné snímky šachovnice. Skús viac snímok / lepšie osvetlenie / správny pattern.")

    rms, camera_matrix, dist_coeffs, rvecs, tvecs = cv2.calibrateCamera(
        objpoints, imgpoints, img_size, None, None
    )
    return CalibrationResult(
        camera_matrix=camera_matrix,
        dist_coeffs=dist_coeffs,
        rvecs=rvecs,
        tvecs=tvecs,
        rms=float(rms),
        image_size=img_size,
    )


def save_calibration(cal: CalibrationResult, out_dir: Path) -> Tuple[Path, Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    npz_path = out_dir / "calibration.npz"
    yml_path = out_dir / "calibration.yaml"

    np.savez_compressed(
        npz_path,
        camera_matrix=cal.camera_matrix,
        dist_coeffs=cal.dist_coeffs,
        rms=cal.rms,
        image_w=cal.image_size[0],
        image_h=cal.image_size[1],
    )

    fs = cv2.FileStorage(str(yml_path), cv2.FILE_STORAGE_WRITE)
    fs.write("camera_matrix", cal.camera_matrix)
    fs.write("dist_coeffs", cal.dist_coeffs)
    fs.write("rms", cal.rms)
    fs.write("image_w", cal.image_size[0])
    fs.write("image_h", cal.image_size[1])
    fs.release()

    return npz_path, yml_path


def load_calibration(path: Path) -> CalibrationResult:
    if path.suffix.lower() == ".npz":
        data = np.load(str(path), allow_pickle=False)
        cm = data["camera_matrix"]
        dc = data["dist_coeffs"]
        rms = float(data["rms"])
        w = int(data["image_w"])
        h = int(data["image_h"])
        return CalibrationResult(cm, dc, [], [], rms, (w, h))

    # yaml
    fs = cv2.FileStorage(str(path), cv2.FILE_STORAGE_READ)
    if not fs.isOpened():
        raise RuntimeError("Nepodarilo sa otvoriť kalibráciu.")
    cm = fs.getNode("camera_matrix").mat()
    dc = fs.getNode("dist_coeffs").mat()
    rms = float(fs.getNode("rms").real())
    w = int(fs.getNode("image_w").real())
    h = int(fs.getNode("image_h").real())
    fs.release()
    return CalibrationResult(cm, dc, [], [], rms, (w, h))


def undistort_frame(frame: np.ndarray, cal: CalibrationResult) -> np.ndarray:
    h, w = frame.shape[:2]
    new_cm, roi = cv2.getOptimalNewCameraMatrix(cal.camera_matrix, cal.dist_coeffs, (w, h), 1, (w, h))
    und = cv2.undistort(frame, cal.camera_matrix, cal.dist_coeffs, None, new_cm)
    x, y, rw, rh = roi
    if rw > 0 and rh > 0:
        und = und[y:y+rh, x:x+rw]
        und = cv2.resize(und, (w, h), interpolation=cv2.INTER_AREA)
    return und


# -------------------------
# Detekcia tvarov + centroid
# -------------------------

def contour_centroid(cnt: np.ndarray) -> Optional[Tuple[int, int]]:
    m = cv2.moments(cnt)
    if abs(m.get("m00", 0.0)) < 1e-6:
        return None
    cx = int(m["m10"] / m["m00"])
    cy = int(m["m01"] / m["m00"])
    return (cx, cy)


def classify_polygon(approx: np.ndarray) -> str:
    n = len(approx)
    if n == 3:
        return "TROJUHOLNIK"
    if n == 4:
        # rozlíšenie štvorec vs obdĺžnik
        pts = approx.reshape(-1, 2)
        x, y, w, h = cv2.boundingRect(pts)
        ar = w / float(h) if h != 0 else 0.0
        if 0.90 <= ar <= 1.10:
            return "STVOREC"
        return "OBDLZNIK"
    if n == 5:
        return "PÄTUHOLNIK"
    return f"{n}-UHOLNIK"


def detect_shapes(frame_bgr: np.ndarray) -> List[Dict]:
    """
    Vráti list detekovaných tvarov s kontúrou, názvom a centroidom.
    """
    out: List[Dict] = []

    gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(gray, 60, 180)

    # zlepšenie kontúr
    edges = cv2.dilate(edges, np.ones((3, 3), np.uint8), iterations=1)

    cnts, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    for cnt in cnts:
        area = cv2.contourArea(cnt)
        if area < 800:  # filtruj šum
            continue

        peri = cv2.arcLength(cnt, True)
        approx = cv2.approxPolyDP(cnt, 0.02 * peri, True)

        name = classify_polygon(approx)

        # kontrola kružnice cez circularity
        if len(approx) > 5:
            perimeter = max(peri, 1e-6)
            circularity = 4 * np.pi * area / (perimeter * perimeter)
            if circularity > 0.78:
                name = "KRUZNICA"

        cen = contour_centroid(cnt)
        out.append({"contour": cnt, "approx": approx, "name": name, "centroid": cen, "area": area})

    return out


# -------------------------
# Farebný filter (červená -> zelená default)
# -------------------------

def color_replace_red_to_green(frame_bgr: np.ndarray, out_bgr: Tuple[int, int, int] = (0, 255, 0)) -> np.ndarray:
    hsv = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2HSV)

    # červená má 2 intervaly v HSV
    lower1 = np.array([0, 80, 60], dtype=np.uint8)
    upper1 = np.array([10, 255, 255], dtype=np.uint8)

    lower2 = np.array([170, 80, 60], dtype=np.uint8)
    upper2 = np.array([180, 255, 255], dtype=np.uint8)

    mask1 = cv2.inRange(hsv, lower1, upper1)
    mask2 = cv2.inRange(hsv, lower2, upper2)
    mask = cv2.bitwise_or(mask1, mask2)

    # vyhladenie masky
    mask = cv2.medianBlur(mask, 7)

    out = frame_bgr.copy()
    out[mask > 0] = out_bgr
    return out


# -------------------------
# Bonus: veľkosť v cm na rovine šachovnice
# -------------------------

def homography_to_board_plane(
    frame_bgr: np.ndarray,
    pattern_size: Tuple[int, int],
    square_size_cm: float,
) -> Optional[np.ndarray]:
    """
    Ak je v obraze šachovnica, vráti homografiu H, ktorá mapuje obrazové body (u,v,1)
    do súradníc na rovine šachovnice (X,Y,1) v cm.
    """
    gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
    ok, corners = find_corners(gray, pattern_size)
    if not ok or corners is None:
        return None

    cols, rows = pattern_size
    objp2d = chessboard_object_points(pattern_size, square_size_cm)[:, :2].astype(np.float32)  # (N,2) v cm
    imgp2d = corners.reshape(-1, 2).astype(np.float32)

    H, _ = cv2.findHomography(imgp2d, objp2d, method=0)
    return H


def project_points_to_plane_cm(H: np.ndarray, pts_xy: np.ndarray) -> np.ndarray:
    """
    pts_xy: (N,2) v pixeloch, vráti (N,2) v cm.
    """
    pts = pts_xy.reshape(-1, 1, 2).astype(np.float32)
    cm = cv2.perspectiveTransform(pts, H).reshape(-1, 2)
    return cm


def measure_shape_cm(shape: Dict, H: np.ndarray) -> Optional[str]:
    name = shape["name"]
    cnt = shape["contour"].reshape(-1, 2)
    cm_pts = project_points_to_plane_cm(H, cnt)

    if name == "KRUZNICA":
        # priemer z minEnclosingCircle na rovine
        (x, y), r = cv2.minEnclosingCircle(cm_pts.astype(np.float32))
        diam = 2.0 * float(r)
        return f"{diam:.1f} cm"

    # štvoruholník / obdĺžnik / iné polygonálne: rozmery z minAreaRect
    rect = cv2.minAreaRect(cm_pts.astype(np.float32))
    (cx, cy), (w, h), ang = rect
    a = max(w, h)
    b = min(w, h)
    if a <= 0 or b <= 0:
        return None
    return f"{a:.1f} x {b:.1f} cm"


# -------------------------
# UI / režimy
# -------------------------

def run_calibration_capture(args) -> None:
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    src = create_source(args)
    pattern = (args.chess_cols, args.chess_rows)

    saved: List[Path] = []
    print("\n[Kalibrácia] Ovládanie:")
    print("  SPACE  uložiť snímku šachovnice")
    print("  ENTER  spustiť kalibráciu z uložených snímok")
    print("  q      koniec\n")

    cv2.namedWindow("Calibration", cv2.WINDOW_NORMAL)

    try:
        while True:
            frame = src.read()
            if frame is None:
                continue

            show = frame.copy()
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            ok, corners = find_corners(gray, pattern)
            if ok and corners is not None:
                cv2.drawChessboardCorners(show, pattern, corners, ok)
                cv2.putText(show, "Chessboard: OK", (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 220, 0), 2)
            else:
                cv2.putText(show, "Chessboard: NOT FOUND", (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)

            cv2.putText(show, f"Saved: {len(saved)}", (20, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 0), 2)
            cv2.imshow("Calibration", show)

            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                break
            if key == 32:  # SPACE
                ts = int(time.time() * 1000)
                path = out_dir / f"chess_{ts}.png"
                cv2.imwrite(str(path), frame)
                saved.append(path)
                print("Saved:", path)
            if key in (13, 10):  # ENTER
                if len(saved) < 8:
                    print("Pozor: málo snímok (odporúčané aspoň 10-15). Skúšam aj tak...")
                cal = calibrate_from_images(saved, pattern, args.square_cm)
                npz_path, yml_path = save_calibration(cal, out_dir)

                print("\n=== Výsledok kalibrácie ===")
                print("RMS reprojection error:", cal.rms)
                print("Camera matrix:\n", cal.camera_matrix)
                print("Dist coeffs:", cal.dist_coeffs.ravel())
                print(f"fx={cal.fx:.3f}, fy={cal.fy:.3f}, cx={cal.cx:.3f}, cy={cal.cy:.3f}")
                print("Uložené:", npz_path)
                print("Uložené:", yml_path)
                print("===========================\n")
                # krátka demo undistort
                demo_undistort_live(src, cal)
                break

    finally:
        src.close()
        cv2.destroyAllWindows()


def demo_undistort_live(src: FrameSource, cal: CalibrationResult, seconds: float = 6.0) -> None:
    cv2.namedWindow("Undistort demo", cv2.WINDOW_NORMAL)
    t0 = time.time()
    while time.time() - t0 < seconds:
        frame = src.read()
        if frame is None:
            continue
        und = undistort_frame(frame, cal)
        both = np.hstack([frame, und])
        cv2.imshow("Undistort demo", both)
        if (cv2.waitKey(1) & 0xFF) == ord("q"):
            break
    cv2.destroyWindow("Undistort demo")


def run_live(args) -> None:
    src = create_source(args)

    cal: Optional[CalibrationResult] = None
    if args.calib_path:
        cal = load_calibration(Path(args.calib_path))

    pattern = (args.chess_cols, args.chess_rows)

    mode_undistort = bool(args.start_undistort)
    mode_shapes = bool(args.start_shapes)
    mode_color = bool(args.start_color)
    mode_bonus = bool(args.start_bonus)

    print("\n[Live] Ovládanie:")
    print("  u  toggle undistort")
    print("  s  toggle shape detection")
    print("  f  toggle color filter (red->green)")
    print("  b  toggle bonus meranie v cm (vyžaduje šachovnicu v zábere)")
    print("  q  koniec\n")

    cv2.namedWindow("Live", cv2.WINDOW_NORMAL)

    try:
        while True:
            frame = src.read()
            if frame is None:
                continue

            work = frame
            if mode_undistort and cal is not None:
                work = undistort_frame(work, cal)

            # farba
            if mode_color:
                work = color_replace_red_to_green(work, out_bgr=(0, 255, 0))

            # bonus homografia (len keď treba)
            H = None
            if mode_bonus:
                H = homography_to_board_plane(work, pattern, args.square_cm)
                if H is None:
                    cv2.putText(work, "BONUS: chessboard not found", (20, 90),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

            # tvary
            if mode_shapes:
                shapes = detect_shapes(work)
                for sh in shapes:
                    cnt = sh["contour"]
                    name = sh["name"]
                    cv2.drawContours(work, [cnt], -1, (0, 255, 255), 2)
                    c = sh["centroid"]
                    if c is not None:
                        cv2.circle(work, c, 4, (0, 0, 255), -1)

                        label = name
                        if H is not None:
                            size_txt = measure_shape_cm(sh, H)
                            if size_txt:
                                label = f"{name} ({size_txt})"

                        cv2.putText(work, label, (c[0] + 10, c[1] - 10),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

            # status overlay
            status = []
            status.append(f"UNDISTORT={'ON' if (mode_undistort and cal is not None) else 'OFF'}")
            status.append(f"SHAPES={'ON' if mode_shapes else 'OFF'}")
            status.append(f"COLOR={'ON' if mode_color else 'OFF'}")
            status.append(f"BONUS={'ON' if mode_bonus else 'OFF'}")
            cv2.putText(work, " | ".join(status), (20, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 180, 255), 2)

            cv2.imshow("Live", work)

            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                break
            if key == ord("u"):
                mode_undistort = not mode_undistort
            if key == ord("s"):
                mode_shapes = not mode_shapes
            if key == ord("f"):
                mode_color = not mode_color
            if key == ord("b"):
                mode_bonus = not mode_bonus

    finally:
        src.close()
        cv2.destroyAllWindows()


def build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="PVSO Zadanie 2 - Kalibrácia + Tvary + Farebný filter (OpenCV)")
    sub = p.add_subparsers(dest="cmd", required=True)

    # spoločné args
    def add_common(sp):
        sp.add_argument("--source", choices=["ximea", "opencv"], default="ximea",
                        help="Zdroj kamery. 'ximea' používa xiapi, 'opencv' používa cv2.VideoCapture.")
        sp.add_argument("--exposure-us", type=int, default=20000, help="Ximea expozícia v mikrosekundách.")
        sp.add_argument("--xi-format", type=str, default="XI_RGB32", help="Ximea imgdataformat (napr. XI_RGB32).")
        sp.add_argument("--cam-index", type=int, default=0, help="Index pre cv2.VideoCapture (ak source=opencv).")
        sp.add_argument("--width", type=int, default=None, help="Šírka pre cv2.VideoCapture.")
        sp.add_argument("--height", type=int, default=None, help="Výška pre cv2.VideoCapture.")
        sp.add_argument("--chess-cols", type=int, default=9, help="Počet vnútorných rohov šachovnice - stĺpce.")
        sp.add_argument("--chess-rows", type=int, default=6, help="Počet vnútorných rohov šachovnice - riadky.")
        sp.add_argument("--square-cm", type=float, default=2.5, help="Veľkosť jedného štvorca šachovnice v cm.")

    sp_cal = sub.add_parser("calibrate", help="Zber snímok šachovnice + kalibrácia + uloženie parametrov")
    add_common(sp_cal)
    sp_cal.add_argument("--out-dir", type=str, default="./ZAD2_OUT", help="Kam uložiť snímky a kalibráciu")

    sp_run = sub.add_parser("run", help="Live režim: undistort + tvary + filter")
    add_common(sp_run)
    sp_run.add_argument("--calib-path", type=str, default="./ZAD2_OUT/calibration.npz", help="Cesta ku kalibrácii (.npz/.yaml)")
    sp_run.add_argument("--start-undistort", action="store_true", help="Spustiť s undistort ON (ak je kalibrácia).")
    sp_run.add_argument("--start-shapes", action="store_true", help="Spustiť s detekciou tvarov ON.")
    sp_run.add_argument("--start-color", action="store_true", help="Spustiť s farebným filtrom ON.")
    sp_run.add_argument("--start-bonus", action="store_true", help="Spustiť s bonus meraním ON (vyžaduje šachovnicu).")
    return p


def main() -> int:
    args = build_argparser().parse_args()

    try:
        if args.cmd == "calibrate":
            run_calibration_capture(args)
        elif args.cmd == "run":
            run_live(args)
        else:
            raise RuntimeError("Neznámy príkaz.")
        return 0
    except KeyboardInterrupt:
        return 0
    except Exception as e:
        print("\n[CHYBA]", e)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())