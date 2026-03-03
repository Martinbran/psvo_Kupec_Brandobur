#!/usr/bin/env python3
# -*- codi:contentReference[oaicite:3]{index=3}SO Zadanie 2 – Bod 1: Kalibrácia kamery (OpenCV chessboard)

#Funkcie:
#- Zber kalibračných snímok šachovnice z kamery (Ximea/xiapi alebo OpenCV kamera)
#- Detekcia rohov: cv2.findChessboardCorners + spresnenie cv2.cornerSubPix
#- Kalibrácia: cv2.calibrateCamera
#- Výpis: camera matrix + fx, fy, cx, cy + dist coeffs + reprojection error
#- Uloženie: calibration.npz aj calibration.yaml
#- Demo: odstránenie distorzie (undistortion) v reálnom čase (original | undistorted)

#Ovládanie v okne:
#- SPACE  : uloží "dobrú" snímku (len keď je šachovnica nájdená) + pridá body do kalibrácie
#- ENTER  : spustí kalibráciu z nazbieraných snímok
#- q      : koniec

from __future__ import annotations

import argparse
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Tuple, List

import cv2
import numpy as np


# -------------------------
# Zdroj snímok (Ximea / fallback)
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
        return frame if ok else None

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
            # XI_RGB32 -> RGB(A), nechaj prvé 3 kanály
            if frame.ndim == 3 and frame.shape[2] >= 3:
                frame = frame[:, :, :3]
            # xiapi typicky vracia RGB, OpenCV chce BGR
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
    mean_reproj_error: float
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
    3D body šachovnice v rovine Z=0 (v cm).
    pattern_size = (cols, rows) = počet VNÚTORNÝCH rohov.
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

    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
    corners2 = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
    return True, corners2


def compute_mean_reprojection_error(
    objpoints: List[np.ndarray],
    imgpoints: List[np.ndarray],
    rvecs: List[np.ndarray],
    tvecs: List[np.ndarray],
    camera_matrix: np.ndarray,
    dist_coeffs: np.ndarray,
) -> float:
    """
    Priemerná reprojekčná chyba podľa OpenCV tutorialu.
    """
    if not objpoints:
        return float("nan")

    total = 0.0
    for i in range(len(objpoints)):
        imgpoints2, _ = cv2.projectPoints(objpoints[i], rvecs[i], tvecs[i], camera_matrix, dist_coeffs)
        err = cv2.norm(imgpoints[i], imgpoints2, cv2.NORM_L2) / max(len(imgpoints2), 1)
        total += float(err)
    return total / float(len(objpoints))


def save_calibration(cal: CalibrationResult, out_dir: Path) -> Tuple[Path, Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    npz_path = out_dir / "calibration.npz"
    yml_path = out_dir / "calibration.yaml"

    np.savez_compressed(
        npz_path,
        camera_matrix=cal.camera_matrix,
        dist_coeffs=cal.dist_coeffs,
        rms=cal.rms,
        mean_reproj_error=cal.mean_reproj_error,
        image_w=cal.image_size[0],
        image_h=cal.image_size[1],
    )

    fs = cv2.FileStorage(str(yml_path), cv2.FILE_STORAGE_WRITE)
    fs.write("camera_matrix", cal.camera_matrix)
    fs.write("dist_coeffs", cal.dist_coeffs)
    fs.write("rms", cal.rms)
    fs.write("mean_reproj_error", cal.mean_reproj_error)
    fs.write("image_w", cal.image_size[0])
    fs.write("image_h", cal.image_size[1])
    fs.release()

    return npz_path, yml_path


def undistort_frame(frame: np.ndarray, camera_matrix: np.ndarray, dist_coeffs: np.ndarray) -> np.ndarray:
    """
    Undistortion podľa OpenCV tutorialu: getOptimalNewCameraMatrix + undistort + crop ROI.
    """
    h, w = frame.shape[:2]
    new_cm, roi = cv2.getOptimalNewCameraMatrix(camera_matrix, dist_coeffs, (w, h), 1, (w, h))
    und = cv2.undistort(frame, camera_matrix, dist_coeffs, None, new_cm)

    x, y, rw, rh = roi
    if rw > 0 and rh > 0:
        und = und[y:y + rh, x:x + rw]
        # aby sa dalo porovnať vedľa seba, vrátime na pôvodnú veľkosť
        und = cv2.resize(und, (w, h), interpolation=cv2.INTER_AREA)

    return und


# -------------------------
# Live kalibrácia (zber + výpočet + demo)
# -------------------------

def run_calibration(args) -> None:
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    src = create_source(args)
    pattern = (args.chess_cols, args.chess_rows)
    objp = chessboard_object_points(pattern, args.square_cm)

    objpoints: List[np.ndarray] = []
    imgpoints: List[np.ndarray] = []
    saved_imgs: List[Path] = []

    print("\n[KALIBRÁCIA] Ovládanie:")
    print("  SPACE  uložiť 'dobrú' snímku (len ak je šachovnica nájdená)")
    print("  ENTER  spustiť kalibráciu z nazbieraných snímok")
    print("  q      koniec\n")
    print(f"Pattern (vnútorné rohy): {pattern[0]} x {pattern[1]}")
    print(f"Square size: {args.square_cm} cm")
    print(f"Odporúčanie: aspoň ~10-15 dobrých snímok (rôzne uhly/vzdialenosti).")

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

            cv2.putText(show, f"Good frames: {len(objpoints)}", (20, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 0), 2)
            cv2.imshow("Calibration", show)

            key = cv2.waitKey(1) & 0xFF

            if key == ord("q"):
                break

            if key == 32:  # SPACE
                if not (ok and corners is not None):
                    print("[-] Šachovnica nenájdená -> snímka sa neuložila.")
                    continue

                ts = int(time.time() * 1000)
                img_path = out_dir / f"chess_{ts}.png"
                cv2.imwrite(str(img_path), frame)
                saved_imgs.append(img_path)

                # do kalibrácie pridaj body
                objpoints.append(objp.copy())
                imgpoints.append(corners)

                print(f"[+] Uložené: {img_path} (good={len(objpoints)})")

            if key in (10, 13):  # ENTER
                if len(objpoints) < args.min_frames:
                    print(f"[!] Málo snímok ({len(objpoints)}). Odporúčané aspoň {args.min_frames}. Skúšam aj tak...")

                # image size
                h, w = frame.shape[:2]
                image_size = (w, h)

                rms, camera_matrix, dist_coeffs, rvecs, tvecs = cv2.calibrateCamera(
                    objpoints, imgpoints, image_size, None, None
                )

                mean_err = compute_mean_reprojection_error(
                    objpoints, imgpoints, rvecs, tvecs, camera_matrix, dist_coeffs
                )

                cal = CalibrationResult(
                    camera_matrix=camera_matrix,
                    dist_coeffs=dist_coeffs,
                    rvecs=rvecs,
                    tvecs=tvecs,
                    rms=float(rms),
                    mean_reproj_error=float(mean_err),
                    image_size=image_size,
                )

                npz_path, yml_path = save_calibration(cal, out_dir)

                print("\n=== VÝSLEDOK KALIBRÁCIE ===")
                print(f"RMS reprojection error (cv2.calibrateCamera): {cal.rms:.6f}")
                print(f"Mean reprojection error (manual/projectPoints): {cal.mean_reproj_error:.6f}")
                print("Camera matrix:\n", cal.camera_matrix)
                print("Dist coeffs:", cal.dist_coeffs.ravel())
                print(f"fx={cal.fx:.3f}, fy={cal.fy:.3f}, cx={cal.cx:.3f}, cy={cal.cy:.3f}")
                print("Uložené:", npz_path)
                print("Uložené:", yml_path)
                print("===========================\n")

                demo_undistort_live(src, cal.camera_matrix, cal.dist_coeffs)
                break

    finally:
        src.close()
        cv2.destroyAllWindows()


def demo_undistort_live(src: FrameSource, camera_matrix: np.ndarray, dist_coeffs: np.ndarray) -> None:
    print("[DEMO] Undistortion: stlač 'q' pre koniec.")
    cv2.namedWindow("Undistort demo (original | undistorted)", cv2.WINDOW_NORMAL)

    try:
        while True:
            frame = src.read()
            if frame is None:
                continue

            und = undistort_frame(frame, camera_matrix, dist_coeffs)
            both = np.hstack([frame, und])

            cv2.imshow("Undistort demo (original | undistorted)", both)
            if (cv2.waitKey(1) & 0xFF) == ord("q"):
                break
    finally:
        cv2.destroyWindow("Undistort demo (original | undistorted)")


# -------------------------
# CLI
# -------------------------

def build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="PVSO Zadanie 2 – Bod 1: Kalibrácia kamery (OpenCV)")
    p.add_argument("--source", choices=["ximea", "opencv"], default="ximea",
                   help="Zdroj snímok: 'ximea' (xiapi) alebo 'opencv' (cv2.VideoCapture).")
    p.add_argument("--exposure-us", type=int, default=20000, help="Ximea expozícia v mikrosekundách.")
    p.add_argument("--xi-format", type=str, default="XI_RGB32", help="Ximea imgdataformat (napr. XI_RGB32).")
    p.add_argument("--cam-index", type=int, default=0, help="Index pre cv2.VideoCapture (ak source=opencv).")
    p.add_argument("--width", type=int, default=None, help="Šírka pre cv2.VideoCapture.")
    p.add_argument("--height", type=int, default=None, help="Výška pre cv2.VideoCapture.")

    p.add_argument("--chess-cols", type=int, default=9, help="Počet vnútorných rohov – stĺpce.")
    p.add_argument("--chess-rows", type=int, default=6, help="Počet vnútorných rohov – riadky.")
    p.add_argument("--square-cm", type=float, default=2.5, help="Veľkosť jedného štvorca šachovnice v cm.")

    p.add_argument("--out-dir", type=str, default="./CALIB_OUT", help="Kam uložiť snímky a kalibráciu.")
    p.add_argument("--min-frames", type=int, default=10, help="Odporúčaný minimálny počet dobrých snímok.")
    return p


def main() -> int:
    args = build_argparser().parse_args()
    try:
        run_calibration(args)
        return 0
    except KeyboardInterrupt:
        return 0
    except Exception as e:
        print("\n[CHYBA]", e)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())