from ximea import xiapi
import cv2
import numpy as np
import os
import shutil

SAVE_DIR = "./FOTKY_SACHOVNICA_bez_resize"
RESIZE_TO = (None)   # ak nechceš resize, nastav na None
EXPOSURE_US = 500000
NUM_SHOTS = 15

def main():
    if os.path.isdir(SAVE_DIR):
        shutil.rmtree(SAVE_DIR)
    os.makedirs(SAVE_DIR, exist_ok=True)

    cam = xiapi.Camera()
    print("Opening camera...")
    cam.open_device()
    cam.set_exposure(EXPOSURE_US)
    cam.set_param("imgdataformat", "XI_RGB32")
    cam.set_param("auto_wb", 1)

    img = xiapi.Image()
    cam.start_acquisition()

    cv2.namedWindow("Live", cv2.WINDOW_NORMAL)

    shot_count = 0

    try:
        print(f"SPACE = uloz snimku ({NUM_SHOTS}x), q = koniec")

        while True:
            cam.get_image(img)
            frame = img.get_image_data_numpy()

            # RGB32 -> zhod alpha (nechaj 3 kanály)
            frame = frame[:, :, :3]

            # voliteľný resize
            if RESIZE_TO is not None:
                frame = cv2.resize(frame, RESIZE_TO, interpolation=cv2.INTER_AREA)

            # info do okna
            vis = frame.copy()
            cv2.putText(
                vis,
                f"Shots: {shot_count}/{NUM_SHOTS}  (SPACE=save, Q=quit)",
                (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (255, 255, 255),
                2
            )

            cv2.imshow("Live", vis)

            key = cv2.waitKey(1) & 0xFF

            if key == ord('q'):
                print("Koniec (q).")
                return

            if key == 32:  # SPACE
                shot_count += 1
                path = os.path.join(SAVE_DIR, f"chess_{shot_count:02d}.png")
                cv2.imwrite(path, frame)
                print(f"Saved {shot_count}/{NUM_SHOTS}: {path}")

                if shot_count >= NUM_SHOTS:
                    print("Hotovo: ulozenych 10 fotiek.")
                    return

    finally:
        cam.stop_acquisition()
        cam.close_device()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main()