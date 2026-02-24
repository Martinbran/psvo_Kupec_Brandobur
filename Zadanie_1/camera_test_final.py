from ximea import xiapi
import cv2
import numpy as np
import os
import shutil

SAVE_DIR = "./FOTKY"
RESIZE_TO = (240, 240)   
EXPOSURE_US = 50000

def rotate90_clockwise_for(src: np.ndarray) -> np.ndarray:
    h, w = src.shape[:2]
    ch = src.shape[2]
    dst = np.zeros((w, h, ch), dtype=src.dtype)  # po otočení (w,h,3)

    for y in range(h):
        for x in range(w):
            dst[x, h - 1 - y, :] = src[y, x, :]

    return dst

def print_image_info(img: np.ndarray) -> None:
    print("dtype:", img.dtype)
    print("shape:", img.shape)
    print("size (elements):", img.size)
    print("bytes:", img.nbytes)

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

    captured = []

    try:
        print("SPACE = uloz snimku (4x), q = koniec")
        while True:
            cam.get_image(img)
            frame = img.get_image_data_numpy()

            # RGB32 -> zhod alpha, sprav 3-kanal
            frame = frame[:, :, :3]

            # zmensime na jednotny rozmer
            frame = cv2.resize(frame, RESIZE_TO, interpolation=cv2.INTER_AREA)

            cv2.imshow("Live", frame)

            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                return
            if key == 32:  # SPACE
                idx = len(captured) + 1
                path = os.path.join(SAVE_DIR, f"shot_{idx}.png")
                cv2.imwrite(path, frame)
                print("Saved:", path)
                captured.append(frame)
                if len(captured) == 4:
                    break

        # 2x2 mozaika cez indexovanie
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

        # 3) kernel 3x3 (sharpen) na cast 1
        kernel = np.array([[0, -1,  0],
                           [-1, 5, -1],
                           [0, -1,  0]], dtype=np.float32)
        part1[:, :] = cv2.filter2D(part1, -1, kernel, borderType=cv2.BORDER_REPLICATE)

        # 4) rotacia cast 2 o 90° pomocou for-cyklu
        part2[:, :] = rotate90_clockwise_for(part2)

        # 5) cast 3: R kanal
        part3[:, :, 0] = 0  # B
        part3[:, :, 1] = 0  # G
        # R ostava

        # 6) info o vyslednom obraze do terminalu
        print("\nINFO o vyslednej mozaike:")
        print_image_info(mosaic)

        out_path = os.path.join(SAVE_DIR, "mosaic_final.png")
        cv2.imwrite(out_path, mosaic)
        print("Saved:", out_path)

        cv2.namedWindow("Mosaic", cv2.WINDOW_NORMAL)
        cv2.imshow("Mosaic", mosaic)
        print("Stlac q v okne Mosaic na koniec.")

        while True:
            if (cv2.waitKey(10) & 0xFF) == ord('q'):
                break

    finally:
        cam.stop_acquisition()
        cam.close_device()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
