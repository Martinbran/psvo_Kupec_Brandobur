from ximea import xiapi
import cv2
import numpy as np

#EXPOSURE_US = 500000
EXPOSURE_US = 50000
RESIZE_TO = (1200, 900)   # alebo None
WINDOW_ORIGINAL = "Povodny obraz - XIMEA"
WINDOW_MASK = "Maska cervenej"
WINDOW_RESULT = "Vysledok - cervena na zelenu"


def replace_red_with_green(image):
    # BGR -> HSV
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    lower_red1 = np.array([0, 70, 50])
    upper_red1 = np.array([15, 255, 255])

    lower_red2 = np.array([165, 70, 50])
    upper_red2 = np.array([180, 255, 255])

    mask1 = cv2.inRange(hsv, lower_red1, upper_red1)
    mask2 = cv2.inRange(hsv, lower_red2, upper_red2)
    mask = cv2.bitwise_or(mask1, mask2)

    kernel = np.ones((5, 5), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)

    result = image.copy()
    result[mask > 0] = [0, 255, 0]   # BGR: zelena

    return mask, result


def main():
    cam = xiapi.Camera()
    print("Opening XIMEA camera...")
    cam.open_device()

    cam.set_exposure(EXPOSURE_US)
    cam.set_param("imgdataformat", "XI_RGB32")
    cam.set_param("auto_wb", 1)

    img = xiapi.Image()
    cam.start_acquisition()

    cv2.namedWindow(WINDOW_ORIGINAL, cv2.WINDOW_NORMAL)
    cv2.namedWindow(WINDOW_MASK, cv2.WINDOW_NORMAL)
    cv2.namedWindow(WINDOW_RESULT, cv2.WINDOW_NORMAL)

    try:
        print("Q = quit")

        while True:
            cam.get_image(img)
            frame = img.get_image_data_numpy()

            frame = frame[:, :, :3]

            if RESIZE_TO is not None:
                frame = cv2.resize(frame, RESIZE_TO, interpolation=cv2.INTER_AREA)

            mask, result = replace_red_with_green(frame)

            cv2.putText(frame, "Q = quit", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

            cv2.putText(result, "Q = quit", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

            cv2.imshow(WINDOW_ORIGINAL, frame)
            cv2.imshow(WINDOW_MASK, mask)
            cv2.imshow(WINDOW_RESULT, result)

            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                print("Koniec.")
                break

    finally:
        cam.stop_acquisition()
        cam.close_device()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()