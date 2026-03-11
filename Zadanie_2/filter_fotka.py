import cv2
import numpy as np
import os

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
IMAGE_PATH = os.path.join(SCRIPT_DIR, "tvary.jpg")


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

    # kopia povodneho obrazka
    result = image.copy()

    # nahradenie cervenej za zelenu v BGR
    result[mask > 0] = [0, 255, 0]

    return mask, result


def main():
    print("Hladam obrazok tu:", IMAGE_PATH)
    print("Subor existuje?", os.path.exists(IMAGE_PATH))

    image = cv2.imread(IMAGE_PATH)

    if image is None:
        print(f"Chyba: obrazok '{IMAGE_PATH}' sa nepodarilo nacitat.")
        return

    print("Obrazok nacitany. Rozmery:", image.shape)

    # volitelne zmensenie, aby sa rychlejsie zobrazoval
    image = cv2.resize(image, (1200, 900), interpolation=cv2.INTER_AREA)

    mask, result = replace_red_with_green(image)

    cv2.namedWindow("Povodny obrazok", cv2.WINDOW_NORMAL)
    cv2.namedWindow("Maska cervenej", cv2.WINDOW_NORMAL)
    cv2.namedWindow("Vysledok - cervena na zelenu", cv2.WINDOW_NORMAL)

    cv2.imshow("Povodny obrazok", image)
    cv2.imshow("Maska cervenej", mask)
    cv2.imshow("Vysledok - cervena na zelenu", result)

    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()