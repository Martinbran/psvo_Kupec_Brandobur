import cv2
import numpy as np
import os

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
IMAGE_PATH = os.path.join(SCRIPT_DIR, "tvary.jpg")


def detect_shapes(frame):
    output = frame.copy()

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    gray_blur = cv2.medianBlur(gray, 5)

    circle_count = 0
    triangle_count = 0
    square_count = 0
    rectangle_count = 0

    circles = cv2.HoughCircles(
        gray_blur,
        cv2.HOUGH_GRADIENT,
        dp=1.2,
        minDist=80,
        param1=100,
        param2=35,
        minRadius=20,
        maxRadius=300
    )

    circle_mask = np.zeros_like(gray)

    if circles is not None:
        circles = np.uint16(np.around(circles))

        for i in circles[0, :]:
            x, y, r = i[0], i[1], i[2]
            circle_count += 1

            cv2.circle(output, (x, y), r, (0, 255, 0), 2)
            cv2.circle(output, (x, y), 3, (0, 0, 255), -1)

            cv2.putText(output, "Kruznica", (x - 40, y - r - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            cv2.putText(output, f"S({x},{y})", (x + 10, y),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 255), 1)

            cv2.circle(circle_mask, (x, y), r + 5, 255, -1)

    edges = cv2.Canny(gray_blur, 50, 150)
    edges_no_circles = cv2.bitwise_and(edges, cv2.bitwise_not(circle_mask))

    contours, _ = cv2.findContours(edges_no_circles, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    print("Pocet kontur:", len(contours))

    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area < 800:
            continue

        perimeter = cv2.arcLength(cnt, True)
        approx = cv2.approxPolyDP(cnt, 0.02 * perimeter, True)

        M = cv2.moments(cnt)
        if M["m00"] == 0:
            continue

        cx = int(M["m10"] / M["m00"])
        cy = int(M["m01"] / M["m00"])

        shape_name = None

        if len(approx) == 3:
            shape_name = "Trojuholnik"
            triangle_count += 1

        elif len(approx) == 4:
            x, y, w, h = cv2.boundingRect(approx)
            aspect_ratio = w / float(h)

            if 0.95 <= aspect_ratio <= 1.05:
                shape_name = "Stvorec"
                square_count += 1
            else:
                shape_name = "Obdlznik"
                rectangle_count += 1

        if shape_name is not None:
            cv2.drawContours(output, [approx], -1, (255, 0, 0), 2)
            cv2.circle(output, (cx, cy), 4, (0, 0, 255), -1)

            cv2.putText(output, shape_name, (cx - 50, cy - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)
            cv2.putText(output, f"S({cx},{cy})", (cx + 10, cy + 15),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 255), 1)

    info_y = 30
    step = 30

    cv2.putText(output, "Detegovane tvary:", (10, info_y),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    info_y += step

    cv2.putText(output, f"Kruznice: {circle_count}", (10, info_y),
                cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 255, 0), 2)
    info_y += step

    cv2.putText(output, f"Trojuholniky: {triangle_count}", (10, info_y),
                cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 0, 0), 2)
    info_y += step

    cv2.putText(output, f"Stvorce: {square_count}", (10, info_y),
                cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 0, 0), 2)
    info_y += step

    cv2.putText(output, f"Obdlzniky: {rectangle_count}", (10, info_y),
                cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 0, 0), 2)

    return output, gray, edges


def main():
    print("Skript sa spustil.")
    print("Hladam obrazok tu:", IMAGE_PATH)
    print("Subor existuje?", os.path.exists(IMAGE_PATH))

    img = cv2.imread(IMAGE_PATH)

    if img is None:
        print(f"Chyba: obrazok '{IMAGE_PATH}' sa nepodarilo nacitat.")
        input("Stlac Enter pre ukoncenie...")
        return

    print("Obrazok nacitany. Rozmery:", img.shape)

    # zmensi obrazok pre rychlejsie spracovanie
    img = cv2.resize(img, (1200, 900), interpolation=cv2.INTER_AREA)

    print("Novy rozmer:", img.shape)

    result, gray, edges = detect_shapes(img)

    cv2.namedWindow("Detekcia tvarov", cv2.WINDOW_NORMAL)
    cv2.imshow("Detekcia tvarov", result)

    # pomocné okná, aby si videl, či sa niečo spracovalo
    cv2.namedWindow("Gray", cv2.WINDOW_NORMAL)
    cv2.imshow("Gray", gray)

    cv2.namedWindow("Edges", cv2.WINDOW_NORMAL)
    cv2.imshow("Edges", edges)

    print("Okna su zobrazene. Stlac lubovolnu klavesu v okne.")
    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()