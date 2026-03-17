from ximea import xiapi
import cv2
import numpy as np

EXPOSURE_US = 50000
RESIZE_TO = (1200, 900)
WINDOW_NAME = "Detekcia tvarov - XIMEA"

# stabilizacia obrazu
last_counts = []
stable_result = None
STABLE_FRAMES = 3

# farby v BGR
COLOR_CIRCLE = (0, 255, 0)        
COLOR_TRIANGLE = (0, 255, 255)   
COLOR_SQUARE = (255, 0, 0)        
COLOR_RECTANGLE = (255, 0, 255)   
COLOR_CENTER = (0, 0, 255)        
COLOR_TEXT = (255, 255, 255)      


def detect_shapes(frame):
    output = frame.copy()

    # grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    gray_blur = cv2.GaussianBlur(gray, (5, 5), 0)
    gray_blur = cv2.medianBlur(gray_blur, 5)

    # HSV maska
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    saturation = hsv[:, :, 1]

    _, color_mask = cv2.threshold(saturation, 50, 255, cv2.THRESH_BINARY)

    kernel = np.ones((5, 5), np.uint8)
    color_mask = cv2.morphologyEx(color_mask, cv2.MORPH_CLOSE, kernel)
    color_mask = cv2.morphologyEx(color_mask, cv2.MORPH_OPEN, kernel)

    # jemne spojenie casti tvarov
    kernel2 = np.ones((3, 3), np.uint8)
    color_mask = cv2.morphologyEx(color_mask, cv2.MORPH_CLOSE, kernel2)

    circle_count = 0
    triangle_count = 0
    square_count = 0
    rectangle_count = 0

    contours, _ = cv2.findContours(color_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area < 1800:
            continue

        perimeter = cv2.arcLength(cnt, True)
        if perimeter == 0:
            continue

        approx = cv2.approxPolyDP(cnt, 0.03 * perimeter, True)

        # tazisko z momentov
        M = cv2.moments(cnt)
        if M["m00"] == 0:
            continue

        cx = int(M["m10"] / M["m00"])
        cy = int(M["m01"] / M["m00"])

        x, y, w, h = cv2.boundingRect(approx)
        aspect_ratio = w / float(h)

        # circularity = 1 pre idealnu kruznicu
        circularity = 4 * np.pi * area / (perimeter * perimeter)

        # =========================
        # 1. TROJUHOLNIK
        # =========================
        if len(approx) == 3:
            triangle_count += 1

            cv2.drawContours(output, [approx], -1, COLOR_TRIANGLE, 2)
            cv2.circle(output, (cx, cy), 4, COLOR_CENTER, -1)

            cv2.putText(
                output,
                "Trojuholnik",
                (cx - 65, cy - 12),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                COLOR_TRIANGLE,
                2
            )

        # =========================
        # 2. STVOREC / OBDLZNIK
        # =========================
        elif len(approx) == 4:
            if 0.70 <= aspect_ratio <= 1.20:
                square_count += 1
                shape_name = "Stvorec"
                shape_color = COLOR_SQUARE
            else:
                rectangle_count += 1
                shape_name = "Obdlznik"
                shape_color = COLOR_RECTANGLE

            cv2.drawContours(output, [approx], -1, shape_color, 2)
            cv2.circle(output, (cx, cy), 4, COLOR_CENTER, -1)

            cv2.putText(
                output,
                shape_name,
                (cx - 50, cy - 12),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                shape_color,
                2
            )

        # =========================
        # 3. KRUZNICA / ELIPSA
        # =========================
        elif len(approx) > 4 and circularity > 0.5 and 0.5 <= aspect_ratio <= 1.5 and len(cnt) >= 5:
            circle_count += 1

            ellipse = cv2.fitEllipse(cnt)
            cv2.ellipse(output, ellipse, COLOR_CIRCLE, 2)
            cv2.circle(output, (cx, cy), 4, COLOR_CENTER, -1)

            cv2.putText(
                output,
                "Kruznica",
                (cx - 50, cy - 18),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                COLOR_CIRCLE,
                2
            )

    # =========================
    # 4. VYPIS POCTOV
    # =========================
    info_y = 30
    step = 30

    cv2.putText(output, "Detegovane tvary:", (10, info_y),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, COLOR_TEXT, 2)
    info_y += step

    cv2.putText(output, f"Kruznice: {circle_count}", (10, info_y),
                cv2.FONT_HERSHEY_SIMPLEX, 0.65, COLOR_CIRCLE, 2)
    info_y += step

    cv2.putText(output, f"Trojuholniky: {triangle_count}", (10, info_y),
                cv2.FONT_HERSHEY_SIMPLEX, 0.65, COLOR_TRIANGLE, 2)
    info_y += step

    cv2.putText(output, f"Stvorce: {square_count}", (10, info_y),
                cv2.FONT_HERSHEY_SIMPLEX, 0.65, COLOR_SQUARE, 2)
    info_y += step

    cv2.putText(output, f"Obdlzniky: {rectangle_count}", (10, info_y),
                cv2.FONT_HERSHEY_SIMPLEX, 0.65, COLOR_RECTANGLE, 2)
    info_y += step

    cv2.putText(output, "Q = quit", (10, info_y + 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.65, COLOR_TEXT, 2)

    counts = (circle_count, triangle_count, square_count, rectangle_count)
    return output, counts


def main():
    global last_counts, stable_result

    cam = xiapi.Camera()
    print("Opening XIMEA camera...")
    cam.open_device()

    cam.set_exposure(EXPOSURE_US)
    cam.set_param("imgdataformat", "XI_RGB32")
    cam.set_param("auto_wb", 1)

    img = xiapi.Image()
    cam.start_acquisition()

    cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)

    try:
        print("Q = quit")

        while True:
            cam.get_image(img)
            frame = img.get_image_data_numpy()

            frame = frame[:, :, :3]

            if RESIZE_TO is not None:
                frame = cv2.resize(frame, RESIZE_TO, interpolation=cv2.INTER_AREA)

            result, counts = detect_shapes(frame)

            # stabilizacia vysledku
            last_counts.append(counts)
            if len(last_counts) > STABLE_FRAMES:
                last_counts.pop(0)

            if len(last_counts) == STABLE_FRAMES and all(c == last_counts[0] for c in last_counts):
                stable_result = result.copy()

            display = stable_result if stable_result is not None else result
            cv2.imshow(WINDOW_NAME, display)

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