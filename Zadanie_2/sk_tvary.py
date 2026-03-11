from ximea import xiapi
import cv2
import numpy as np

EXPOSURE_US = 500000
RESIZE_TO = (1200, 900)
WINDOW_NAME = "Detekcia tvarov - XIMEA"

last_counts = []
stable_result = None
STABLE_FRAMES = 3


def detect_shapes(frame):
    output = frame.copy()

    # grayscale pre Hough kruznice
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    gray_blur = cv2.GaussianBlur(gray, (5, 5), 0)
    gray_blur = cv2.medianBlur(gray_blur, 5)

    # HSV maska pre farebne objekty
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    saturation = hsv[:, :, 1]

    # farebne papiere maju vyssiu saturaciu ako stena
    _, color_mask = cv2.threshold(saturation, 55, 255, cv2.THRESH_BINARY)

    kernel = np.ones((5, 5), np.uint8)
    color_mask = cv2.morphologyEx(color_mask, cv2.MORPH_CLOSE, kernel)
    color_mask = cv2.morphologyEx(color_mask, cv2.MORPH_OPEN, kernel)

    circle_count = 0
    triangle_count = 0
    square_count = 0
    rectangle_count = 0

    # =========================
    # 1. DETEKCIA KRUZNIC
    # =========================
    circles = cv2.HoughCircles(
        gray_blur,
        cv2.HOUGH_GRADIENT,
        dp=1.2,
        minDist=80,
        param1=100,
        param2=32,
        minRadius=20,
        maxRadius=300
    )

    circle_mask = np.zeros_like(gray)

    if circles is not None:
        circles = np.uint16(np.around(circles))

        for i in circles[0, :]:
            x, y, r = int(i[0]), int(i[1]), int(i[2])

            circle_count += 1
            cv2.circle(output, (x, y), r, (0, 255, 0), 2)

            # cervena bodka - stred / tazisko
            cv2.circle(output, (x, y), 4, (0, 0, 255), -1)

            cv2.putText(
                output,
                "Kruznica",
                (x - 45, y - r - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 255, 0),
                2
            )

            cv2.circle(circle_mask, (x, y), r + 10, 255, -1)

    # =========================
    # 2. DETEKCIA OSTATNYCH TVAROV
    # =========================
    # odstran kruznice z masky, aby sa nehodnotili este raz cez kontury
    shapes_mask = cv2.bitwise_and(color_mask, cv2.bitwise_not(circle_mask))

    contours, _ = cv2.findContours(shapes_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area < 1500:
            continue

        perimeter = cv2.arcLength(cnt, True)
        approx = cv2.approxPolyDP(cnt, 0.025 * perimeter, True)

        x, y, w, h = cv2.boundingRect(approx)
        cx = x + w // 2
        cy = y + h // 2

        shape_name = None

        if len(approx) == 3:
            shape_name = "Trojuholnik"
            triangle_count += 1

        elif len(approx) == 4:
            aspect_ratio = w / float(h)

            if 0.50 <= aspect_ratio <= 1.30:
                shape_name = "Stvorec"
                square_count += 1
            else:
                shape_name = "Obdlznik"
                rectangle_count += 1

        if shape_name is not None:
            cv2.drawContours(output, [approx], -1, (255, 0, 0), 2)

            # cervena bodka - tazisko
            cv2.circle(output, (cx, cy), 4, (0, 0, 255), -1)

            cv2.putText(
                output,
                shape_name,
                (cx - 50, cy - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (255, 0, 0),
                2
            )

    # =========================
    # 3. VYPIS POCTOV
    # =========================
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
    info_y += step

    cv2.putText(output, "Q = quit", (10, info_y + 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 2)

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