from ximea import xiapi
import cv2
import numpy as np

EXPOSURE_US = 500000
RESIZE_TO = (1200, 900)   # pre rychlejsie spracovanie; ak nechces, daj None
WINDOW_NAME = "Detekcia tvarov - XIMEA"


def detect_shapes(frame):
    output = frame.copy()

    # grayscale pre Houghovu transformaciu aj hrany
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    gray_blur = cv2.medianBlur(gray, 5)

    # pocitadla
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

            # obrys kruznice
            cv2.circle(output, (x, y), r, (0, 255, 0), 2)
            # stred
            cv2.circle(output, (x, y), 3, (0, 0, 255), -1)

            cv2.putText(
                output,
                "Kruznica",
                (x - 40, y - r - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 255, 0),
                2
            )

            cv2.putText(
                output,
                f"S({x},{y})",
                (x + 10, y),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.45,
                (0, 0, 255),
                1
            )

            # maska proti dvojitej detekcii cez kontury
            cv2.circle(circle_mask, (x, y), r + 8, 255, -1)

    # =========================
    # 2. DETEKCIA OSTATNYCH TVAROV
    # =========================
    edges = cv2.Canny(gray_blur, 50, 150)
    edges_no_circles = cv2.bitwise_and(edges, cv2.bitwise_not(circle_mask))

    contours, _ = cv2.findContours(edges_no_circles, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

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

            cv2.putText(
                output,
                shape_name,
                (cx - 50, cy - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (255, 0, 0),
                2
            )

            cv2.putText(
                output,
                f"S({cx},{cy})",
                (cx + 10, cy + 15),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.45,
                (0, 0, 255),
                1
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

    return output


def main():
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

            # XI_RGB32 -> nechame len prve 3 kanaly
            frame = frame[:, :, :3]

            # RGB -> BGR pre OpenCV
            frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

            if RESIZE_TO is not None:
                frame = cv2.resize(frame, RESIZE_TO, interpolation=cv2.INTER_AREA)

            result = detect_shapes(frame)
            cv2.imshow(WINDOW_NAME, result)

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