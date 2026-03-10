from ximea import xiapi
import cv2
import numpy as np

EXPOSURE_US = 500000
RESIZE_TO = None
WINDOW_NAME = "Detekcia geometrickych tvarov - XIMEA"

# nastav si podľa reálnych objektov
MIN_SHAPE_AREA = 1500

# Hough kružnice - prísnejšie parametre
HOUGH_DP = 1
HOUGH_MINDIST = 20
HOUGH_PARAM1 = 50
HOUGH_PARAM2 = 30
HOUGH_MINRADIUS = 0
HOUGH_MAXRADIUS = 0


def is_valid_circle(gray, edges, x, y, r):
    h, w = gray.shape

    # kružnica nesmie byť príliš pri okraji
    if x - r < 0 or y - r < 0 or x + r >= w or y + r >= h:
        return False

    # maska vnútra kružnice
    mask_fill = np.zeros_like(gray, dtype=np.uint8)
    cv2.circle(mask_fill, (x, y), r, 255, -1)

    # maska obvodu kružnice
    mask_ring = np.zeros_like(gray, dtype=np.uint8)
    cv2.circle(mask_ring, (x, y), r, 255, 2)

    # 1. Overenie cez hrany na obvode
    ring_edges = cv2.bitwise_and(edges, mask_ring)
    ring_pixels = np.count_nonzero(mask_ring)
    edge_pixels = np.count_nonzero(ring_edges)

    if ring_pixels == 0:
        return False

    edge_ratio = edge_pixels / ring_pixels

    # ak na obvode skoro nie sú hrany, je to falošný nález
    if edge_ratio < 0.12:
        return False

    # 2. Overenie cez kontúry vo vnútri kružnice
    contours, _ = cv2.findContours(mask_fill, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # toto samotné nestačí, preto spravíme ešte výrez z pôvodných hrán
    x1 = max(0, x - r - 5)
    y1 = max(0, y - r - 5)
    x2 = min(w, x + r + 5)
    y2 = min(h, y + r + 5)

    roi_edges = edges[y1:y2, x1:x2]
    roi_gray = gray[y1:y2, x1:x2]

    # v ROI hľadáme kontúry
    cnts, _ = cv2.findContours(roi_edges.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    found_round_contour = False

    for cnt in cnts:
        area = cv2.contourArea(cnt)
        if area < 200:
            continue

        perimeter = cv2.arcLength(cnt, True)
        if perimeter == 0:
            continue

        circularity = 4 * np.pi * area / (perimeter * perimeter)

        # kruh má circularity blízko 1
        if circularity > 0.75:
            found_round_contour = True
            break

    if not found_round_contour:
        return False

    return True


def detect_shapes(frame):
    output = frame.copy()

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    gray_blur = cv2.medianBlur(gray, 5)

    # pre ostatné tvary aj na overenie kružníc
    edges = cv2.Canny(gray_blur, 70, 170)

    circle_count = 0
    triangle_count = 0
    square_count = 0
    rectangle_count = 0

    circle_mask = np.zeros_like(gray)

    # =========================
    # 1. Hough kružnice + validácia
    # =========================
    circles = cv2.HoughCircles(
        gray_blur,
        cv2.HOUGH_GRADIENT,
        dp=HOUGH_DP,
        minDist=HOUGH_MINDIST,
        param1=HOUGH_PARAM1,
        param2=HOUGH_PARAM2,
        minRadius=HOUGH_MINRADIUS,
        maxRadius=HOUGH_MAXRADIUS
    )

    accepted_circles = []

    if circles is not None:
        circles = np.uint16(np.around(circles))

        for i in circles[0, :]:
            x, y, r = int(i[0]), int(i[1]), int(i[2])

            if is_valid_circle(gray_blur, edges, x, y, r):
                accepted_circles.append((x, y, r))

    # odstránenie duplikátov blízko seba
    filtered_circles = []
    for x, y, r in accepted_circles:
        duplicate = False
        for fx, fy, fr in filtered_circles:
            dist = np.sqrt((x - fx) ** 2 + (y - fy) ** 2)
            if dist < 0.5 * max(r, fr):
                duplicate = True
                break
        if not duplicate:
            filtered_circles.append((x, y, r))

    for x, y, r in filtered_circles:
        circle_count += 1

        cv2.circle(output, (x, y), r, (0, 255, 0), 2)
        cv2.circle(output, (x, y), 3, (0, 0, 255), -1)

        cv2.putText(output, "Kruznica", (x - 45, y - r - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        cv2.putText(output, f"S({x},{y})", (x + 10, y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 255), 1)

        cv2.circle(circle_mask, (x, y), r + 8, 255, -1)

    # =========================
    # 2. Ostatné tvary
    # =========================
    edges_no_circles = cv2.bitwise_and(edges, cv2.bitwise_not(circle_mask))

    contours, _ = cv2.findContours(edges_no_circles, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area < MIN_SHAPE_AREA:
            continue

        perimeter = cv2.arcLength(cnt, True)
        if perimeter == 0:
            continue

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
            if h == 0:
                continue

            aspect_ratio = w / float(h)

            if 0.92 <= aspect_ratio <= 1.08:
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

    # =========================
    # 3. Info v jednom okne
    # =========================
    info_lines = [
        f"Kruznice: {circle_count}",
        f"Trojuholniky: {triangle_count}",
        f"Stvorce: {square_count}",
        f"Obdlzniky: {rectangle_count}",
        "Q = quit"
    ]

    y0 = 30
    for idx, text in enumerate(info_lines):
        cv2.putText(output, text, (10, y0 + idx * 28),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

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
        print("Q = koniec")

        while True:
            cam.get_image(img)
            frame = img.get_image_data_numpy()

            # RGB32 -> RGB
            frame = frame[:, :, :3]

            # RGB -> BGR
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