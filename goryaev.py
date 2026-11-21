import argparse
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np


def detect(gray, corners, diameter, bright):
    field = np.zeros_like(gray)
    forbidden = np.zeros_like(gray)
    cv2.fillConvexPoly(field, corners, 255)
    cv2.polylines(forbidden, [corners[1:4]], False, 255, 1)

    # Выделяем клетки относительно местного фона.
    size = 2 * diameter + 1
    mode = cv2.THRESH_BINARY if bright else cv2.THRESH_BINARY_INV
    mask = cv2.adaptiveThreshold(cv2.medianBlur(gray, 3), 255,
                                 cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                 mode, size, -5 if bright else 5)
    # Убираем длинные прямые штрихи горизонтальной и вертикальной сетки.
    for shape in [(size, 1), (1, size)]:
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, shape)
        lines = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        mask = cv2.subtract(mask, lines)

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL,
                                   cv2.CHAIN_APPROX_SIMPLE)
    cells = []
    for c in contours:
        area = cv2.contourArea(c)
        perimeter = cv2.arcLength(c, True)
        if not (0.2 * diameter**2 <= area <= 2 * diameter**2):
            continue
        if perimeter == 0 or 4 * np.pi * area / perimeter**2 < 0.45:
            continue
        x, y, w, h = cv2.boundingRect(c)
        local = np.zeros((h, w), np.uint8)
        cv2.drawContours(local, [(c - (x, y)).astype(np.int32)], -1, 255, cv2.FILLED)
        region = np.s_[y:y+h, x:x+w]
        if np.any(local & field[region]) and not np.any(local & forbidden[region]):
            m = cv2.moments(c)
            cells.append((m["m10"] / m["m00"], m["m01"] / m["m00"]))
    return cells


def main():
    p = argparse.ArgumentParser()
    src = p.add_mutually_exclusive_group(required=True)
    src.add_argument("--image", type=Path)
    src.add_argument("--camera", type=int)
    p.add_argument("--cols", type=int, default=1)
    p.add_argument("--rows", type=int, default=1)
    p.add_argument("--dilution", type=float, default=1)
    p.add_argument("--diameter", type=int, required=True)
    p.add_argument("--bright", action="store_true")
    a = p.parse_args()
    if min(a.rows, a.cols, a.dilution) <= 0 or a.diameter < 3 or not np.isfinite(a.dilution):
        p.error("Проверьте число квадратов, разведение и диаметр (минимум 3 пикселя).")

    if a.image:
        frame = cv2.imdecode(np.fromfile(a.image, np.uint8), cv2.IMREAD_COLOR)
    else:
        cap = cv2.VideoCapture(a.camera)
        frame = None
        try:
            for _ in range(10):
                ok, frame = cap.read()
                if not ok:
                    frame = None
                    break
        finally:
            cap.release()
    if frame is None:
        raise SystemExit("Не удалось получить изображение.")

    fig, ax = plt.subplots()
    ax.imshow(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    ax.set_title("Углы поля: левый верхний → правый верхний →\n"
                 "правый нижний → левый нижний")
    pts = plt.ginput(4, timeout=0)
    if len(pts) != 4:
        return
    corners = np.rint(pts).astype(np.int32)
    if not cv2.isContourConvex(corners) or cv2.contourArea(corners, oriented=True) <= 0:
        raise SystemExit("Проверьте порядок углов.")
    cells = detect(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY),
                   corners, a.diameter, a.bright)
    ax.plot(*corners[[3, 0, 1]].T, color="lime")
    ax.plot(*corners[[1, 2, 3]].T, color="red")
    marks, = ax.plot([], [], "y+", ms=9)
    volume = a.cols * a.rows * 0.2**2 * 0.1 / 1000

    def refresh():
        xy = np.asarray(cells).reshape(-1, 2)
        marks.set_data(xy[:, 0], xy[:, 1])
        ax.set_title(f"{len(cells)} клеток | {len(cells)*a.dilution/volume:,.0f} клеток/мл\n"
                     "ЛКМ добавить, ПКМ убрать, Enter подтвердить")
        fig.canvas.draw_idle()

    def click(e):
        if e.inaxes != ax or getattr(fig.canvas.toolbar, "mode", ""):
            return
        point = (e.xdata, e.ydata)
        if e.button == 1:
            cells.append(point)
        elif e.button == 3 and cells:
            distances = np.linalg.norm(np.asarray(cells) - point, axis=1)
            if distances.min() <= a.diameter:
                cells.pop(int(distances.argmin()))
        refresh()

    def finish(e):
        if e.key == "enter":
            print(f"N = {len(cells)}, V = {volume:.8g} мл, "
                  f"C = {len(cells)*a.dilution/volume:.6g} клеток/мл")
            plt.close(fig)

    fig.canvas.mpl_connect("button_press_event", click)
    fig.canvas.mpl_connect("key_press_event", finish)
    refresh()
    plt.show()


if __name__ == "__main__":
    main()
