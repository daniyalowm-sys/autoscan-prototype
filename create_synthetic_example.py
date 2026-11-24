from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import numpy as np


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Создать синтетический стендовый кадр для smoke-теста.")
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.output.parent.mkdir(parents=True, exist_ok=True)

    rng = np.random.default_rng(42)
    height, width = 720, 1280
    base = np.full((height, width, 3), (125, 132, 137), dtype=np.float32)
    gradient = np.linspace(-22, 18, width, dtype=np.float32)[None, :, None]
    noise = rng.normal(0, 9, size=(height, width, 1)).astype(np.float32)
    image = np.clip(base + gradient + noise, 0, 255).astype(np.uint8)
    image = cv2.GaussianBlur(image, (5, 5), 0)

    cv2.line(image, (90, 360), (1190, 350), (75, 78, 80), 8)
    for x, y in [(210, 155), (1040, 170), (260, 560), (1010, 550)]:
        cv2.circle(image, (x, y), 35, (82, 86, 90), -1)
        cv2.circle(image, (x, y), 18, (165, 168, 170), -1)

    stain_layer = image.copy()
    cv2.ellipse(stain_layer, (700, 390), (175, 65), -8, 0, 360, (34, 62, 90), -1)
    cv2.ellipse(stain_layer, (770, 412), (88, 28), 8, 0, 360, (24, 45, 70), -1)
    cv2.ellipse(stain_layer, (650, 368), (62, 12), -12, 0, 360, (210, 220, 225), -1)
    image = cv2.addWeighted(stain_layer, 0.72, image, 0.28, 0)
    image = cv2.GaussianBlur(image, (3, 3), 0)

    cv2.putText(
        image,
        "SYNTHETIC BENCH IMAGE - NOT A REAL VEHICLE",
        (30, 685),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (235, 235, 235),
        2,
        cv2.LINE_AA,
    )
    cv2.imwrite(str(args.output), image)
    print(args.output)


if __name__ == "__main__":
    main()

