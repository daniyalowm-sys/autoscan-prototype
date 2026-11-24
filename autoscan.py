from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from pathlib import Path

import cv2
import numpy as np


SUPPORTED_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}
PROTOTYPE_VERSION = "0.1.0"
ALGORITHM_VERSION = "heuristic_baseline_v0.1"


@dataclass
class Candidate:
    x: int
    y: int
    width: int
    height: int
    area_px: float
    score: float


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Ранний прототип выделения подозрительных следов жидкости на днище."
    )
    parser.add_argument("--input", required=True, type=Path, help="Файл или папка с изображениями")
    parser.add_argument("--output", required=True, type=Path, help="Папка для результатов")
    parser.add_argument("--min-area", type=int, default=350, help="Минимальная площадь кандидата, пикселей")
    parser.add_argument("--max-width", type=int, default=1600, help="Максимальная ширина обработки")
    parser.add_argument("--max-candidates", type=int, default=12, help="Максимальное число кандидатов")
    parser.add_argument("--save-debug", action="store_true", help="Сохранять диагностическую карту")
    parser.add_argument("--version", action="version", version=f"AutoScan {PROTOTYPE_VERSION}")
    return parser.parse_args()


def list_images(source: Path) -> list[Path]:
    if source.is_file():
        if source.suffix.lower() not in SUPPORTED_EXTENSIONS:
            raise ValueError(f"Неподдерживаемое расширение: {source.suffix}")
        return [source]
    if not source.is_dir():
        raise FileNotFoundError(source)
    return sorted(
        path for path in source.rglob("*") if path.is_file() and path.suffix.lower() in SUPPORTED_EXTENSIONS
    )


def read_image(path: Path) -> np.ndarray | None:
    """Read an image through memory so Windows paths with Cyrillic work reliably."""
    try:
        encoded = np.fromfile(str(path), dtype=np.uint8)
    except OSError:
        return None
    if encoded.size == 0:
        return None
    return cv2.imdecode(encoded, cv2.IMREAD_COLOR)


def write_image(path: Path, image: np.ndarray) -> None:
    suffix = path.suffix or ".png"
    success, encoded = cv2.imencode(suffix, image)
    if not success:
        raise ValueError(f"Не удалось закодировать изображение: {path}")
    encoded.tofile(str(path))


def resize_if_needed(image: np.ndarray, max_width: int) -> tuple[np.ndarray, float]:
    height, width = image.shape[:2]
    if width <= max_width:
        return image, 1.0
    scale = max_width / width
    resized = cv2.resize(image, (max_width, int(round(height * scale))), interpolation=cv2.INTER_AREA)
    return resized, scale


def build_candidate_mask(image: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Return a conservative heuristic mask and a 0..255 diagnostic score map."""
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    lightness, channel_a, channel_b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    lightness = clahe.apply(lightness)

    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    hue, saturation, value = cv2.split(hsv)

    local_background = cv2.GaussianBlur(lightness, (0, 0), 21)
    darker_than_background = cv2.subtract(local_background, lightness)

    contrast_score = np.clip(darker_than_background.astype(np.float32) * 4.0, 0, 255)
    saturation_score = np.clip((saturation.astype(np.float32) - 20.0) * 2.0, 0, 255)

    warm_colour = cv2.inRange(hsv, np.array([3, 35, 25]), np.array([38, 255, 235]))
    warm_score = warm_colour.astype(np.float32)

    specular = cv2.inRange(hsv, np.array([0, 0, 185]), np.array([179, 80, 255]))
    specular_neighbourhood = cv2.dilate(specular, np.ones((17, 17), np.uint8), iterations=1)

    # Colour evidence is weighted more strongly than simple darkness. This reduces
    # false candidates on bolts, seams and shadows while retaining brown/oily areas.
    score = 0.25 * contrast_score + 0.45 * saturation_score + 0.30 * warm_score
    score = np.where(specular_neighbourhood > 0, np.minimum(score + 10, 255), score)
    score = cv2.GaussianBlur(score.astype(np.uint8), (7, 7), 0)

    _, mask = cv2.threshold(score, 78, 255, cv2.THRESH_BINARY)
    kernel_open = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    kernel_close = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (13, 13))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel_open)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel_close)
    return mask, score


def find_candidates(
    mask: np.ndarray,
    score_map: np.ndarray,
    min_area: int,
    max_candidates: int,
) -> tuple[list[Candidate], list[np.ndarray]]:
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    image_area = mask.shape[0] * mask.shape[1]
    selected: list[tuple[Candidate, np.ndarray]] = []

    for contour in contours:
        area = float(cv2.contourArea(contour))
        if area < min_area or area > image_area * 0.22:
            continue
        x, y, width, height = cv2.boundingRect(contour)
        component_mask = np.zeros_like(mask)
        cv2.drawContours(component_mask, [contour], -1, 255, thickness=-1)
        mean_score = float(cv2.mean(score_map, mask=component_mask)[0])
        compactness = min(1.0, area / max(width * height, 1))
        confidence = round(min(0.99, 0.55 * (mean_score / 255.0) + 0.45 * compactness), 3)
        selected.append(
            (
                Candidate(x=x, y=y, width=width, height=height, area_px=round(area, 1), score=confidence),
                contour,
            )
        )

    selected.sort(key=lambda item: (item[0].score, item[0].area_px), reverse=True)
    selected = selected[:max_candidates]
    return [item[0] for item in selected], [item[1] for item in selected]


def annotate(image: np.ndarray, contours: list[np.ndarray], candidates: list[Candidate]) -> np.ndarray:
    result = image.copy()
    overlay = image.copy()
    if contours:
        cv2.drawContours(overlay, contours, -1, (0, 120, 255), thickness=-1)
        result = cv2.addWeighted(overlay, 0.22, result, 0.78, 0)

    for index, (candidate, contour) in enumerate(zip(candidates, contours), start=1):
        cv2.drawContours(result, [contour], -1, (0, 120, 255), thickness=3)
        cv2.rectangle(
            result,
            (candidate.x, candidate.y),
            (candidate.x + candidate.width, candidate.y + candidate.height),
            (0, 120, 255),
            2,
        )
        label = f"candidate {index}: {candidate.score:.2f}"
        cv2.putText(
            result,
            label,
            (candidate.x, max(24, candidate.y - 8)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.65,
            (0, 60, 180),
            2,
            cv2.LINE_AA,
        )
    return result


def process_image(
    image_path: Path,
    output_dir: Path,
    min_area: int,
    max_width: int,
    max_candidates: int,
    save_debug: bool,
) -> dict:
    image = read_image(image_path)
    if image is None:
        raise ValueError(f"Не удалось прочитать изображение: {image_path}")

    image, scale = resize_if_needed(image, max_width)
    mask, score_map = build_candidate_mask(image)
    candidates, contours = find_candidates(mask, score_map, min_area, max_candidates)
    annotated = annotate(image, contours, candidates)

    stem = image_path.stem
    annotated_path = output_dir / f"{stem}_annotated.jpg"
    mask_path = output_dir / f"{stem}_mask.png"
    write_image(annotated_path, annotated)
    write_image(mask_path, mask)
    debug_path: Path | None = None
    if save_debug:
        debug_path = output_dir / f"{stem}_debug.jpg"
        heatmap = cv2.applyColorMap(score_map, cv2.COLORMAP_TURBO)
        write_image(debug_path, heatmap)

    return {
        "source": str(image_path),
        "processed_width": int(image.shape[1]),
        "processed_height": int(image.shape[0]),
        "resize_scale": round(scale, 4),
        "algorithm": ALGORITHM_VERSION,
        "diagnostic_only": True,
        "candidate_count": len(candidates),
        "candidates": [asdict(candidate) for candidate in candidates],
        "outputs": {
            "annotated": str(annotated_path),
            "mask": str(mask_path),
            "debug": str(debug_path) if debug_path else None,
        },
    }


def main() -> None:
    args = parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    images = list_images(args.input)
    if not images:
        raise SystemExit("В указанной папке нет поддерживаемых изображений.")

    report = {
        "project": "AutoScan",
        "prototype_version": PROTOTYPE_VERSION,
        "warning": "Результат является подсказкой для механика и не заменяет диагностику.",
        "images": [],
    }
    for image_path in images:
        report["images"].append(
            process_image(
                image_path=image_path,
                output_dir=args.output,
                min_area=args.min_area,
                max_width=args.max_width,
                max_candidates=args.max_candidates,
                save_debug=args.save_debug,
            )
        )

    report_path = args.output / "report.json"
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Обработано изображений: {len(images)}")
    print(f"Отчёт: {report_path}")


if __name__ == "__main__":
    main()

