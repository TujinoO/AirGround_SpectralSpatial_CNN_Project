import argparse
import shutil
from pathlib import Path

import numpy as np
from matplotlib import colors as mcolors
from PIL import Image, ImageFilter


def _dilate_mask(mask: np.ndarray, size: int = 3, iterations: int = 2) -> np.ndarray:
    mask_img = Image.fromarray(mask.astype(np.uint8) * 255)
    for _ in range(max(1, iterations)):
        mask_img = mask_img.filter(ImageFilter.MaxFilter(size=size))
    return np.asarray(mask_img) > 0


def _dilate_colors(masked_rgb: np.ndarray, size: int = 3, iterations: int = 2) -> np.ndarray:
    channels = []
    for channel_idx in range(3):
        channel_img = Image.fromarray(masked_rgb[..., channel_idx])
        for _ in range(max(1, iterations)):
            channel_img = channel_img.filter(ImageFilter.MaxFilter(size=size))
        channels.append(np.asarray(channel_img))
    return np.stack(channels, axis=-1)


def restyle_curve_image(image_path: Path, backup: bool = True) -> None:
    image_path = image_path.resolve()
    if not image_path.exists():
        raise FileNotFoundError(f'Image not found: {image_path}')

    if backup:
        backup_path = image_path.with_name(f'{image_path.stem}_original{image_path.suffix}')
        if not backup_path.exists():
            shutil.copy2(image_path, backup_path)

    img = Image.open(image_path).convert('RGBA')
    arr = np.asarray(img).copy()
    rgb = arr[..., :3].astype(np.float32) / 255.0
    alpha = arr[..., 3]
    hsv = mcolors.rgb_to_hsv(rgb)

    # Lift the background to pure white while keeping dark axes/text intact.
    background_mask = (hsv[..., 1] < 0.16) & (hsv[..., 2] > 0.74) & (alpha > 0)
    arr[background_mask, :3] = 255

    # Treat saturated pixels as plot curves or legend color keys.
    curve_mask = (hsv[..., 1] > 0.22) & (hsv[..., 2] > 0.15) & (alpha > 0)
    boosted_hsv = hsv.copy()
    boosted_hsv[..., 1] = np.clip(boosted_hsv[..., 1] * 1.55 + 0.05, 0.0, 1.0)
    boosted_hsv[..., 2] = np.clip(boosted_hsv[..., 2] * 1.12 + 0.03, 0.0, 1.0)
    boosted_rgb = (mcolors.hsv_to_rgb(boosted_hsv) * 255.0).astype(np.uint8)

    masked_rgb = np.zeros_like(arr[..., :3], dtype=np.uint8)
    masked_rgb[curve_mask] = boosted_rgb[curve_mask]

    thick_mask = _dilate_mask(curve_mask, size=3, iterations=2)
    thick_rgb = _dilate_colors(masked_rgb, size=3, iterations=2)
    arr[thick_mask, :3] = thick_rgb[thick_mask]

    # Keep the canvas fully opaque and clean.
    arr[..., 3] = 255
    Image.fromarray(arr).save(image_path)


def main() -> None:
    parser = argparse.ArgumentParser(description='Restyle existing training curve images without changing plotted data.')
    parser.add_argument('images', nargs='+', help='One or more image paths to restyle.')
    parser.add_argument('--no-backup', action='store_true', help='Do not create *_original backups.')
    args = parser.parse_args()

    for image in args.images:
        restyle_curve_image(Path(image), backup=not args.no_backup)
        print(f'Restyled: {Path(image).resolve()}')


if __name__ == '__main__':
    main()
