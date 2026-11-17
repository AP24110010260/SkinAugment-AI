import random
import numpy as np
from PIL import Image, ImageEnhance, ImageFilter

def generate_augmentations(image, count=5):
    image = image.convert("RGB")
    results = []

    for _ in range(count):
        augmented = image.copy()

        # Small camera angle variation
        angle = random.uniform(-7, 7)
        augmented = augmented.rotate(
            angle,
            resample=Image.Resampling.BICUBIC,
            expand=False,
            fillcolor=(235, 235, 235)
        )

        # Realistic lighting variation
        augmented = ImageEnhance.Brightness(augmented).enhance(
            random.uniform(0.82, 1.18)
        )

        # Camera exposure and contrast variation
        augmented = ImageEnhance.Contrast(augmented).enhance(
            random.uniform(0.88, 1.12)
        )

        # Mild color variation
        augmented = ImageEnhance.Color(augmented).enhance(
            random.uniform(0.90, 1.10)
        )

        # Slight sharpness variation
        augmented = ImageEnhance.Sharpness(augmented).enhance(
            random.uniform(0.90, 1.15)
        )

        # Occasional mild camera blur
        if random.random() < 0.25:
            augmented = augmented.filter(
                ImageFilter.GaussianBlur(
                    radius=random.uniform(0.2, 0.6)
                )
            )

        # Subtle sensor noise
        if random.random() < 0.30:
            array = np.asarray(augmented).astype(np.float32)
            noise = np.random.normal(0, 2.0, array.shape)
            array = np.clip(array + noise, 0, 255).astype(np.uint8)
            augmented = Image.fromarray(array)

        results.append(augmented)

    return results
