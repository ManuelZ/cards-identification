import albumentations as A
import cv2
from siamese.config import WHITE

al_augmentations = A.Compose(
    [
        A.Blur(blur_limit=5),
        A.CoarseDropout(p=0.1),
        # Zoom
        A.ShiftScaleRotate(
            shift_limit=0,
            rotate_limit=0,
            scale_limit=(-0.2, 0),  # Zoom out only
            border_mode=cv2.BORDER_CONSTANT,
            value=WHITE,
            p=1.0,
        ),
        A.Perspective(
            fit_output=True, pad_mode=cv2.BORDER_CONSTANT, pad_val=WHITE, p=0.3
        ),
        # Shift only
        A.ShiftScaleRotate(
            shift_limit=0.05,
            rotate_limit=0,
            scale_limit=0,
            border_mode=cv2.BORDER_CONSTANT,
            value=WHITE,
            p=0.5,
        ),
        A.SafeRotate(limit=10, border_mode=cv2.BORDER_CONSTANT, value=WHITE, p=0.1),
        A.OpticalDistortion(border_mode=cv2.BORDER_CONSTANT, value=WHITE),
    ]
)
