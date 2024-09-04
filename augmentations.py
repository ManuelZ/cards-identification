import albumentations as A
import cv2
import siamese.config as config

al_augmentations = A.Compose(
    [
        A.HueSaturationValue(p=0.5),
        A.RandomBrightnessContrast(p=0.5),
        A.RandomGamma(gamma_limit=(60, 120), p=0.5),
        A.ISONoise(p=0.5),
        A.PixelDropout(p=0.5),
        A.Blur(blur_limit=5),
        A.CoarseDropout(p=0.1),
        # Zoom out only
        A.ShiftScaleRotate(
            shift_limit=0,
            rotate_limit=0,
            scale_limit=(-0.1, 0),  # Zoom out only
            border_mode=cv2.BORDER_CONSTANT,
            value=config.WHITE,
            p=0.5,
        ),
        A.Perspective(
            fit_output=True, pad_mode=cv2.BORDER_CONSTANT, pad_val=config.WHITE, p=0.3
        ),
        # Shift only
        A.ShiftScaleRotate(
            shift_limit=0.05,
            rotate_limit=0,
            scale_limit=0,
            border_mode=cv2.BORDER_CONSTANT,
            value=config.WHITE,
            p=0.5,
        ),
        A.SafeRotate(
            limit=10, border_mode=cv2.BORDER_CONSTANT, value=config.WHITE, p=0.1
        ),
        A.OpticalDistortion(border_mode=cv2.BORDER_CONSTANT, value=config.WHITE),
    ]
)
