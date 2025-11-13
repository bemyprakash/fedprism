"""
Model Definition - YOLO Model Wrapper
Imports and wraps the YOLOv12 model from yolo.py
"""

from yolo import YOLOv12Model


def create_yolo_model(num_classes=80, img_size=320, device='cpu', width_mult=0.25, depth_mult=0.33):
    """
    Create a YOLOv12 model instance
    
    Args:
        num_classes: Number of object classes
        img_size: Input image size (assumes square images)
        device: Device to run on ('cpu' or 'cuda')
        width_mult: Width multiplier for model size
        depth_mult: Depth multiplier for model size
    
    Returns:
        YOLOv12Model instance
    """
    return YOLOv12Model(
        num_classes=num_classes,
        device=device,
        img_size=img_size,
        width_mult=width_mult,
        depth_mult=depth_mult
    )

