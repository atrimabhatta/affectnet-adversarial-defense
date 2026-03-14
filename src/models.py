# src/models.py
import timm

def get_model(name, num_classes):
    if name == "resnet50":
        return timm.create_model("resnet50", pretrained=True, num_classes=num_classes,
                                 drop_rate=0.4, drop_path_rate=0.1)
    elif name == "convnext_tiny":
        return timm.create_model("convnext_tiny", pretrained=True, num_classes=num_classes,
                                 drop_rate=0.5, drop_path_rate=0.2)
    elif name == "efficientnetv2_rw_s":
        return timm.create_model("efficientnetv2_rw_s", pretrained=True, num_classes=num_classes,
                                 drop_rate=0.4, drop_path_rate=0.1)
    else:
        raise ValueError(f"Unknown model: {name}")