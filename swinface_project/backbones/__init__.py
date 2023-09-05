

def get_model(name, **kwargs):

    if name == "swin_t":
        num_features = kwargs.get("num_features", 512)
        from .swin import SwinTransformer
        return SwinTransformer(num_classes=num_features)


    else:
        raise ValueError()
