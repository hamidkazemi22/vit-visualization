_layers = [256, 256, 256, 256, 256, 256, 256, 256, 256, 256, 256, 384, 384, 384, 384, 384, 384, 384, 384, 384, 384, 512,
           512, 512, 512, 512, 512, 512, 512, 512, 512, 640, 640, 640, 640, 640, 640, 640, 640, 640, 640, 768, 768, 768,
           768, 768, 768, 768, 768, 768, 768, 896]


def layer_range(stop=-1):
    for li in range(len(_layers)):
        if stop != -1 and li > stop:
            break
        yield li


def feature_range(layer: int, stop=-1):
    for fi in range(_layers[layer]):
        if stop != -1 and fi > stop:
            break
        yield fi
