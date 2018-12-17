class TargetType:
    geometry = (1 << 0)
    reproject = (1 << 1)
    normal = (1 << 2)
    depth = (1 << 3)

    pretrain = geometry
    finetune = reproject
    full = geometry | reproject
    extend = geometry | reproject | normal | depth
