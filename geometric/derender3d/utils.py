def to_cpu(v):
    return v.detach().cpu()


def to_numpy(v):
    return v.detach().cpu().numpy()
