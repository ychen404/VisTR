from .vistr import build, build_early_exit


def build_model(args):
    return build(args)

def build_model_early_exit(args):
    return build_early_exit(args)
