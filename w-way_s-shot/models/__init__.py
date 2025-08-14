from . import Minkowski, ppat


def make(scaling=4, name="ppat"):
    if name == "ppat":
        model = ppat.make(scaling)
    elif name == "MinkResNet34":
        model = Minkowski.MinkResNet34()
    return model
