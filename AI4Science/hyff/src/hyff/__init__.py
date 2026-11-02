"""HyFF: machine-learning force fields for molecular simulation.

    import hyff
    model = hyff.load("presto")
    energy, forces = model.energy_forces(positions, numbers)

See README.md for engines, molecular dynamics and batched inference.
"""

__version__ = '0.2.0'

__all__ = ['load', 'create_engine', 'Simulation', 'predict_batch', 'Model', 'Engine', 'MODELS']


def __getattr__(name):
    # Importing hyff stays free of torch and CUDA until an API object is used.
    if name in {'load', 'create_engine', 'Simulation', 'predict_batch', 'Model', 'Engine'}:
        from hyff import api
        value = getattr(api, name)
    elif name == 'MODELS':
        from hyff.registry import MODELS as value
    else:
        raise AttributeError(f'module {__name__!r} has no attribute {name!r}')
    globals()[name] = value
    return value


def __dir__():
    return sorted(set(globals()) | set(__all__))
