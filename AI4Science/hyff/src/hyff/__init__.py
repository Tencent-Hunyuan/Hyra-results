"""HyFF: machine-learning force fields for molecular simulation.

    import hyff
    model = hyff.load("presto")
    energy, forces = model.energy_forces(positions, numbers)

See README.md for engines, molecular dynamics, batched inference, the ASE calculator
(``hyff.HyFFCalculator``, needs ase) and fine-tuning (``hyff.finetune``).
"""

__version__ = '0.3.0'

__all__ = ['load', 'create_engine', 'Simulation', 'predict_batch', 'finetune', 'Model', 'Engine', 'MODELS']


def __getattr__(name):
    # Importing hyff stays free of torch and CUDA until an API object is used.
    if name in {'load', 'create_engine', 'Simulation', 'predict_batch', 'Model', 'Engine'}:
        from hyff import api
        value = getattr(api, name)
    elif name == 'finetune':
        from hyff.finetuning import finetune as value
    elif name == 'HyFFCalculator':
        # Not in __all__: it needs the optional ase package.
        from hyff.calculator import HyFFCalculator as value
    elif name == 'MODELS':
        from hyff.registry import MODELS as value
    else:
        raise AttributeError(f'module {__name__!r} has no attribute {name!r}')
    globals()[name] = value
    return value


def __dir__():
    return sorted(set(globals()) | set(__all__) | {'HyFFCalculator'})
