"""Command-line interface: ``hyff models|download|infer|md|predict|finetune``."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

MODEL_CHOICES = ('presto', 'vivace', 'moderato', 'andante', 'largo')


class _ModelAction(argparse.Action):
    """Accept the public model names (and silent aliases)."""

    def __call__(self, parser, namespace, value, option_string=None):
        from hyff.registry import model_name
        name = model_name(value)
        if name is None:
            parser.error(f'argument {option_string}: invalid choice {value!r} '
                         f'(choose from {", ".join(MODEL_CHOICES)})')
        setattr(namespace, self.dest, name)


def _add_model(parser, required=False):
    parser.add_argument('--model', action=_ModelAction, default=None, required=required,
                        metavar='{' + ','.join(MODEL_CHOICES) + '}',
                        help='model name (default: presto, or inferred from --weights)')
    parser.add_argument('--weights', help='explicit checkpoint: a .safetensors file with its sibling .json, '
                        'a sharded .safetensors.index.json, or a directory holding one')


def _parser():
    parser = argparse.ArgumentParser(prog='hyff', description='HyFF machine-learning force fields')
    from hyff import __version__
    parser.add_argument('--version', action='version', version=f'hyff {__version__}')
    # --debug is accepted before or after the command: full tracebacks (and, for md, diagnostics in result.json).
    common = argparse.ArgumentParser(add_help=False)
    common.add_argument('--debug', action='store_true', default=argparse.SUPPRESS,
                        help='show full tracebacks on errors (md: also keep engine diagnostics in result.json)')
    parser.add_argument('--debug', action='store_true', help=argparse.SUPPRESS)
    commands = parser.add_subparsers(dest='command', required=True)

    commands.add_parser('models', parents=[common],
                        help='list the models, their download sizes and whether they are installed')

    download = commands.add_parser('download', parents=[common],
                                   help='download and verify model weights into the cache directory')
    download.add_argument('models', nargs='+', metavar='MODEL', help='model names, or "all"')

    infer = commands.add_parser('infer', parents=[common], help='energy and forces of one structure')
    _add_model(infer)
    infer.add_argument('--system', required=True, help='NPZ with positions_A and atomic_numbers')
    infer.add_argument('--output', required=True, help='output NPZ with energy_eV and forces_eV_A')
    infer.add_argument('--device', default='cuda', help='cuda (default) or cpu')

    md = commands.add_parser('md', parents=[common], help='molecular dynamics on a prepared system NPZ')
    _add_model(md)
    md.add_argument('--system', required=True, help='prepared system NPZ (see docs/inputs.md)')
    md.add_argument('--output', required=True, help='output directory')
    md.add_argument('--steps', type=int, default=10000)
    md.add_argument('--timestep-fs', type=float, default=1.0)
    md.add_argument('--temperature-K', type=float, default=300.0)
    md.add_argument('--tau-ps', type=float, default=0.5, help='Bussi thermostat time constant; 0 runs NVE')
    md.add_argument('--seed', type=int, default=1)
    md.add_argument('--chunk', type=int, default=100, help='steps per device launch, 1-1024 (thermostat interval)')
    md.add_argument('--save-every', type=int, default=500, help='steps between saved frames (the first frame is t=0)')
    md.add_argument('--max-wall-seconds', type=float, help='stop cleanly after this wall time (s)')
    md.add_argument('--restart', help='final_state.npz of an earlier run to continue from')
    md.add_argument('--recenter-interval-ps', type=float,
                    help='translate bonded groups back into the cell at this interval (ps)')
    md.add_argument('--molecule-topology',
                    help='OpenMM system.xml defining the recentering and barostat groups '
                         '(default: constraints + bond_pairs)')
    md.add_argument('--pressure-bar', type=float,
                    help='NPT: Monte Carlo barostat at this pressure (bar); needs the thermostat')
    md.add_argument('--barostat', choices=('isotropic', 'semi-isotropic', 'anisotropic'), default='isotropic',
                    help='coupled cell axes: all, xy | z, or x | y | z (the last two need an orthorhombic box)')
    md.add_argument('--barostat-interval', type=int, default=100, help='steps between volume moves')
    md.add_argument('--hmr-mass', type=float, help='hydrogen mass (amu) for non-water X-H bonds')
    md.add_argument('--inclusive-hmr', action='store_true',
                    help='3 amu for every hydrogen, including water; heavy atoms compensate')
    md.add_argument('--no-recovery', action='store_true',
                    help='NVT: stop at the first rejected step instead of rolling back')
    md.add_argument('--gpus', type=int, choices=(1, 2, 4, 8), default=1,
                    help='experimental multi-GPU largo MD (automatic card policy)')
    _add_engine_options(md)

    finetune = commands.add_parser('finetune', parents=[common],
                                   help='fine-tune a model on reference energies and forces (see docs/finetune.md)')
    _add_model(finetune)
    finetune.add_argument('--train', help='training structures: extended XYZ (or another ASE-readable file) '
                          'with energies and forces, or an NPZ (see docs/finetune.md)')
    finetune.add_argument('--valid', help='validation structures (default: a random --valid-fraction of --train)')
    finetune.add_argument('--valid-fraction', type=float, default=0.05)
    finetune.add_argument('--output', help='output directory (checkpoint, best/ and final/ weights)')
    finetune.add_argument('--epochs', type=int, default=10)
    finetune.add_argument('--lr', type=float, help='peak learning rate (default: per model, 3e-5 for presto '
                          'to 3e-6 for largo; see docs/finetune.md)')
    finetune.add_argument('--batch-atoms', type=int, default=2048, help='atoms per batch')
    finetune.add_argument('--energy-weight', type=float, default=10.0,
                          help='weight of the squared energy error per atom, (eV/atom)^2')
    finetune.add_argument('--force-weight', type=float, default=1.0,
                          help='weight of the squared force error per component, (eV/A)^2')
    finetune.add_argument('--freeze', nargs='+', default=[], metavar='PATTERN',
                          help='keep parameters matching these name patterns fixed (e.g. "embedding.*" "blocks.0.*")')
    finetune.add_argument('--train-only', nargs='+', default=[], metavar='PATTERN',
                          help='train only parameters matching these name patterns (e.g. "readouts.*")')
    finetune.add_argument('--no-refit-element-energies', action='store_true',
                          help='keep the per-element energy offsets instead of refitting them first')
    finetune.add_argument('--ema-decay', type=float, help='keep an exponential moving average of the weights')
    finetune.add_argument('--energy-key', help='atoms.info key of the energy (default: the frame\'s energy)')
    finetune.add_argument('--forces-key', help='atoms.arrays key of the forces (default: the frame\'s forces)')
    finetune.add_argument('--device', help='cuda (default when available) or cpu')
    finetune.add_argument('--gpus', type=int, default=1,
                          help='data-parallel GPUs of this machine (--batch-atoms is per GPU)')
    finetune.add_argument('--seed', type=int, default=0)
    finetune.add_argument('--resume', action='store_true', help='continue the run in --output from its checkpoint')
    finetune.add_argument('--list-parameters', action='store_true',
                          help='print the parameter groups of the model (for --freeze/--train-only) and exit')

    predict = commands.add_parser('predict', parents=[common], help='batched energies and forces of many molecules')
    _add_model(predict)
    predict.add_argument('--system', required=True,
                         help='NPZ with positions, numbers, offsets, total_charge, multiplicity')
    predict.add_argument('--output', required=True, help='output NPZ with energies_eV, forces_eV_A, offsets')
    predict.add_argument('--device', default='cuda')
    predict.add_argument('--route', choices=('native', 'reference'), default='native')
    predict.add_argument('--mode', choices=('energy', 'energy_forces'), default='energy_forces')
    predict.add_argument('--batch-size', type=int, default=0, help='molecules per batch (default 512; largo 64)')
    predict.add_argument('--atom-capacity', type=int, default=0, help='atoms per batch (default 8192; largo 2048)')
    predict.add_argument('--edge-capacity', type=int, default=0)
    predict.add_argument('--neighbor-capacity', type=int, default=0, help='multiple of 8; 0 plans from geometry')
    predict.add_argument('--bucket-policy', choices=('single', 'size_bands'), default=None)
    predict.add_argument('--precision', choices=('fp32', 'tf32x3'))
    predict.add_argument('--statistics', choices=('fp64', 'fp32'))
    return parser


def _add_engine_options(parser):
    group = parser.add_argument_group('engine options (defaults: the model\'s fastest validated settings)')
    group.add_argument('--engine', help='presto, vivace, moderato, andante, largo, reference or reference_cuda_graph')
    group.add_argument('--precision', choices=('fp32', 'tf32x3'))
    group.add_argument('--statistics', choices=('fp64', 'fp32'))
    group.add_argument('--edge-reserve', type=float, help='andante/largo edge buffer reserve (default 1.05)')
    group.add_argument('--long-range', choices=('native', 'dense', 'tree', 'auto', 'spme'),
                       help='largo long-range evaluation (default auto)')
    group.add_argument('--conservation', choices=('native', 'fp64'), help='largo charge conservation arithmetic')
    group.add_argument('--capacity-policy', choices=('native', 'partial', 'tiled', 'auto'),
                       help='andante/largo edge-frame-mixing memory route for very large systems')
    group.add_argument('--skin', type=float, default=1.5, help='neighbour-list skin (A)')
    group.add_argument('--reserve', type=float, default=1.5, help='neighbour buffer reserve')


def _models():
    from hyff.download import size_text
    from hyff.registry import MODEL_INFO, MODELS, cache_dir, download_bytes, installed
    rows = []
    for name in MODELS:
        directory = installed(name)
        status = f'installed ({directory / name})' if directory is not None else 'downloaded on first use'
        rows.append((name, MODEL_INFO[name]['parameters'], MODEL_INFO[name]['active_per_atom'],
                     size_text(download_bytes(name)), status))
    header = ('model', 'parameters', 'active/atom', 'weights', 'status')
    widths = [max(len(str(row[i])) for row in rows + [header]) for i in range(len(header))]
    for row in [header] + rows:
        print('  '.join(str(value).ljust(width) for value, width in zip(row, widths)).rstrip())
    print(f'cache directory: {cache_dir()} (set HYFF_HOME to change it)')


def _download(args):
    from hyff.registry import MODELS, install, installed, model_name, verify
    names = list(MODELS) if 'all' in args.models else [model_name(value) for value in args.models]
    unknown = [value for value, name in zip(args.models, names) if name is None]
    if unknown:
        raise SystemExit(f'hyff download: unknown model {unknown[0]!r} (choose from {", ".join(MODELS)} or all)')
    for name in dict.fromkeys(names):
        directory = installed(name)
        if directory is not None:
            try:
                verify(name, directory)
            except ValueError:
                directory = None
        if directory is None:
            directory = install(name)
            verify(name, directory)
        print(f'{name}: installed ({directory / name})')


def _model(args):
    return args.model or ('presto' if args.weights is None else None)


def _infer(args):
    import numpy as np
    import hyff
    model = hyff.load(_model(args), weights=args.weights, device=args.device)
    with np.load(args.system) as system:
        missing = [key for key in ('positions_A', 'atomic_numbers') if key not in system]
        if missing:
            raise ValueError(f'{args.system} is missing {", ".join(missing)}')
        # Values are passed unchanged; the model checks them (a non-integer multiplicity is an error).
        energy, forces = model.energy_forces(
            system['positions_A'], system['atomic_numbers'],
            system['box_A'] if 'box_A' in system else None,
            system['total_charge'] if 'total_charge' in system else 0.,
            system['multiplicity'] if 'multiplicity' in system else 1)
    target = Path(args.output)
    target.parent.mkdir(parents=True, exist_ok=True)
    np.savez(target, energy_eV=np.float64(energy), forces_eV_A=forces)
    print(json.dumps(dict(model=model.name, n_atoms=len(forces), energy_eV=energy)))


def _md(args):
    import hyff
    sim = hyff.Simulation(
        args.system, _model(args), weights=args.weights, steps=args.steps, timestep_fs=args.timestep_fs,
        temperature_K=args.temperature_K, tau_ps=args.tau_ps, output=args.output, gpus=args.gpus,
        seed=args.seed, chunk=args.chunk, save_every=args.save_every, initial_positions=args.restart,
        hmr_mass=args.hmr_mass, inclusive_hmr=args.inclusive_hmr,
        recovery='off' if args.no_recovery else 'on', engine=args.engine, precision=args.precision,
        statistics=args.statistics, edge_reserve=args.edge_reserve, long_range=args.long_range,
        conservation=args.conservation, capacity_policy=args.capacity_policy, skin=args.skin,
        reserve=args.reserve, debug=args.debug, max_wall_seconds=args.max_wall_seconds,
        recenter_interval_ps=args.recenter_interval_ps, molecule_system_xml=args.molecule_topology,
        pressure_bar=args.pressure_bar, barostat=args.barostat, barostat_interval=args.barostat_interval)
    with sim:
        result = sim.run()
    summary = {key: result.get(key) for key in ('status', 'model', 'engine', 'steps_done', 'ms_per_step', 'ns_per_day')}
    print(json.dumps(summary))
    return 0 if result.get('status') in ('complete', 'stopped') else 1


def _predict(args):
    import numpy as np
    import hyff
    from hyff.batch.collection import load_collection_npz
    values = load_collection_npz(args.system)
    result = hyff.predict_batch(_model(args), values['positions'], values['numbers'], values['offsets'],
                                values['total_charge'], values['multiplicity'], weights=args.weights,
                                device=args.device, route=args.route, mode=args.mode,
                                atom_capacity=args.atom_capacity, graph_capacity=args.batch_size,
                                edge_capacity=args.edge_capacity, neighbor_capacity=args.neighbor_capacity,
                                bucket_policy=args.bucket_policy, precision=args.precision,
                                statistics=args.statistics)
    outputs = dict(energies_eV=result['energies'], offsets=result['offsets'])
    if result['forces'] is not None:
        outputs['forces_eV_A'] = result['forces']
    target = Path(args.output)
    target.parent.mkdir(parents=True, exist_ok=True)
    np.savez(target, **outputs)
    meta = result['metadata']
    print(json.dumps(dict(model=meta['model'], molecules=meta['molecules'], atoms=meta['atoms'],
                          route=meta['route'], seconds=round(meta['prediction_host_seconds'], 3))))


def _finetune(args):
    if args.list_parameters:
        from hyff.finetuning import parameter_groups
        from hyff.registry import resolve
        from hyff.weights import load_model
        model, _ = load_model(resolve(_model(args), args.weights, engine='reference').weights, 'cpu')
        for group, count, tensors in parameter_groups(model):
            print(f'{group + ".*":<40} {count:>14,} parameters  {tensors:>4} tensors')
        return 0
    missing = [flag for flag, value in (('--train', args.train), ('--output', args.output)) if value is None]
    if missing:
        raise ValueError(f'{" and ".join(missing)} required')
    from hyff.finetuning import finetune
    summary = finetune(_model(args), train=args.train, output=args.output, weights=args.weights, valid=args.valid,
                       valid_fraction=args.valid_fraction, epochs=args.epochs, lr=args.lr,
                       batch_atoms=args.batch_atoms, energy_weight=args.energy_weight,
                       force_weight=args.force_weight, freeze=args.freeze, train_only=args.train_only,
                       refit_element_energies=not args.no_refit_element_energies, ema_decay=args.ema_decay,
                       energy_key=args.energy_key, forces_key=args.forces_key, device=args.device,
                       gpus=args.gpus, seed=args.seed, resume=args.resume)
    print(json.dumps(dict(best=summary['best'], best_epoch=summary['best_epoch'], final=summary['final'])))
    return 0


def _main(argv=None):
    import logging
    args = _parser().parse_args(argv)
    logging.basicConfig(format='%(message)s')
    logging.getLogger('hyff').setLevel(logging.INFO)
    commands = dict(models=lambda a: _models(), download=_download, infer=_infer, md=_md, predict=_predict,
                    finetune=_finetune)
    try:
        code = commands[args.command](args)
    except KeyboardInterrupt:
        print('hyff: interrupted', file=sys.stderr)
        return 130
    except Exception as error:
        if args.debug:
            raise
        message = ' '.join(str(error).split()) or type(error).__name__
        if isinstance(error, FileNotFoundError) and error.filename and str(error.filename) not in message:
            message = f'{message}: {error.filename}'
        print(f'hyff {args.command}: error: {message} (run with --debug for the traceback)', file=sys.stderr)
        return 1
    return code or 0


def main(argv=None):
    # Under torchrun, record the first failing rank's exception for the elastic agent.
    import os
    if 'TORCHELASTIC_ERROR_FILE' in os.environ:
        from torch.distributed.elastic.multiprocessing.errors import record
        return record(_main)(argv)
    return _main(argv)


if __name__ == '__main__':
    sys.exit(main())
