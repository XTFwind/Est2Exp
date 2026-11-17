import argparse
from pathlib import Path

from .config import load_config, load_scenario, resolve_topology


def main(argv=None):
    parser = argparse.ArgumentParser(prog="est2exp", description="Simulate an anonymous model-level training DAG.")
    actions = parser.add_subparsers(dest="command", required=True)
    for name, help_text in (("topology", "Build a hardware topology JSON"), ("run", "Simulate and export the full DAG"),
                            ("evolve", "Evolve strategy scores using a configured model service")):
        action = actions.add_parser(name, help=help_text)
        action.add_argument("config", type=Path)
        action.add_argument("--output", required=True, type=Path,
                            help="JSON file" if name == "topology" else "Output directory")
        if name == "evolve":
            action.add_argument("--rounds", type=int, default=2)
    args = parser.parse_args(argv)
    if args.command == "topology":
        import json
        hardware = resolve_topology(load_config(args.config), args.config)
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(hardware, indent=2, allow_nan=False) + "\n", encoding="utf-8")
        print(f"{hardware['summary']['total_devices']} devices · {len(hardware['links'])} links · {args.output}")
    elif args.command == "run":
        from .pipeline import run_scenario
        dag, report = run_scenario(load_scenario(args.config), args.output)
        print(f"{len(dag['nodes'])} events · {len(dag['edges'])} dependencies · {report['latency_s']:.9g} s")
        print(args.output / "dag.html")
    else:
        from .evolve import run_evolution
        summary = run_evolution(load_scenario(args.config), args.output, rounds=args.rounds)
        print(f"Best evaluated strategy: {summary['best']['latency_s']:.9g} s")
        print(args.output / "best" / "dag.html")


if __name__ == "__main__":
    main()
