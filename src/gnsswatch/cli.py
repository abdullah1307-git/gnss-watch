"""Command line interface.

    gnsswatch download --date 2026-11-08
    gnsswatch analyze data/raw/2026-11-08 --region baltic --out reports/2026-11-08
    gnsswatch live --region baltic --out reports/live
    gnsswatch demo --out reports/demo
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from dataclasses import replace
from datetime import date, datetime, timedelta, timezone
from pathlib import Path

from . import __version__
from .config import REGIONS, BBox, Settings

log = logging.getLogger("gnsswatch")


def _settings(args: argparse.Namespace) -> Settings:
    s = Settings(region=BBox.parse(args.region), grid_resolution=args.resolution,
                 grid_backend=args.grid)
    if getattr(args, "window", None):
        s = replace(s, jamming=replace(s.jamming, window_s=args.window))
    return s


def _print_summary(written: dict, summary: dict) -> None:
    keys = ["aircraft", "position_reports", "cells_by_level", "aircraft_faulty_receivers",
            "position_jumps", "aircraft_with_jumps", "altitude_events", "spoofing_hotspots"]
    print(json.dumps({k: summary.get(k) for k in keys}, indent=2))
    print(f"\nReport: {written['report'].resolve()}")


def cmd_download(args: argparse.Namespace) -> int:
    from .sources.adsblol_history import download_day

    day = date.fromisoformat(args.date)
    out = Path(args.out) / day.isoformat()
    paths = download_day(day, out, pod=args.pod)
    total = sum(p.stat().st_size for p in paths) / (1 << 30)
    print(f"Downloaded {len(paths)} part(s), {total:.2f} GiB, to {out}")
    print(f"Next: gnsswatch analyze {out} --region baltic --out reports/{day.isoformat()}")
    return 0


def cmd_analyze(args: argparse.Namespace) -> int:
    from .outputs import write_outputs
    from .pipeline import analyze
    from .sources.adsblol_history import load_history

    settings = _settings(args)
    inputs = [Path(p) for p in args.input]
    for p in inputs:
        if not p.exists():
            print(f"error: {p} does not exist", file=sys.stderr)
            return 2
    source = inputs[0] if len(inputs) == 1 and inputs[0].is_dir() else inputs
    df = load_history(source, settings.region, max_traces=args.max_traces)
    if df.empty:
        print("No position reports inside the region. Check --region and the input files.",
              file=sys.stderr)
        return 1
    result = analyze(df, settings)
    title = args.title or f"GNSS interference: {args.region} ({inputs[0].name})"
    written = write_outputs(result, Path(args.out), title=title)
    _print_summary(written, result.summary())
    return 0


def cmd_live(args: argparse.Namespace) -> int:
    from .outputs import write_outputs
    from .pipeline import analyze
    from .sources.adsblol_live import fetch_region

    settings = _settings(args)
    df = fetch_region(settings.region)
    if df.empty:
        print("No aircraft returned by the API for this region.", file=sys.stderr)
        return 1
    result = analyze(df, settings)
    stamp = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    written = write_outputs(result, Path(args.out), title=f"GNSS live snapshot: {args.region}, {stamp}")
    _print_summary(written, result.summary())
    return 0


def cmd_demo(args: argparse.Namespace) -> int:
    from .outputs import write_outputs
    from .parsing import frame_from_rows, parse_trace
    from .pipeline import analyze
    from .synthetic import Scenario, synthetic_traces

    settings = _settings(args)
    docs, truth = synthetic_traces(Scenario(bbox=settings.region, seed=args.seed))
    df = frame_from_rows(r for d in docs for r in parse_trace(d, settings.region))
    result = analyze(df, settings)
    written = write_outputs(result, Path(args.out), title="GNSS Watch demo (synthetic data)")
    _print_summary(written, result.summary())
    found = set(result.hotspots["aircraft"].str.split(",").explode()) if len(result.hotspots) else set()
    print(f"\nGround truth: {len(truth.spoofed_aircraft)} spoofed aircraft injected, "
          f"{len(found & truth.spoofed_aircraft)} found in hotspots; "
          f"{len(truth.faulty_aircraft)} faulty receivers injected, "
          f"{len(set(result.faulty_aircraft) & truth.faulty_aircraft)} identified.")
    return 0


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="gnsswatch",
        description="Detect GNSS jamming and spoofing affecting aircraft from open ADS-B data.",
    )
    p.add_argument("--version", action="version", version=f"%(prog)s {__version__}")
    p.add_argument("-v", "--verbose", action="store_true", help="debug logging")
    sub = p.add_subparsers(dest="command", required=True)

    def common(sp: argparse.ArgumentParser, default_out: str) -> None:
        sp.add_argument("--region", default="baltic",
                        help=f"named region ({', '.join(REGIONS)}) or 'min_lat,min_lon,max_lat,max_lon'")
        sp.add_argument("--resolution", type=int, default=4, help="H3 resolution (default 4)")
        sp.add_argument("--grid", choices=["auto", "h3", "square"], default="auto")
        sp.add_argument("--out", default=default_out, help="output directory")

    d = sub.add_parser("download", help="download one day of adsb.lol history")
    yesterday = (datetime.now(timezone.utc).date() - timedelta(days=2)).isoformat()
    d.add_argument("--date", default=yesterday, help="YYYY-MM-DD (default: two days ago)")
    d.add_argument("--pod", default="prod-0", help="adsb.lol pod (prod-0, prod-1, staging-0)")
    d.add_argument("--out", default="data/raw")
    d.set_defaults(func=cmd_download)

    a = sub.add_parser("analyze", help="analyse downloaded history and write a report")
    a.add_argument("input", nargs="+", help="day directory, tar part files, or extracted traces dir")
    common(a, "reports/latest")
    a.add_argument("--window", type=int, default=3600, help="jamming time window in seconds")
    a.add_argument("--max-traces", type=int, default=None, help="stop after N aircraft (quick test)")
    a.add_argument("--title", default=None)
    a.set_defaults(func=cmd_analyze)

    lv = sub.add_parser("live", help="analyse a live snapshot from the adsb.lol API")
    common(lv, "reports/live")
    lv.set_defaults(func=cmd_live)

    dm = sub.add_parser("demo", help="run the full pipeline on synthetic data")
    common(dm, "reports/demo")
    dm.add_argument("--seed", type=int, default=7)
    dm.set_defaults(func=cmd_demo)
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
