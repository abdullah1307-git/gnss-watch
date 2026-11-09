# Detection methodology

This document explains what GNSS Watch measures, why, and where it can be wrong. All
thresholds live in `src/gnsswatch/config.py` and are written into every `summary.json`, so
any result can be reproduced.

## 1. Input data

Aircraft equipped with ADS-B broadcast their GNSS position about twice per second, together
with quality indicators defined in the ADS-B standard (DO-260B):

| Field | Meaning | Typical healthy airliner |
|---|---|---|
| NIC (Navigation Integrity Category) | Radius the position is guaranteed to be within | 8 (under 0.1 NM) |
| NACp (Navigation Accuracy Category, position) | 95% accuracy bound of the position | 9 to 10 (under 30 m) |

Volunteer receivers collect these messages. [adsb.lol](https://adsb.lol) aggregates them and
publishes one archive per day in the readsb `globe_history` format: for each aircraft a
`trace_full_<icao>.json` with positions, altitudes, speed and flags. NIC and NACp only appear
in occasional "details" objects, so the parser carries the last known value forward.

Only positions whose source is the aircraft's own ADS-B (`adsb_icao`, `adsr_icao`, ...) are
used. MLAT and TIS-B positions are computed on the ground and say nothing about the aircraft's
receiver.

## 2. Jamming indicator

Jamming drowns the satellite signal in noise. The receiver knows it has a problem and reports
low NIC/NACp, or loses its fix. One bad report proves nothing; many aircraft degrading in the
same place at the same time does.

1. A report is **degraded** if NIC < 7 or NACp < 8 (or, in live data, readsb marks the
   aircraft with `gpsOkBefore`, meaning it recently lost GNSS).
2. Reports are grouped by **hour** and **cell** (H3 resolution 4, cells about 22 km across).
3. An aircraft counts as degraded in a cell-hour if at least half of its reports there are.
4. **Faulty receiver filter (peer check).** For each degraded aircraft-cell-hour with at least
   two other aircraft present, we check whether most of those peers are healthy. If an
   aircraft is "contradicted" by healthy peers at least twice, and in at least half of its
   comparable cell-hours, the cause is its own equipment. Such aircraft are excluded from the
   statistics and listed in `summary.json`. Without this step, two aircraft with old avionics
   crossing paths in a quiet area produce a false "high" cell.
5. Ratio = degraded aircraft / aircraft. Levels:
   - `insufficient`: fewer than 3 aircraft
   - `high`: ratio of 10% or more **and** at least 2 degraded aircraft
   - `medium`: ratio of 2% or more **and** at least 2 degraded aircraft
   - `low`: otherwise

The whole-period map sums aircraft-hours per cell. Because a short event gets diluted over a
day, it also reports the peak hourly ratio and the number of hours at `high`, and any cell
that was `high` in some hour is shown as at least `medium`.

## 3. Spoofing indicators

Spoofing transmits a fake but healthy-looking signal. NIC/NACp often stay good, so integrity
alone cannot detect it. Physics can.

### Position jumps
For consecutive reports of the same aircraft (same leg, at most 15 minutes apart):

```
implied_speed = great_circle_distance / time_difference
jump if distance >= 3 NM
    and implied_speed > max(750 kt, 1.5 x reported_ground_speed + 100 kt)
```

The reported ground speed is used as a reference because spoofed positions usually come
with plausible velocities. Coverage gaps are handled naturally: an aircraft that disappears
for 10 minutes and reappears 75 NM further on has an implied speed of 450 kt, which is fine.

### GNSS vs barometric altitude
Aircraft report a barometric altitude (from air pressure) and a geometric altitude (from
GNSS). Their difference depends on temperature and pressure, so it changes slowly. A change of
more than 1,000 ft within 60 seconds means the GNSS altitude has suddenly gone wrong.

### Hotspots
A single jump can be a decoding glitch. A spoofer, however, broadcasts **one** fake location
to every aircraft in range. Jump destinations are clustered greedily in time order: a jump joins
the nearest cluster whose centre is within 15 NM and whose latest jump was less than an hour
earlier. Clusters with at least 3 distinct aircraft are reported as hotspots. Distance-based
clustering avoids splitting a hotspot that sits on a grid-cell border.

## 4. Known limitations

- **Coverage.** Volunteer receivers are mostly on land. Over open sea, and at low altitude,
  data is sparse and many cells are `insufficient`.
- **Upstream filtering.** readsb rejects positions that fail its own speed check before they
  reach the archive. Some spoofing therefore shows up as a gap followed by a jump, and very
  short episodes may be missed completely.
- **Equipment.** Some older transponders report NIC/NACp of 0 permanently. The peer check
  removes most of them, but only where peers exist.
- **No ground truth.** There is no public, complete list of interference events. Phase 4 will
  compare results with published incident reports.
- **Not certified.** These are indicators for research and situational awareness only.

## 5. Tuning

Every threshold is a field in `JammingConfig` or `SpoofingConfig`. Useful ones to explore:

| Setting | Default | Effect |
|---|---|---|
| `grid_resolution` | 4 | Higher means smaller cells and more detail, but fewer aircraft per cell |
| `jamming.window_s` | 3600 | Shorter windows react faster but are noisier |
| `jamming.high_ratio` | 0.10 | Ratio needed for `high` |
| `spoofing.min_jump_nm` | 3.0 | Raise if you see decoder glitches |
| `spoofing.hotspot_radius_nm` | 15.0 | Size of a spoofing cluster |
