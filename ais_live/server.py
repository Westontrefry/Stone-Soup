"""Live AIS data server for sharing via ngrok.

Replays the real AIS dataset bundled with Stone Soup
(docs/demos/SolentAIS_20160112_130211.csv, recorded in the Solent) as a
continuous, time-ordered "live" feed over HTTP. The JSON schema matches the
column names used by ``stonesoup.reader.generic.CSVDetectionReader`` in the
``AIS_Solent_Tracker`` demo, so it can be consumed directly with a Stone Soup
``DictionaryDetectionReader`` (see client_example.py).

Usage
-----
    python server.py
    ngrok http 5001

Then share the ngrok https URL (append ``/api/ais``) with your friend.
"""
import csv
import itertools
import threading
import time
from collections import deque
from datetime import datetime, timedelta
from pathlib import Path

from flask import Flask, jsonify, request

CSV_PATH = (Path(__file__).resolve().parent.parent
            / "docs" / "demos" / "SolentAIS_20160112_130211.csv")
SPEED_FACTOR = 30       # replay this many times faster than the original recording
MAX_STEP_SECONDS = 5     # cap the sleep between rows so any large gaps don't stall the feed
HISTORY_SIZE = 20000     # roughly one full loop of the dataset

app = Flask(__name__)

_events = deque(maxlen=HISTORY_SIZE)
_next_id = itertools.count(1)
_lock = threading.Lock()


def _load_rows():
    with CSV_PATH.open(newline="") as f:
        rows = [
            {
                "Time": datetime.strptime(row["Time"], "%Y-%m-%d %H:%M:%S.%f"),
                "MMSI": int(row["MMSI"]),
                "Latitude_degrees": float(row["Latitude_degrees"]),
                "Longitude_degrees": float(row["Longitude_degrees"]),
                "COG_degrees": float(row["COG_degrees"]),
                "SOG_knots": float(row["SOG_knots"]),
            }
            for row in csv.DictReader(f)
        ]
    rows.sort(key=lambda r: r["Time"])
    return rows


def _replay_loop():
    """Continuously emits rows in time order, onto a virtual clock that always
    moves forward (even across loop wrap-around), so consumers can rely on
    detections arriving in non-decreasing time order."""
    rows = _load_rows()
    virtual_time = datetime.utcnow()
    prev_original_time = None

    for row in itertools.cycle(rows):
        if prev_original_time is None:
            dt = timedelta(0)
        else:
            dt = row["Time"] - prev_original_time
            if dt.total_seconds() < 0:
                dt = timedelta(seconds=1)  # loop wrap-around
        prev_original_time = row["Time"]

        virtual_time += dt
        sleep_for = min(dt.total_seconds() / SPEED_FACTOR, MAX_STEP_SECONDS)
        if sleep_for > 0:
            time.sleep(sleep_for)

        event = dict(row)
        event["Time"] = virtual_time.strftime("%Y-%m-%d %H:%M:%S.%f")

        with _lock:
            _events.append((next(_next_id), event))


@app.route("/")
def index():
    return (
        "<h2>Stone Soup live AIS feed</h2>"
        "<p>Real AIS data recorded in the Solent, replayed as a live feed "
        f"({SPEED_FACTOR}x speed).</p>"
        "<p>Poll <code>GET /api/ais?since_id=0</code> for new detections (JSON), "
        "<code>GET /api/heatmap</code> for density/heatmap data ([lat, lon, weight]), "
        "or check <code>GET /api/status</code>.</p>"
        "<p>See <code>client_example.py</code> for a ready-to-run Stone Soup consumer.</p>"
    )


@app.route("/api/status")
def status():
    with _lock:
        count = len(_events)
        latest_id = _events[-1][0] if _events else 0
    return jsonify(
        source="SolentAIS_20160112_130211.csv (replayed)",
        speed_factor=SPEED_FACTOR,
        buffered_events=count,
        latest_id=latest_id,
    )


@app.route("/api/ais")
def ais():
    since_id = request.args.get("since_id", default=0, type=int)
    with _lock:
        new_events = [(i, e) for i, e in _events if i > since_id]
    cursor = new_events[-1][0] if new_events else since_id
    return jsonify(cursor=cursor, detections=[e for _, e in new_events])


@app.route("/api/heatmap")
def heatmap():
    """Point density data for a heatmap layer.

    Returns a JSON array of ``[lat, lon, weight]`` triples covering all
    currently buffered positions - the format expected directly by
    Leaflet.heat (``L.heatLayer(data)``) and similar heatmap libraries.
    """
    with _lock:
        points = [[e["Latitude_degrees"], e["Longitude_degrees"], 1] for _, e in _events]
    return jsonify(points)


if __name__ == "__main__":
    threading.Thread(target=_replay_loop, daemon=True).start()
    app.run(host="0.0.0.0", port=5001)
