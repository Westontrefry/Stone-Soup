"""Tests for the live feed client.

These run against a local stand-in for ``ais_live/server.py``, replaying
a fixture captured from the real feed. Nothing here touches the network,
so the suite keeps working after the ngrok tunnel is gone.

``sample_live_feed.json`` is genuine feed output, including the one
corrupt position the recorded dataset contains.
"""
import json
import threading
from datetime import datetime
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path

import numpy as np
import pytest

from .heatmap import density_grid
from .live import (SOLENT_BBOX, collect_live, fetch_records, fetch_status,
                   to_detections)

FIXTURE = Path(__file__).parent / 'sample_live_feed.json'
FEED_FIELDS = {'Time', 'MMSI', 'Latitude_degrees', 'Longitude_degrees',
               'COG_degrees', 'SOG_knots'}


def fixture_records():
    return json.loads(FIXTURE.read_text())['detections']


class FakeFeedHandler(BaseHTTPRequestHandler):
    """Stand-in for ais_live/server.py, same routes and payload shapes."""

    records = []

    def do_GET(self):
        from urllib.parse import parse_qs, urlparse

        request = urlparse(self.path)
        if request.path == '/api/status':
            self._send({'source': 'fixture', 'speed_factor': 30,
                        'buffered_events': len(self.records),
                        'latest_id': len(self.records)})
        elif request.path == '/api/ais':
            since_id = int(parse_qs(request.query).get('since_id', [0])[0])
            # The real server assigns each event a monotonic id; mirror
            # that by using one-based position in the fixture.
            new = self.records[since_id:]
            cursor = len(self.records) if new else since_id
            self._send({'cursor': cursor, 'detections': new})
        else:
            self._send({'error': 'not found'}, status=404)

    def _send(self, payload, status=200):
        body = json.dumps(payload).encode()
        self.send_response(status)
        self.send_header('Content-Type', 'application/json')
        self.send_header('Content-Length', str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, *args):
        """Keep the test output clean."""


@pytest.fixture
def feed():
    """Run a fake feed; yields its base URL."""
    FakeFeedHandler.records = fixture_records()
    server = HTTPServer(('localhost', 0), FakeFeedHandler)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    yield f'http://localhost:{server.server_address[1]}'
    server.shutdown()
    server.server_close()


# --------------------------------------------------------------------------
# Feed contract
# --------------------------------------------------------------------------

def test_fixture_matches_the_documented_schema():
    """Guards against ais_live/server.py's field names drifting."""
    for record in fixture_records():
        assert set(record) == FEED_FIELDS


def test_fetch_status(feed):
    status = fetch_status(feed)

    assert status['buffered_events'] == len(fixture_records())
    assert 'speed_factor' in status


def test_fetch_records_returns_records_and_cursor(feed):
    records, cursor = fetch_records(feed)

    assert len(records) == len(fixture_records())
    assert cursor == len(records)


def test_cursor_pagination_returns_only_new_records(feed):
    first, cursor = fetch_records(feed, since_id=0)
    again, next_cursor = fetch_records(feed, since_id=cursor)

    assert len(first) > 0
    assert again == []
    # With nothing new, the feed echoes the cursor back rather than resetting.
    assert next_cursor == cursor


def test_partial_cursor_skips_consumed_records(feed):
    all_records, _ = fetch_records(feed, since_id=0)
    tail, _ = fetch_records(feed, since_id=10)

    assert len(tail) == len(all_records) - 10
    assert tail[0] == all_records[10]


# --------------------------------------------------------------------------
# Conversion to Stone Soup detections
# --------------------------------------------------------------------------

def test_to_detections_builds_lon_lat_state_vectors():
    records = fixture_records()
    detections = to_detections(records)

    assert len(detections) == len(records)
    assert all(d.state_vector.shape == (2, 1) for d in detections)

    # Longitude first, matching heatmap.LON_LAT_MAPPING. Compared as
    # sorted pairs rather than positionally: the reader groups each
    # timestamp's detections into a set, so order within a timestamp is
    # arbitrary. Checked against the source records rather than by range,
    # since the fixture deliberately keeps the one corrupt position.
    assert sorted((float(d.state_vector[0]), float(d.state_vector[1]))
                  for d in detections) == sorted(
        (r['Longitude_degrees'], r['Latitude_degrees']) for r in records)

    clean = [d for d in detections if d.metadata['MMSI'] != 245188000]
    lons = [float(d.state_vector[0]) for d in clean]
    lats = [float(d.state_vector[1]) for d in clean]
    assert -2.0 < min(lons) and max(lons) < 0  # Solent is west of Greenwich
    assert 50.0 < min(lats) and max(lats) < 51.5


def test_to_detections_uses_double_precision():
    """Unlike the AISHub reader, which stores float32."""
    detections = to_detections(fixture_records())
    assert detections[0].state_vector.dtype == np.float64


def test_to_detections_parses_the_feed_timestamp_format():
    record = dict(fixture_records()[0])
    record['Time'] = '2026-07-28 23:49:09.193814'
    detection = to_detections([record])[0]

    assert detection.timestamp == datetime(2026, 7, 28, 23, 49, 9, 193814)


def test_to_detections_carries_identity_metadata():
    detection = to_detections(fixture_records())[0]

    assert 'MMSI' in detection.metadata
    assert 'SOG_knots' in detection.metadata
    assert 'COG_degrees' in detection.metadata
    # Position and time live on the Detection itself.
    for key in ('Time', 'Latitude_degrees', 'Longitude_degrees'):
        assert key not in detection.metadata


def test_to_detections_handles_an_empty_poll():
    """A poll with nothing new must give no detections, not a phantom one."""
    assert to_detections([]) == []


def test_collect_live_end_to_end(feed):
    detections, cursor = collect_live(feed)

    assert len(detections) == len(fixture_records())
    assert cursor == len(detections)
    assert all(d.state_vector.shape == (2, 1) for d in detections)


def test_collect_live_feeds_density_grid(feed):
    detections, _ = collect_live(feed)
    grid = density_grid(detections, n_bins=32, bbox=SOLENT_BBOX)

    assert len(grid['z']) == 32
    assert np.all(np.isfinite(np.array(grid['z'])))


# --------------------------------------------------------------------------
# The corrupt record
# --------------------------------------------------------------------------

def test_fixture_contains_the_known_corrupt_position():
    """MMSI 245188000 reports longitude +54.83 instead of about -1.1.

    It is line 8209 of docs/demos/SolentAIS_20160112_130211.csv, so it
    comes from the recorded dataset, not from the feed server.
    """
    corrupt = [r for r in fixture_records() if r['Longitude_degrees'] > 0]

    assert len(corrupt) == 1
    assert corrupt[0]['MMSI'] == 245188000
    assert corrupt[0]['Longitude_degrees'] == pytest.approx(54.83172)


def test_one_corrupt_position_wrecks_the_unclipped_grid():
    """Documents why bbox is not optional in practice.

    The grid spans its input's extent, so a single bad longitude widens
    it by two orders of magnitude and squeezes all real traffic into the
    first column of bins.
    """
    grid = density_grid(to_detections(fixture_records()), n_bins=32)
    span = grid['lon'][-1] - grid['lon'][0]

    assert span > 50  # degrees; the Solent is under half a degree wide


def test_bbox_excludes_the_corrupt_position():
    records = fixture_records()
    grid = density_grid(to_detections(records), n_bins=32, bbox=SOLENT_BBOX)

    assert grid['n_points'] == len(records) - 1
    assert grid['lon'][-1] - grid['lon'][0] < 1.0
    assert -2.0 <= grid['lon'][0] and grid['lon'][-1] <= -0.5
    assert 50.4 <= grid['lat'][0] and grid['lat'][-1] <= 51.0


def test_bbox_keeps_the_density_peak_over_real_traffic():
    grid = density_grid(to_detections(fixture_records()), n_bins=64,
                        bbox=SOLENT_BBOX)
    z = np.array(grid['z'])
    lon_index, lat_index = np.unravel_index(np.argmax(z), z.shape)

    # Southampton Water / Portsmouth approaches.
    assert -1.25 < grid['lon'][lon_index] < -1.0
    assert 50.75 < grid['lat'][lat_index] < 50.85
