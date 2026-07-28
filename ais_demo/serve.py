"""One endpoint serving the density grid, on the standard library only.

Against the bundled fixture::

    python -m ais_demo.serve

Against the live feed from ``ais_live/server.py``::

    python -m ais_demo.serve --live
    python -m ais_demo.serve --live https://your-tunnel.ngrok-free.app

Then::

    curl 'http://localhost:8000/density?n_bins=64'
"""
import argparse
import json
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path
from urllib.error import URLError
from urllib.parse import parse_qs, urlparse

from .heatmap import collect, density_grid
from .live import FEED_URL, SOLENT_BBOX, collect_live

DEFAULT_SOURCE = Path(__file__).parent / 'sample_ais.json'


def file_source(path):
    """Detections from a local AISHub-format JSON file."""
    return lambda: collect(path)


def live_source(base_url):
    """Detections from the live feed. Snapshots the whole buffer each call.

    Polling from ``since_id=0`` every time keeps the server stateless:
    each request returns the density of everything currently buffered,
    which is what a heatmap wants. Incremental polling via the cursor
    matters for a tracker, not for this.
    """
    return lambda: collect_live(base_url)[0]


class DensityHandler(BaseHTTPRequestHandler):
    """Serves ``GET /density`` and nothing else."""

    source = DEFAULT_SOURCE
    load_detections = None
    bbox = None

    def _detections(self):
        if self.load_detections is not None:
            return self.load_detections()
        return collect(self.source)

    def do_GET(self):
        request = urlparse(self.path)
        if request.path != '/density':
            self._send_json(404, {'error': 'Not found. Try /density'})
            return

        query = parse_qs(request.query)
        try:
            n_bins = int(query.get('n_bins', [128])[0])
        except ValueError:
            self._send_json(400, {'error': 'n_bins must be an integer'})
            return

        try:
            bbox = self._parse_bbox(query)
        except ValueError as error:
            self._send_json(400, {'error': str(error)})
            return

        try:
            detections = self._detections()
        except (URLError, OSError) as error:
            # The feed is someone else's tunnel; it can and will vanish.
            self._send_json(502, {'error': f'Could not reach the AIS feed: {error}'})
            return

        try:
            grid = density_grid(detections, n_bins=n_bins, bbox=bbox)
        except ValueError as error:
            self._send_json(422, {'error': str(error)})
            return

        self._send_json(200, grid)

    def _parse_bbox(self, query):
        if 'bbox' not in query:
            return self.bbox
        parts = query['bbox'][0].split(',')
        if len(parts) != 4:
            raise ValueError('bbox must be min_lon,min_lat,max_lon,max_lat')
        try:
            return tuple(float(part) for part in parts)
        except ValueError:
            raise ValueError('bbox values must be numbers') from None

    def do_OPTIONS(self):
        self.send_response(204)
        self._send_cors_headers()
        self.end_headers()

    def _send_json(self, status, payload):
        body = json.dumps(payload).encode()
        self.send_response(status)
        self.send_header('Content-Type', 'application/json')
        self.send_header('Content-Length', str(len(body)))
        self._send_cors_headers()
        self.end_headers()
        self.wfile.write(body)

    def _send_cors_headers(self):
        # The front end will be served from a different origin during
        # development, so it needs these to read the response.
        self.send_header('Access-Control-Allow-Origin', '*')
        self.send_header('Access-Control-Allow-Headers', 'Content-Type')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path, default=DEFAULT_SOURCE,
                        help='AIS JSON file to read (default: sample_ais.json)')
    parser.add_argument('--live', nargs='?', const=FEED_URL, default=None,
                        metavar='URL',
                        help=f'poll the live feed instead of a file '
                             f'(default URL: {FEED_URL})')
    parser.add_argument('--bbox', default=None,
                        help='default clip box "min_lon,min_lat,max_lon,max_lat"; '
                             'applied when the request does not supply one')
    parser.add_argument('--port', type=int, default=8000)
    args = parser.parse_args()

    if args.live:
        DensityHandler.load_detections = staticmethod(live_source(args.live))
        # Without a clip box the feed's one corrupt position blows out
        # the grid extent, so default to the Solent unless told otherwise.
        default_bbox = SOLENT_BBOX
        described = f'live feed {args.live}'
    else:
        DensityHandler.source = args.source
        default_bbox = None
        described = str(args.source)

    if args.bbox:
        default_bbox = tuple(float(part) for part in args.bbox.split(','))
    DensityHandler.bbox = default_bbox

    server = HTTPServer(('localhost', args.port), DensityHandler)
    print(f"Serving {described} at http://localhost:{args.port}/density")
    if default_bbox:
        print(f"Clipping to bbox {default_bbox}")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nStopped.")


if __name__ == '__main__':
    main()
