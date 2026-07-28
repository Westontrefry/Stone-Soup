"""Tests for the /density endpoint.

Each test starts a real server on an ephemeral port and makes a real
request, so the routing, status codes and CORS headers the front end
depends on are exercised end to end.
"""
import json
import threading
from http.server import HTTPServer
from pathlib import Path
from urllib.error import HTTPError
from urllib.request import urlopen

import pytest

from .serve import DensityHandler
from .test_heatmap import write_ais_json

SAMPLE = Path(__file__).parent / 'sample_ais.json'


class QuietHandler(DensityHandler):
    def log_message(self, *args):
        """Keep the test output clean."""


@pytest.fixture
def serve():
    """Start a server against a given source file; yields its base URL."""
    servers = []

    def start(source=SAMPLE):
        QuietHandler.source = source
        server = HTTPServer(('localhost', 0), QuietHandler)
        servers.append(server)
        threading.Thread(target=server.serve_forever, daemon=True).start()
        return f'http://localhost:{server.server_address[1]}'

    yield start

    for server in servers:
        server.shutdown()
        server.server_close()


def get(url):
    with urlopen(url) as response:
        return response.status, response.headers, json.loads(response.read())


def test_density_endpoint_returns_a_grid(serve):
    status, _, payload = get(f'{serve()}/density?n_bins=16')

    assert status == 200
    assert len(payload['lon']) == 16
    assert len(payload['lat']) == 16
    assert len(payload['z']) == 16
    assert payload['n_points'] > 0


def test_density_defaults_to_128_bins(serve):
    _, _, payload = get(f'{serve()}/density')
    assert len(payload['lon']) == 128


def test_response_allows_cross_origin_requests(serve):
    """The front end will run on a different origin during development."""
    _, headers, _ = get(f'{serve()}/density?n_bins=8')
    assert headers['Access-Control-Allow-Origin'] == '*'


def test_response_is_declared_as_json(serve):
    _, headers, _ = get(f'{serve()}/density?n_bins=8')
    assert headers['Content-Type'] == 'application/json'


def test_unknown_path_returns_404(serve):
    with pytest.raises(HTTPError) as error:
        urlopen(f'{serve()}/heatmap')
    assert error.value.code == 404


def test_non_integer_n_bins_returns_400(serve):
    with pytest.raises(HTTPError) as error:
        urlopen(f'{serve()}/density?n_bins=lots')
    assert error.value.code == 400


def test_degenerate_source_returns_422_not_500(serve, tmp_path):
    """A bad feed must surface as a clean error, not a stack trace."""
    source = write_ais_json(tmp_path / 'flat.json', [(1.4, 51.2)] * 5)

    with pytest.raises(HTTPError) as error:
        urlopen(f'{serve(source)}/density')

    assert error.value.code == 422
    assert 'no spread' in json.loads(error.value.read())['error']
