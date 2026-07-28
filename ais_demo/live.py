"""Client for the live AIS feed served by ``ais_live/server.py``.

The feed replays real Solent AIS data over HTTP. Its JSON uses the same
field names as the ``AIS_Solent_Tracker`` demo's CSV, which is what lets
:class:`~.DictionaryDetectionReader` consume it directly -- the same
approach ``ais_live/client_example.py`` takes, so both consumers parse
the feed identically.

Shape and style of the feed, confirmed against a running instance:

* ``GET /api/ais?since_id=N`` returns
  ``{"cursor": int, "detections": [...]}``.
* Positions are plain decimal degrees. There is no scaling, unlike the
  AISHub format that :class:`~.JSON_AISDetectionReader` handles.
* Longitudes in the Solent are negative (around -1.1).
* ``Time`` is a naive local-style string, ``"%Y-%m-%d %H:%M:%S.%f"``.
* It is a **polling** feed, not a stream: pass the previous ``cursor``
  back as ``since_id`` to get only what is new. ``since_id=0`` returns
  the whole buffer, which the server caps at 20,000 events.
"""
import json
from urllib.parse import urlencode, urljoin
from urllib.request import Request, urlopen

from stonesoup.reader.generic import DictionaryDetectionReader

#: Default feed location. Override per call; ngrok URLs are short-lived.
FEED_URL = 'https://bc55-2603-6010-9200-c00-1819-5480-f8d5-29d8.ngrok-free.app'

#: Field names, matching ais_live/server.py's JSON schema.
STATE_VECTOR_FIELDS = ('Longitude_degrees', 'Latitude_degrees')
TIME_FIELD = 'Time'
#: Given explicitly so the reader uses ``strptime`` rather than falling
#: back to ``dateutil.parse`` on every one of 20,000 records.
TIME_FIELD_FORMAT = '%Y-%m-%d %H:%M:%S.%f'

#: Generous bounding box around the Solent.
#:
#: The recorded dataset contains one corrupt position -- MMSI 245188000
#: at longitude +54.83 instead of about -1.1, line 8209 of
#: docs/demos/SolentAIS_20160112_130211.csv. It is a single record in
#: 20,000, but the density grid spans the extent of its input, so that
#: one point stretches the map across 56 degrees of longitude instead of
#: 0.4 and collapses the real traffic into a single column of bins.
#: Passing this as ``bbox`` to :func:`~.density_grid` excludes it.
SOLENT_BBOX = (-2.0, 50.4, -0.5, 51.0)

_HEADERS = {
    'Accept': 'application/json',
    # Free ngrok tunnels can serve an HTML interstitial to non-browser
    # clients. This endpoint currently returns JSON without it, but the
    # header costs nothing and stops a tunnel reconfiguration from
    # breaking us with an HTML parse error.
    'ngrok-skip-browser-warning': 'true',
}


def _get_json(base_url, path, params=None, timeout=30):
    url = urljoin(base_url.rstrip('/') + '/', path)
    if params:
        url = f'{url}?{urlencode(params)}'
    with urlopen(Request(url, headers=_HEADERS), timeout=timeout) as response:
        return json.loads(response.read())


def fetch_status(base_url=FEED_URL, timeout=30):
    """Return the feed's status document.

    Returns
    -------
    : dict
        ``source``, ``speed_factor``, ``buffered_events``, ``latest_id``.
    """
    return _get_json(base_url, 'api/status', timeout=timeout)


def fetch_records(base_url=FEED_URL, since_id=0, timeout=30):
    """Fetch raw feed records.

    Parameters
    ----------
    since_id : int
        Cursor from a previous call. ``0`` returns the whole buffer.

    Returns
    -------
    : list of dict
        Raw records, in time order.
    : int
        Cursor to pass as ``since_id`` next time. When nothing new has
        arrived the feed echoes back the ``since_id`` it was given.
    """
    payload = _get_json(base_url, 'api/ais', {'since_id': since_id},
                        timeout=timeout)
    return payload['detections'], payload['cursor']


def to_detections(records):
    """Convert raw feed records into Stone Soup detections.

    Uses :class:`~.DictionaryDetectionReader`, so the state vector is
    ``[[lon], [lat]]`` -- matching
    :data:`~ais_demo.heatmap.LON_LAT_MAPPING` -- and MMSI, SOG and COG
    are carried through as detection metadata.
    """
    if not records:
        # The reader yields a single (None, set()) pair when handed an
        # empty sequence, which would otherwise surface as a phantom
        # detection-less timestep.
        return []

    reader = DictionaryDetectionReader(
        dictionaries=iter(records),
        state_vector_fields=STATE_VECTOR_FIELDS,
        time_field=TIME_FIELD,
        time_field_format=TIME_FIELD_FORMAT)
    return [detection for _, detections in reader for detection in detections]


def collect_live(base_url=FEED_URL, since_id=0, timeout=30):
    """Fetch the feed and return detections. Live counterpart of ``collect``.

    Returns
    -------
    : list of :class:`~.Detection`
        Detections from the feed, in time order.
    : int
        Cursor for the next poll.
    """
    records, cursor = fetch_records(base_url, since_id=since_id,
                                    timeout=timeout)
    return to_detections(records), cursor
