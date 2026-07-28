"""Minimal AIS -> density heatmap pipeline.

Reads AIS detections via Stone Soup and turns them into a grid a front
end can render as a heatmap.

From a local AISHub-format file::

    grid = density_grid(collect('sample_ais.json'))

From the live feed served by ``ais_live/server.py``::

    detections, cursor = collect_live()
    grid = density_grid(detections, bbox=SOLENT_BBOX)
"""
from .heatmap import LON_LAT_MAPPING, collect, density_grid
from .live import FEED_URL, SOLENT_BBOX, collect_live, fetch_status

__all__ = ['collect', 'collect_live', 'density_grid', 'fetch_status',
           'FEED_URL', 'LON_LAT_MAPPING', 'SOLENT_BBOX']
