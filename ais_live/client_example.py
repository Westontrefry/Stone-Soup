"""Example Stone Soup consumer for the live AIS feed served by server.py.

This mirrors docs/demos/AIS_Solent_Tracker.py, but sources detections from
the live HTTP feed (e.g. shared via ngrok) instead of a static CSV file, and
periodically writes an updated map to live_map.html so it can be viewed in a
browser while the tracker keeps running.

Usage
-----
    Set FEED_URL below to your ngrok URL + "/api/ais", then:
        python client_example.py
"""
import datetime
import time
from collections import defaultdict
from itertools import cycle

import folium
import numpy as np
import requests
import utm

FEED_URL = "https://bc55-2603-6010-9200-c00-1819-5480-f8d5-29d8.ngrok-free.app/api/ais"
POLL_INTERVAL = 2  # seconds between polls when no new data is available


def ais_stream(url=FEED_URL, poll_interval=POLL_INTERVAL):
    """Generator yielding AIS detection dicts, in time order, as they arrive."""
    since_id = 0
    while True:
        resp = requests.get(url, params={"since_id": since_id}, timeout=10)
        resp.raise_for_status()
        payload = resp.json()
        since_id = payload["cursor"]
        yield from payload["detections"]
        if not payload["detections"]:
            time.sleep(poll_interval)


# %% Detector: live feed instead of CSVDetectionReader, rest matches the demo
from stonesoup.reader.generic import DictionaryDetectionReader
detector = DictionaryDetectionReader(
    dictionaries=ais_stream(),
    state_vector_fields=("Longitude_degrees", "Latitude_degrees"),
    time_field="Time")

from stonesoup.feeder.filter import MetadataReducer
detector = MetadataReducer(detector, 'MMSI')

from stonesoup.feeder.geo import LongLatToUTMConverter
detector = LongLatToUTMConverter(detector)

# %% Models, predictor/updater, associator, initiator/deleter - same as the demo
from stonesoup.models.transition.linear import (
    CombinedLinearGaussianTransitionModel, OrnsteinUhlenbeck)
transition_model = CombinedLinearGaussianTransitionModel(
    (OrnsteinUhlenbeck(0.5, 1e-4), OrnsteinUhlenbeck(0.5, 1e-4)))

from stonesoup.models.measurement.linear import LinearGaussian
measurement_model = LinearGaussian(
    ndim_state=4, mapping=[0, 2], noise_covar=np.diag([15, 15]))

from stonesoup.predictor.kalman import KalmanPredictor
predictor = KalmanPredictor(transition_model)

from stonesoup.updater.kalman import KalmanUpdater
updater = KalmanUpdater(measurement_model)

from stonesoup.gater.filtered import FilteredDetectionsGater
from stonesoup.hypothesiser.distance import DistanceHypothesiser
from stonesoup.measures import Mahalanobis
hypothesiser = FilteredDetectionsGater(
    DistanceHypothesiser(predictor, updater, Mahalanobis(), missed_distance=3),
    metadata_filter="MMSI")

from stonesoup.dataassociator.neighbour import NearestNeighbour
data_associator = NearestNeighbour(hypothesiser)

from stonesoup.types.state import GaussianState
from stonesoup.initiator.simple import SimpleMeasurementInitiator
initiator = SimpleMeasurementInitiator(
    GaussianState([[0], [0], [0], [0]], np.diag([0, 10, 0, 10])),
    measurement_model)

from stonesoup.deleter.time import UpdateTimeDeleter
deleter = UpdateTimeDeleter(time_since_update=datetime.timedelta(minutes=10))

from stonesoup.tracker.simple import MultiTargetTracker
tracker = MultiTargetTracker(
    initiator=initiator,
    deleter=deleter,
    detector=detector,
    data_associator=data_associator,
    updater=updater,
)


def save_map(tracks, path="live_map.html"):
    colour_iter = iter(cycle(
        ['red', 'blue', 'green', 'purple', 'orange', 'darkred',
         'lightred', 'beige', 'darkblue', 'darkgreen', 'cadetblue',
         'darkpurple', 'pink', 'lightblue', 'lightgreen',
         'gray', 'black', 'lightgray']))
    colour = defaultdict(lambda: next(colour_iter))

    m = folium.Map(location=[50.75, -1], zoom_start=10)
    for track in tracks:
        points = [
            utm.to_latlon(
                *state.state_vector[measurement_model.mapping, :],
                detector.zone_number, northern=detector.northern, strict=False)
            for state in track]
        folium.PolyLine(points, color=colour[track.metadata.get('MMSI')]).add_to(m)
        folium.Marker(
            points[-1],
            icon=folium.Icon(
                icon='fa-ship', prefix="fa", color=colour[track.metadata.get('MMSI')]),
            popup="\n".join(f"{key}: {value}" for key, value in track.metadata.items())
        ).add_to(m)
    m.save(path)


if __name__ == "__main__":
    tracks = set()
    for step, (current_time, current_tracks) in enumerate(tracker, 1):
        tracks.update(current_tracks)
        if not step % 5:
            print(f"Step: {step}  Time: {current_time}  Tracks: {len(tracks)}")
            save_map(tracks)
