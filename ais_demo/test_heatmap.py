"""Tests for the AIS -> density grid pipeline.

Most tests build their own AIS data with :func:`write_ais_json` so the
expected answer is known exactly, rather than relying on the committed
sample file. That lets us assert on recovered structure (where the
density actually peaks) instead of just on shapes.
"""
import json
import math
from datetime import datetime
from pathlib import Path

import numpy as np
import pytest

from stonesoup.types.detection import Detection

from .heatmap import collect, density_grid

SAMPLE = Path(__file__).parent / 'sample_ais.json'
EPOCH = 1516233600  # 2018-01-18 00:00:00 UTC


# --------------------------------------------------------------------------
# Simulation helpers
# --------------------------------------------------------------------------

def write_ais_json(path, positions, start_time=EPOCH, interval=60):
    """Write AISHub-format JSON for a list of ``(lon, lat)`` positions.

    Longitude and latitude are stored the way AISHub does it, as degrees
    multiplied by 600,000, so this also exercises the reader's scaling.
    """
    records = [
        {
            'NAME': f'VESSEL {index}',
            'MMSI': 205466000 + index,
            'LONGITUDE': int(round(lon * 600000)),
            'LATITUDE': int(round(lat * 600000)),
            'TIME': str(start_time + index * interval),
            'SOG': 12.5,
            'COG': 90.0,
        }
        for index, (lon, lat) in enumerate(positions)
    ]
    path.write_text(json.dumps([{'ERROR': 'false'}, records]))
    return path


def simulate_cluster(centre, count, spread=0.02, seed=0):
    """Normally-distributed positions around a ``(lon, lat)`` centre."""
    rng = np.random.default_rng(seed)
    lon, lat = centre
    return list(zip(rng.normal(lon, spread, count),
                    rng.normal(lat, spread, count)))


def detections_from(positions, path):
    return collect(write_ais_json(path, positions))


def make_detections(positions):
    """Detections built directly, bypassing the reader."""
    return [Detection(np.array([[lon], [lat]]), datetime(2018, 1, 18))
            for lon, lat in positions]


@pytest.fixture
def sample_detections():
    return collect(SAMPLE)


# --------------------------------------------------------------------------
# Reading and parsing
# --------------------------------------------------------------------------

def test_collect_reads_every_record(sample_detections):
    with SAMPLE.open() as sample_file:
        expected = len(json.load(sample_file)[1])
    assert len(sample_detections) == expected


def test_collect_recovers_exact_positions(tmp_path):
    positions = [(1.25, 51.4), (0.5, 50.0), (2.0, 51.75)]
    detections = detections_from(positions, tmp_path / 'ais.json')

    recovered = sorted((float(d.state_vector[0]), float(d.state_vector[1]))
                       for d in detections)
    # Detections store float32, so ~1e-7 relative precision: about a
    # centimetre on the ground at these magnitudes.
    for (lon, lat), (want_lon, want_lat) in zip(recovered, sorted(positions)):
        assert lon == pytest.approx(want_lon, abs=1e-5)
        assert lat == pytest.approx(want_lat, abs=1e-5)


def test_collect_produces_two_row_state_vectors(tmp_path):
    detections = detections_from([(1.2, 51.1), (1.4, 51.3)],
                                 tmp_path / 'ais.json')
    assert all(d.state_vector.shape == (2, 1) for d in detections)


def test_collect_handles_southern_and_western_hemispheres(tmp_path):
    positions = [(-45.5, -22.9), (-46.1, -23.4), (-45.8, -23.1)]
    detections = detections_from(positions, tmp_path / 'ais.json')

    assert all(float(d.state_vector[0]) < 0 for d in detections)
    assert all(float(d.state_vector[1]) < 0 for d in detections)


def test_collect_parses_epoch_as_naive_utc(tmp_path):
    detections = detections_from([(1.2, 51.1)], tmp_path / 'ais.json')
    timestamp = detections[0].timestamp

    assert timestamp == datetime(2018, 1, 18, 0, 0, 0)
    assert timestamp.tzinfo is None


def test_collect_returns_detections_in_time_order(tmp_path):
    # Written out of order; the reader is expected to sort them.
    path = write_ais_json(tmp_path / 'ais.json',
                          [(1.2, 51.1), (1.3, 51.2), (1.4, 51.3)])
    payload = json.loads(path.read_text())
    payload[1].reverse()
    path.write_text(json.dumps(payload))

    timestamps = [d.timestamp for d in collect(path)]
    assert timestamps == sorted(timestamps)


def test_collect_keeps_identity_metadata_and_drops_position_keys(tmp_path):
    detections = detections_from([(1.2, 51.1)], tmp_path / 'ais.json')
    metadata = detections[0].metadata

    assert metadata['MMSI'] == 205466000
    assert metadata['NAME'] == 'VESSEL 0'
    assert metadata['SOG'] == 12.5
    # Position and time live on the Detection itself, not in metadata.
    for key in ('LONGITUDE', 'LATITUDE', 'TIME'):
        assert key not in metadata


# --------------------------------------------------------------------------
# Grid geometry
# --------------------------------------------------------------------------

@pytest.mark.parametrize('n_bins', [2, 8, 32, 64])
def test_grid_shape_follows_n_bins(sample_detections, n_bins):
    grid = density_grid(sample_detections, n_bins=n_bins)

    assert len(grid['lon']) == n_bins
    assert len(grid['lat']) == n_bins
    assert len(grid['z']) == n_bins
    assert all(len(row) == n_bins for row in grid['z'])
    assert grid['n_points'] == len(sample_detections)


def test_grid_spans_exactly_the_data_extent(sample_detections):
    grid = density_grid(sample_detections, n_bins=32)
    lon = [float(d.state_vector[0]) for d in sample_detections]
    lat = [float(d.state_vector[1]) for d in sample_detections]

    assert grid['lon'][0] == pytest.approx(min(lon))
    assert grid['lon'][-1] == pytest.approx(max(lon))
    assert grid['lat'][0] == pytest.approx(min(lat))
    assert grid['lat'][-1] == pytest.approx(max(lat))


def test_grid_axes_are_monotonically_increasing(sample_detections):
    grid = density_grid(sample_detections, n_bins=32)

    assert grid['lon'] == sorted(grid['lon'])
    assert grid['lat'] == sorted(grid['lat'])


def test_z_is_indexed_longitude_then_latitude(tmp_path):
    """z[i][j] must be the density at (lon[i], lat[j]), not the transpose.

    A transposed grid has an identical shape and would render as a
    plausible-looking but mirrored heatmap, so this is the one geometry
    error nothing else here would catch. The cluster is placed
    deliberately off-centre and at different fractions along each axis:
    20% along longitude, 80% along latitude. Under a transpose the peak
    would be reported at 80%/20% instead.
    """
    corners = [(0.0, 50.0), (1.0, 51.0)]  # pin the extent
    cluster = simulate_cluster((0.2, 50.8), 200, spread=0.01, seed=1)
    detections = detections_from(corners + cluster, tmp_path / 'ais.json')

    grid = density_grid(detections, n_bins=64)
    z = np.array(grid['z'])
    lon_index, lat_index = np.unravel_index(np.argmax(z), z.shape)

    assert grid['lon'][lon_index] == pytest.approx(0.2, abs=0.05)
    assert grid['lat'][lat_index] == pytest.approx(50.8, abs=0.05)


# --------------------------------------------------------------------------
# Density correctness
# --------------------------------------------------------------------------

def test_density_peaks_at_a_simulated_cluster(tmp_path):
    corners = [(0.0, 50.0), (2.0, 52.0)]
    cluster = simulate_cluster((1.5, 51.5), 300, spread=0.03, seed=2)
    detections = detections_from(corners + cluster, tmp_path / 'ais.json')

    grid = density_grid(detections, n_bins=64)
    z = np.array(grid['z'])
    lon_index, lat_index = np.unravel_index(np.argmax(z), z.shape)

    assert grid['lon'][lon_index] == pytest.approx(1.5, abs=0.1)
    assert grid['lat'][lat_index] == pytest.approx(51.5, abs=0.1)


def test_two_separated_clusters_are_both_visible(tmp_path):
    left = simulate_cluster((0.3, 50.3), 200, spread=0.02, seed=3)
    right = simulate_cluster((1.7, 51.7), 200, spread=0.02, seed=4)
    detections = detections_from(left + right, tmp_path / 'ais.json')

    grid = density_grid(detections, n_bins=64)
    z = np.array(grid['z'])
    lon = np.array(grid['lon'])
    lat = np.array(grid['lat'])

    def density_near(target_lon, target_lat):
        i = int(np.argmin(np.abs(lon - target_lon)))
        j = int(np.argmin(np.abs(lat - target_lat)))
        return z[i, j]

    midpoint = density_near(1.0, 51.0)
    assert density_near(0.3, 50.3) > midpoint * 10
    assert density_near(1.7, 51.7) > midpoint * 10


def test_empty_water_is_less_dense_than_shipping_lane(tmp_path):
    lane = [(0.2 + 0.02 * step, 50.2 + 0.02 * step) for step in range(60)]
    outlier = [(1.8, 50.1)]  # lone vessel far off the lane
    detections = detections_from(lane + outlier, tmp_path / 'ais.json')

    grid = density_grid(detections, n_bins=64)
    z = np.array(grid['z'])
    lon = np.array(grid['lon'])
    lat = np.array(grid['lat'])

    lane_i = int(np.argmin(np.abs(lon - 0.8)))
    lane_j = int(np.argmin(np.abs(lat - 50.8)))
    empty_i = int(np.argmin(np.abs(lon - 1.7)))
    empty_j = int(np.argmin(np.abs(lat - 51.0)))

    assert z[lane_i, lane_j] > z[empty_i, empty_j]


def test_density_integrates_to_approximately_one(tmp_path):
    """Sanity check on units: a KDE is a normalised probability density."""
    cluster = simulate_cluster((1.0, 51.0), 400, spread=0.15, seed=5)
    detections = detections_from(cluster, tmp_path / 'ais.json')

    n_bins = 128
    grid = density_grid(detections, n_bins=n_bins)
    z = np.array(grid['z'])
    d_lon = (grid['lon'][-1] - grid['lon'][0]) / (n_bins - 1)
    d_lat = (grid['lat'][-1] - grid['lat'][0]) / (n_bins - 1)

    # Below 1.0 because the grid is clipped to the data extent and the
    # KDE's tails extend past it.
    assert 0.5 < float(z.sum() * d_lon * d_lat) < 1.01


def test_density_is_positive_and_finite(sample_detections):
    z = np.array(density_grid(sample_detections, n_bins=32)['z'])

    assert np.all(np.isfinite(z))
    assert np.all(z > 0)


def test_matches_stone_soup_plot_density(sample_detections):
    """Must be the same calculation Stone Soup's own plotter performs.

    Compares against the real ``Plotter.plot_density`` output rather than
    a reimplementation of it, so a future change to the library's method
    surfaces here.

    The tolerance is loose because ``plot_density`` builds its arrays
    straight from the state vectors, which the AIS reader stores as
    float32, so it runs the KDE in single precision. ``density_grid``
    casts to float64 first. Same calculation, ~1e-5 relative difference
    in the arithmetic; ours is the more accurate of the two.
    """
    import matplotlib
    matplotlib.use('Agg')
    from stonesoup.plotter import Plotter
    from stonesoup.types.groundtruth import GroundTruthPath

    n_bins = 16
    grid = density_grid(sample_detections, n_bins=n_bins)

    plotter = Plotter()
    plotter.plot_density([GroundTruthPath(states=list(sample_detections))],
                         index=None, mapping=(0, 1), n_bins=n_bins)
    expected = np.asarray(plotter.ax.collections[0].get_array())

    np.testing.assert_allclose(np.array(grid['z']),
                               expected.reshape(n_bins, n_bins),
                               rtol=1e-3)


# --------------------------------------------------------------------------
# Determinism and purity
# --------------------------------------------------------------------------

def test_result_is_independent_of_detection_order(sample_detections):
    shuffled = list(sample_detections)
    np.random.default_rng(6).shuffle(shuffled)

    np.testing.assert_allclose(
        np.array(density_grid(sample_detections, n_bins=16)['z']),
        np.array(density_grid(shuffled, n_bins=16)['z']))


def test_repeated_calls_are_identical(sample_detections):
    first = density_grid(sample_detections, n_bins=16)
    second = density_grid(sample_detections, n_bins=16)
    assert first == second


def test_does_not_mutate_the_detections(sample_detections):
    before = [d.state_vector.copy() for d in sample_detections]
    density_grid(sample_detections, n_bins=16)

    for detection, original in zip(sample_detections, before):
        np.testing.assert_array_equal(detection.state_vector, original)


def test_output_is_json_serialisable(sample_detections):
    # Served over HTTP, so numpy scalars would raise at encode time.
    payload = json.dumps(density_grid(sample_detections, n_bins=16))
    restored = json.loads(payload)

    assert isinstance(restored['z'][0][0], float)
    assert isinstance(restored['n_points'], int)
    assert all(math.isfinite(value) for row in restored['z'] for value in row)


def test_handles_a_realistic_volume_of_vessels(tmp_path):
    """Guards against the KDE's O(points x grid) cost becoming a problem."""
    positions = simulate_cluster((1.0, 51.0), 2000, spread=0.4, seed=7)
    detections = detections_from(positions, tmp_path / 'ais.json')

    grid = density_grid(detections, n_bins=32)
    assert grid['n_points'] == 2000
    assert np.all(np.isfinite(np.array(grid['z'])))


# --------------------------------------------------------------------------
# Rejected input
# --------------------------------------------------------------------------

def test_no_detections_rejected():
    with pytest.raises(ValueError, match='no detections'):
        density_grid([])


def test_single_detection_rejected():
    with pytest.raises(ValueError, match='no spread'):
        density_grid(make_detections([(1.4, 51.2)]))


def test_all_vessels_at_one_position_rejected():
    with pytest.raises(ValueError, match='no spread'):
        density_grid(make_detections([(1.4, 51.2)] * 20))


def test_no_latitude_spread_rejected():
    positions = [(1.0 + 0.1 * step, 51.2) for step in range(10)]
    with pytest.raises(ValueError, match='no spread'):
        density_grid(make_detections(positions))


def test_perfectly_collinear_detections_rejected():
    """Collinear data is singular despite having spread on both axes.

    The offset latitude is derived from the same array as longitude so
    the points are exactly collinear in floating point. Computing it as
    ``50.0 + 0.1 * step`` instead leaves rounding crumbs that make the
    covariance merely near-singular, which the KDE accepts.
    """
    lon = np.linspace(0.0, 1.9, 20)
    positions = list(zip(lon, lon + 50.0))
    with pytest.raises(ValueError, match='degenerate'):
        density_grid(make_detections(positions))


def test_non_finite_positions_rejected():
    positions = [(1.2, 51.1), (1.4, 51.3), (float('nan'), 51.2)]
    with pytest.raises(ValueError, match='non-finite'):
        density_grid(make_detections(positions))


def test_bbox_excluding_everything_rejected():
    positions = simulate_cluster((1.0, 51.0), 20, seed=8)
    with pytest.raises(ValueError, match='inside bbox'):
        density_grid(make_detections(positions), bbox=(10.0, 10.0, 11.0, 11.0))


# --------------------------------------------------------------------------
# Clipping
# --------------------------------------------------------------------------

def test_bbox_drops_outliers_and_restores_the_extent(tmp_path):
    cluster = simulate_cluster((1.0, 51.0), 200, spread=0.05, seed=9)
    outlier = [(54.8, 51.0)]  # the shape of the Solent feed's bad record
    detections = detections_from(cluster + outlier, tmp_path / 'ais.json')

    unclipped = density_grid(detections, n_bins=32)
    clipped = density_grid(detections, n_bins=32, bbox=(0.0, 50.0, 2.0, 52.0))

    assert unclipped['lon'][-1] - unclipped['lon'][0] > 50
    assert clipped['lon'][-1] - clipped['lon'][0] < 1
    assert clipped['n_points'] == len(cluster)


def test_bbox_is_inclusive_at_its_edges():
    # Two points sit exactly on the bbox corners. The third is off the
    # diagonal so the three are not collinear, which would be singular.
    positions = [(1.0, 51.0), (2.0, 52.0), (1.2, 51.9)]
    grid = density_grid(make_detections(positions), n_bins=8,
                        bbox=(1.0, 51.0, 2.0, 52.0))

    assert grid['n_points'] == 3


def test_bbox_does_not_change_a_grid_that_contains_everything(
        sample_detections):
    wide = density_grid(sample_detections, n_bins=16,
                        bbox=(-180.0, -90.0, 180.0, 90.0))
    plain = density_grid(sample_detections, n_bins=16)

    assert wide == plain
