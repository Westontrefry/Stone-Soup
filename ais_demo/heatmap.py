"""AIS detections in, heatmap-ready density grid out.

The whole pipeline is two steps::

    detections = collect(path)
    grid = density_grid(detections)

``grid`` is plain JSON-serialisable data, ready for a front end to draw.
"""
import numpy as np
from scipy.stats import gaussian_kde

from stonesoup.reader.aishub import JSON_AISDetectionReader

#: Indices of longitude and latitude in an AIS detection's state vector.
#:
#: ``JSON_AISDetectionReader`` emits a 2-row ``[[lon], [lat]]`` state
#: vector, so lon/lat are rows 0 and 1. Note this differs from
#: ``Plotter.plot_density``, which defaults to ``(0, 2)`` because it
#: assumes a ``[x, vx, y, vy]`` constant-velocity state.
LON_LAT_MAPPING = (0, 1)


def collect(path):
    """Drain an AIS source into a flat list of detections.

    Parameters
    ----------
    path : str or pathlib.Path
        Path to a JSON file in AISHub format.

    Returns
    -------
    : list of :class:`~.Detection`
        Every detection in the file, in the reader's time order.

    Notes
    -----
    This is the only function that knows where the data comes from.
    Pointing the demo at a live feed means changing this function and
    nothing else.
    """
    reader = JSON_AISDetectionReader(path)
    return [detection for _, detections in reader for detection in detections]


def density_grid(detections, n_bins=128, mapping=LON_LAT_MAPPING, bbox=None):
    """Turn AIS detections into a density grid for a heatmap.

    Evaluates a Gaussian kernel density estimate over the detections and
    samples it on a regular ``n_bins`` x ``n_bins`` grid spanning their
    extent. This is the same calculation as
    :meth:`~.Plotter.plot_density`, but it returns the numbers rather
    than drawing them, so the rendering can happen in the browser.

    Parameters
    ----------
    detections : sequence of :class:`~.Detection`
        AIS detections, as returned by :func:`collect`.
    n_bins : int
        Grid resolution per axis. Default 128, which keeps the JSON
        payload manageable; ``plot_density`` uses 300.
    mapping : tuple of int
        Rows of the state vector holding longitude and latitude.
    bbox : tuple of float, optional
        ``(min_lon, min_lat, max_lon, max_lat)``. Detections outside are
        discarded before the grid is built. Default `None`, no filtering.

        This matters more than it looks. The grid spans the extent of
        whatever it is given, so a single bad position drags the extent
        out with it and squeezes all the real data into a sliver of the
        map. Live AIS does contain such records: the Solent feed carries
        one vessel reporting a longitude of +54.8 instead of -1.1, which
        on its own stretches the grid across 56 degrees instead of 0.4.

    Returns
    -------
    : dict
        ``lon``: list of ``n_bins`` longitudes (grid columns).
        ``lat``: list of ``n_bins`` latitudes (grid rows).
        ``z``: ``n_bins`` x ``n_bins`` nested list of densities,
        indexed ``z[lon_index][lat_index]``.
        ``n_points``: number of detections the grid was built from,
        after any ``bbox`` filtering.

    Raises
    ------
    ValueError
        If there are no detections (or none left inside ``bbox``), or if
        they are too degenerate for a kernel density estimate (all at one
        spot, or perfectly collinear), which would make the KDE's
        covariance singular.
    """
    if len(detections) == 0:
        raise ValueError("Cannot build a density grid from no detections.")

    x = np.array([float(d.state_vector[mapping[0]]) for d in detections])
    y = np.array([float(d.state_vector[mapping[1]]) for d in detections])

    if bbox is not None:
        min_lon, min_lat, max_lon, max_lat = bbox
        inside = ((x >= min_lon) & (x <= max_lon)
                  & (y >= min_lat) & (y <= max_lat))
        x, y = x[inside], y[inside]
        if len(x) == 0:
            raise ValueError(
                f"Cannot build a density grid: no detections fall inside "
                f"bbox {bbox}.")

    # A single malformed record from a live feed can carry a NaN position.
    # NaN would propagate silently through the KDE and yield an all-NaN
    # grid, which renders as an empty heatmap with no error anywhere. Fail
    # loudly instead.
    if not (np.all(np.isfinite(x)) and np.all(np.isfinite(y))):
        raise ValueError(
            "Cannot build a density grid: detections contain non-finite "
            "longitude or latitude values.")

    # gaussian_kde inverts the data's covariance matrix, so it fails on
    # inputs with no spread. Check up front to give a clearer error than
    # the LinAlgError scipy would raise. Mirrors the guard in
    # Plotter.plot_density.
    if np.allclose(x, y, atol=1e-10):
        raise ValueError(
            "Cannot build a density grid: longitude and latitude values are "
            "identical, which makes the KDE covariance singular.")
    if np.ptp(x) == 0 or np.ptp(y) == 0:
        raise ValueError(
            "Cannot build a density grid: detections have no spread in "
            "longitude or latitude.")

    try:
        estimate = gaussian_kde([x, y])
    except np.linalg.LinAlgError as error:
        # Catches the degenerate cases the cheap checks above miss, most
        # notably perfectly collinear detections (a vessel reporting along
        # an exact straight line), which are singular despite having
        # spread on both axes.
        raise ValueError(
            "Cannot build a density grid: detections are degenerate "
            f"(collinear or duplicated), giving a singular KDE. {error}"
        ) from error
    xi, yi = np.mgrid[x.min():x.max():n_bins * 1j,
                      y.min():y.max():n_bins * 1j]
    zi = estimate(np.vstack([xi.flatten(), yi.flatten()]))

    return {
        'lon': xi[:, 0].tolist(),
        'lat': yi[0, :].tolist(),
        'z': zi.reshape(xi.shape).tolist(),
        'n_points': len(x),
    }
