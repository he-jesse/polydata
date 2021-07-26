import numpy as np
import math
import numbers
import collections.abc
import warnings
from matplotlib import path
from itertools import combinations
from sklearn.metrics import pairwise_distances as pw_dist

def compute_overlap(points, polygons):
    """
    Compute the proportion of a 2-d point set that lies in more than one polygon.

    Parameters
    ----------
    points : ndarray of shape (N, 2)
        The set of points.
    
    polygons : array-like of matplotlib-path-Path object
        The list of polygons.

    Returns
    -------
    overlap : float
        The proportion of points that lie in the boundaries of more than one polygon.

    """
    overlap_pairwise = [np.logical_and(p1.contains_points(points), p2.contains_points(points)) for (p1, p2) in combinations(polygons, 2)]
    overlap_list = np.zeros(points.shape[0], dtype=bool)
    for ary in overlap_pairwise:
        overlap_list = np.logical_or(overlap_list, ary)
    return np.count_nonzero(overlap_list) / points.shape[0]

def make_overlap(points, labels, polygons, min_overlap=0.0, max_overlap=1.0):
    """
    Moves polygons and clusters closer to or farther from (0,0) to make the overlap fall in the specified range.

    Parameters
    ----------
    points : ndarray of shape (N, 2)
        The set of points.

    labels : array-like of int
        The integer labels of each point. Must be of length N, and each entry must be between 0 and the number of polygons.

    polygons : array-like of matplotlib-path-Path object
        The list of polygons.

    min_overlap : float, default=0.0
        The minimum proportion of points that lie in the boundaries of more than one polygon.
        Must be less than max_overlap.

    max_overlap : float, default=1.0
        The maximum proportion of points that lie in the boundaries of more than one polygon.
        Must be greater than min_overlap or 0.0.

    Returns
    -------
    new_points : ndarray of shape (N, 2)
        The set of shifted points.
        
    new_polygons : array-like of matplotlib.path.Path
        The list of shifted polygons.
    
    """
    if max_overlap > 0.0 and max_overlap <= min_overlap:
        raise ValueError('max_overlap must be bigger than min_overlap')
    overlap = compute_overlap(points, polygons)
    if overlap >= min_overlap and overlap <= max_overlap:
        return points, polygons
    vertices = [p.vertices for p in polygons]
    centroids = np.array([np.average(p_vertices, axis=0) for p_vertices in vertices])
    min_scale = -1.0
    max_scale = 1.0
    scale = (min_scale + max_scale) / 2
    while overlap < min_overlap or overlap > max_overlap:
        if scale == -1.0:
            warnings.warn("min_overlap is too high. All polygons are centered at (0,0).")
            break
        if overlap < min_overlap:
            max_scale = scale
        elif overlap > max_overlap:
            if max_scale - min_scale < 0.01:
                max_scale += 1.0
            min_scale = scale
        scale = (min_scale + max_scale) / 2
        new_points = points + centroids[labels] * scale
        new_polygons = [path.Path(poly + centroid * scale, closed=True) for (poly, centroid) in zip(vertices, list(centroids))]
        overlap = compute_overlap(new_points, new_polygons)
    return new_points, new_polygons

def random_poly(center=[0.0, 0.0], n_vertices=None, max_radius=1.0, min_radius=0.0, random_state=None):
    """
    Generates a random star-shaped polygon about a point.

    Parameters
    ----------
    center : array-like or tuple of float [x, y], default=[0.0, 0.0]
        The center point around which to generate vertices.
    
    n_vertices : int or None, default=None
        If int, the number of vertices. If None, the number of vertices is a random integer in [3, 21).
    
    max_radius : float, default=1.0
        The maximum distance from a vertex to the center. Must be greater than or equal to min_radius and must be > 0.

    min_radius : float, default=0.0
        The minimum distance from a vertex to the center. Must be less than or equal to min_radius and must be >= 0.

    random_state : int, RandomState instance or None, default=None
        Determines random number generation for dataset creation. Pass an int
        for reproducible output across multiple function calls.

    Returns
    -------
    polygon : matplotlib.path.Path instance
        The generated polygon.
    
    """
    if isinstance(n_vertices, numbers.Integral) and n_vertices < 3:
        raise ValueError('n_vertices must be > 3')
    if max_radius <= 0.0:
        raise ValueError('max_radius must be positive')
    if min_radius < 0.0:
        raise ValueError('min_radius cannot be negative')
    if max_radius < min_radius:
        raise ValueError('max_radius must be greater than or equal to min_radius')

    generator = np.random.default_rng(random_state)
    if n_vertices == None:
        n_vertices = generator.integers(3, 21)
    vertices = generator.random((n_vertices + 1, 2)) * [max_radius - min_radius, 2 * math.pi] + [min_radius, 0.0]
    vertices = vertices[vertices[:,1].argsort()]
    return path.Path((np.array([vertices[:,0] * np.cos(vertices[:,1]), vertices[:,0] * np.sin(vertices[:,1])]).transpose()) + center, closed=True)

def sample_poly(n_samples, polygon,
    dimension=2, distribution=None, center=None, std=1.0, outside=0.0, shuffle=True, random_state=None, max_iter=1000):
    """
    Draw samples from a polygon. For dimensions higher than 2, draw samples from a polytope whose orthogonal 2-d projections are the given polygon.

    Parameters
    ----------
    n_samples : int
        The total number of points to generate.

    polygon : matplotlib.path.Path instance
        The polygon to draw points from.

    dimension : int, default=2
        The number of features to generate. If dimension > 2, then points will be drawn based on the orthogonal 2-d projections of the polygon.
        Note that for dimension > 2, a polygon that is not centered at (0.0, 0.0) may be impossible to draw points from.
    
    distribution : string or None, default=None:
        If "Guassian" then generate points using a multivariate normal distribution.

    center : ndarray of size (1, dimension) or array-like of length dimension or None, default=None
        The center of a Gaussian distribution. If None, the distribution will be centered at the centroid of the polygon.

    std : float, default = 1.0
        The standard deviation of a Gaussian distribution.

    outside : float, default=0.0
        The total proportion of points that lie outside the given polygon.

    shuffle : bool, default=True
        Shuffle the samples.

    random_state : int, RandomState instance or None, default=None
        Determines random number generation for dataset creation. Pass an int
        for reproducible output across multiple function calls.

    max_iter : int, default=1000
        The maximum number of trial point sets to generate.
        If reached, the function will halt and return.

    Returns
    -------
    X : ndarray of size (n_samples, dimension)
        The generated points.
    
    """
    generator = np.random.default_rng(random_state)

    if distribution == 'Gaussian' and center is None:
        center = np.average(polygon.vertices, axis=0)
        if dimension > 2:
            center = np.append(center, np.zeros([1, dimension - 2]))
    elif distribution != 'Gaussian':
        x, y, width, height = polygon.get_extents().bounds
        box_dims = [width * (1.0 + 2 * outside), height * (1.0 + 2 * outside)]
        box_corner = [x - width * outside, y - height * outside]
        if dimension > 2:
            box_dims += [max(width, height) * (1.0 + 2 * outside) for i in range(dimension - 2)]
            if x < y:
                box_corner += [x - width * outside for i in range(dimension-2)]
            else:
                box_corner += [y - height * outside for i in range(dimension-2)]

    n_out = int(outside * n_samples)
    X_in = np.empty([0, dimension])
    X_out = np.empty([0, dimension])

    iterations = 0
    while X_in.shape[0] < n_samples - n_out or X_out.shape[0] < n_out:
        if distribution == 'Gaussian':
            X = generator.normal(center, std, (n_samples, dimension))
        else:
            X = generator.random((n_samples, dimension)) * box_dims + box_corner
            for (i, j) in combinations(range(dimension), 2):
                X = X[polygon.contains_points(X[:,[i,j]], radius = outside * min(width, height))]
        in_poly_list = np.ones(X.shape[0], dtype=bool)
        for (i, j) in combinations(range(dimension), 2):
            in_poly_list = np.logical_and(in_poly_list, polygon.contains_points(X[:,[i,j]]))
        X_in = np.vstack((X_in, X[in_poly_list]))
        X_out = np.vstack((X_out, X[np.logical_not(in_poly_list)]))
        iterations += 1
        if iterations > max_iter:
            warnings.warn("Max iterations reached.")
            break

    X = np.vstack((X_in[:n_samples - n_out,:], X_out[:n_out, :]))
    if shuffle:
        generator.shuffle(X)
    return X

def make_poly(n_samples=100, n_poly=1, dimension=2, *,
    min_center_distance=0.0, max_radius=1.0, min_radius=0.0, 
    outside=0.0, distribution=None, std=1.0, centers=None,
    bounding_box=(-10.0, 10.0), shuffle=True, random_state=None,
    max_iter=10):
    """
    Generate a random n-d clustering problem with polytope boundaries.

    Parameters
    ----------
    n_samples : int or array-like, default=100
        If int, it is the total number of points equally divided among clusters.
        If array-like, each element of the sequence indicates the number of samples per cluster.
    
    n_poly : int, default=1
        The number of clusters to generate.

    dimension : int, default=2
        The number of features to generate (and dimension of the polytopes).
        Must be at least 2.

    min_center_distance : float, default=0.0
        The minimum difference between cluster centers.

    max_radius : float or array-like, default=1.0
        If float, the maximum distance from a vertex to the center.
        If array-like, each entry in the sequence is the max_radius for the corresponding polytope.
        (Each entry) must be greater than or equal to min_radius and must be > 0.

    min_radius : float, default=0.0
        If float, the minimum distance from a vertex to the center.
        If array-like, each entry in the sequence is the mmin_radius for the corresponding polytope.
        (Each entry) must be less than or equal to max_radius and must be >= 0.
    
    outside : float or array-like, default=0.0
        If float, it is the total proportion of points that lie outside their assigned polytope.
        If array-like, each element of the sequence indicates the proportion of points that lie outside the corresponding polytope.

    distribution : array-like of string or string or None, default=None
        If "Gaussian" then generate points using a multivariate normal distribution centered inside each polytope
        with standard deviation std.
        If array-like then each element of the sequence specifies the distribution for the corresponding cluster.
        Otherwise, generate points with a uniform distribution.
    
    std : float or array-like, default = 1.0
        If float, the standard deviation of a Gaussian distribution.
        If array-like, each element in the sequence is the standard deviation of the corresponding Gaussian polytope.

    centers : ndarray of size (n_poly, 2) or None, default=None
        The centers around which to generate vertices, and points if the distribution is "Gaussian".
        If None, then centers will be randomly generated.
    
    bounding_box : tuple of float (min, max), default=(-10.0, 10.0)
        The bounding box for each cluster center when centers are
        generated at random.
    
    shuffle : bool, default=True
        Shuffle the samples.

    random_state : int, RandomState instance or None, default=None
        Determines random number generation for dataset creation. Pass an int
        for reproducible output across multiple function calls.
    
    max_iter : int, default=10
        The maximum number of times to call sample_poly per polytope.
        If reached, the function will move to the next polytope.

    Returns
    -------
    X : ndarray of size (N, 2), where N is the total number of samples.
        The generated points.
    
    y : array-like of length N.
        The integer labels for each point.

    poly : array-like of matplotlib.path.Path
        A list of generated polygons. If dimension=2, These will be the boundraries.
        If dimension > 2, these will be the (x_1, x_2)-projections of the polytopes.
        Note that each polytope will have the same shaped projection, but may be centered at different locations when projected
        onto different planes.

    centers : ndarray of size (n_poly, 2)
        The centers of each cluster.

    """
    generator = np.random.default_rng(random_state)

    if dimension < 2:
        raise ValueError('dimension must be >= 2')

    if isinstance(n_samples, collections.abc.Iterable):
        if len(n_samples) != n_poly:
            raise ValueError('Length of n_samples must match n_poly')
        samples_per_poly = n_samples
    else:
        samples_per_poly = [n_samples // n_poly for i in range(n_poly)]
        for i in range(n_samples % n_poly):
            samples_per_poly[i] += 1
    
    if isinstance(max_radius, collections.abc.Iterable):
        if len(max_radius) != n_poly:
            raise ValueError('Length of max_radius must match n_poly')
        max_r_per_poly = max_radius
    else:
        max_r_per_poly = [max_radius for i in range(n_poly)]

    if isinstance(min_radius, collections.abc.Iterable):
        if len(min_radius) != n_poly:
            raise ValueError('Length of min_radius must match n_poly')
        min_r_per_poly = min_radius
    else:
        min_r_per_poly = [min_radius for i in range(n_poly)]

    if isinstance(outside, collections.abc.Iterable):
        if len(outside) != n_poly:
            raise ValueError('Length of n_samples must match n_poly')
        out_per_poly = outside
    else:
        out_per_poly = [outside for i in range(n_poly)]

    if isinstance(distribution, collections.abc.Iterable) and not isinstance(distribution, str):
        if len(distribution) != n_poly:
            raise ValueError('Length of distribution must match n_poly')
        dist_per_poly = distribution
    else:
        dist_per_poly = [distribution for i in range(n_poly)]

    if isinstance(std, collections.abc.Iterable):
        if len(std) != n_poly:
            raise ValueError('Length of std must match n_poly')
        std_per_poly = std
    else:
        std_per_poly = [std for i in range(n_poly)]

    if isinstance(centers, collections.abc.Iterable) and len(centers) != n_poly:
            raise ValueError('Length of centers must match n_poly')
    elif isinstance(centers, numbers.Integral):
        raise ValueError('centers must match n_poly')
    while centers is None or np.any(pw_dist(centers) + min_center_distance * np.identity(len(centers)) < min_center_distance):
        centers = (bounding_box[1] - bounding_box[0]) * generator.random((n_poly, dimension)) + bounding_box[0]

    polygons = []
    X = np.empty([0, dimension])
    y = np.empty([1, 0], dtype=int)

    label = 0
    for (samples, out, dist, stdev, center, max_r, min_r) in \
        zip(samples_per_poly, out_per_poly, dist_per_poly, std_per_poly, centers, max_r_per_poly, min_r_per_poly):
        for i in range(max_iter):
            with warnings.catch_warnings():
                warnings.filterwarnings('error')
                polygon = random_poly([0.0, 0.0], None, max_r, min_r, generator)
                X_new = np.empty([0, dimension])
                try:
                    X_new = sample_poly(samples, polygon, dimension, dist, np.zeros([1, dimension]), stdev, out, shuffle, random_state) + center
                except Warning:
                    continue
                else:
                    break
        else:
            warnings.warn("max_iter reached. std may be too small")
        X = np.vstack((X, X_new))
        polygons.append(path.Path(polygon.vertices + center[:2], closed=True))
        y = np.append(y, label * np.ones([1, samples], dtype=int))
        label += 1
    
    if shuffle:
        p = generator.permutation(X.shape[0])
        X = X[p]
        y = y[p]
    return X, y, polygons, centers

def make_correlated_features(points, n_features=1, noise=0.0, random_state=None):
    """
    Generate new coordinates as a random linear combination of given points and adds noise.

    Parameters
    ----------
    points : ndarray of size (N, M)
        The supplied points.
    
    n_features : int, default=1
        The number of features to generate.

    noise : float or array-like of length n_features, default = 0.0
        If float, the standard deviation of Gaussian noise to add the the new data.
        If array-like, each element in the sequence is the noise to add to each dimension.
        Must be non-negative.

    random_state : int, RandomState instance or None, default=None
        Determines random number generation for dataset creation. Pass an int
        for reproducible output across multiple function calls.

    Returns
    -------
    X : ndarray of size (N, M + n_features)
        The new array.
    
    """
    generator = np.random.default_rng(random_state)
    n_informative = points.shape[1]
    coef = 2 * generator.random((n_informative, n_features)) - 1
    X_new = np.dot(points, coef)
    X_new += generator.normal(scale=noise, size=X_new.shape)
    return np.concatenate((points, X_new), axis=1)