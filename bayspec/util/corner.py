import logging
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from scipy.ndimage import gaussian_filter
from typing import List, Optional, Sequence, Tuple, Union


def corner_plotly(
    xs: Union[np.ndarray, Sequence],
    bins: int = 30,
    ranges: Optional[List[Tuple[float, float]]] = None,
    weights: Optional[np.ndarray] = None,
    color: Optional[str] = None,
    smooth1d: Optional[float] = 1.0,
    smooth: Optional[float] = 1.0,
    labels: Optional[List[str]] = None,
    quantiles: Optional[List[float]] = None,
    levels: Optional[List[float]] = None
    ) -> go.Figure:

    xs = _parse_input(xs)
    K = xs.shape[0]
    
    bins_list = [int(bins)] * K
    
    if ranges is None:
        ranges = [(np.min(x), np.max(x)) for x in xs]
    
    if levels is None:
        levels = 1.0 - np.exp(-0.5 * np.array([1, 2]) ** 2)
        
    if quantiles is None:
        quantiles = []
        
    if labels is None:
        labels = [f'label{i}' for i in range(K)]
    
    if color is None:
        color = '#08519c'
        
    fig = make_subplots(
        rows=K, cols=K, 
        vertical_spacing=0.02, horizontal_spacing=0.02, 
        shared_xaxes=False, shared_yaxes=False)

    for i, x in enumerate(xs):
        n_bins_1d = bins_list[i]
        bins_1d = np.linspace(ranges[i][0], ranges[i][1], n_bins_1d + 1)

        n, _ = np.histogram(x, bins=bins_1d, weights=weights, density=True)
        
        if smooth1d is not None:
            n = gaussian_filter(n, smooth1d)
            
        x0 = np.repeat(bins_1d, 2)[1:-1]
        y0 = np.repeat(n, 2)

        # Plot 1D histogram on the diagonal
        fig.add_trace(
            go.Scatter(
                x=x0, y=y0, mode='lines', 
                name=labels[i], showlegend=False, 
                line=dict(width=2, color=color)), 
            row=i + 1, col=i + 1)

        # Plot quantiles on the 1D histogram
        if quantiles:
            qvalues = quantile(x, quantiles, weights=weights)
            for q in qvalues:
                idx = np.argmin(np.abs(q - x0))
                yq = y0[idx]
                fig.add_shape(
                    go.layout.Shape(
                        type="line",
                        x0=q, y0=0, x1=q, y1=yq,
                        name=labels[i], showlegend=False, 
                        line=dict(color=color, dash="dash")),
                    row=i + 1, col=i + 1)
                
        # Plot 2D histograms on the off-diagonals (lower triangle)
        for j, y in enumerate(xs[:i]):
            fig = plot_hist2d(
                y, x, 
                bins=[bins_list[j], bins_list[i]], 
                ranges=[ranges[j], ranges[i]],
                weights=weights, smooth=smooth, 
                labels=[labels[j], labels[i]], levels=levels, 
                fig=fig, subfig_idx=(i, j))

    fig.update_layout(template='plotly_white', height=200 * K, width=200 * K)
    
    # Hide all tick labels by default, set angle
    fig.update_xaxes(tickangle=-45, showticklabels=False)
    fig.update_yaxes(tickangle=-45, showticklabels=False)

    # Enable X tick labels for the bottom row, and set titles
    for i in range(K):
        fig.update_xaxes(title_text=labels[i], row=K, col=i + 1, showticklabels=True)

    # Enable Y tick labels for the leftmost column (skipping the top-left diagonal plot), and set titles
    for i in range(1, K):
        fig.update_yaxes(title_text=labels[i], row=i + 1, col=1, showticklabels=True)

    return fig


def plot_hist2d(
    x: np.ndarray, y: np.ndarray, 
    bins: List[int], ranges: List[Tuple[float, float]], 
    weights: Optional[np.ndarray], smooth: Optional[float], 
    labels: List[str], levels: np.ndarray, 
    fig: go.Figure, subfig_idx: Tuple[int, int]
    ) -> go.Figure:
    
    i2, j2 = subfig_idx
    
    bins_x = np.linspace(ranges[0][0], ranges[0][1], bins[0] + 1)
    bins_y = np.linspace(ranges[1][0], ranges[1][1], bins[1] + 1)
    
    H, X, Y = np.histogram2d(x.flatten(), y.flatten(), bins=[bins_x, bins_y], weights=weights)
    
    if smooth is not None:
        H = gaussian_filter(H, smooth)
        
    # Calculate contour levels using vectorized searchsorted
    Hflat = np.sort(H.flatten())[::-1]
    sm = np.cumsum(Hflat)
    sm /= sm[-1]

    # Find indices where the CDF crosses the requested levels
    idx = np.searchsorted(sm, levels, side='right') - 1
    idx = np.clip(idx, 0, len(Hflat) - 1)
    V = Hflat[idx]
    V.sort()
    
    # Handle edge case where contours might be identical
    m = np.diff(V) == 0
    if np.any(m):
        logging.warning("Too few points to create valid contours.")
        while np.any(m):
            V[np.where(m)[0][0]] *= (1.0 - 1e-4)
            m = np.diff(V) == 0
    V.sort()
    
    X1 = 0.5 * (X[1:] + X[:-1])
    Y1 = 0.5 * (Y[1:] + Y[:-1])

    # Pad H using numpy's native padding
    H_edge = np.pad(H, pad_width=1, mode='edge')
    H2 = np.pad(H_edge, pad_width=1, mode='constant', constant_values=H.min())
    
    # Extrapolate coordinates for the padded array
    X2 = np.concatenate([
        X1[0] + np.array([-2, -1]) * np.diff(X1[:2]),
        X1,
        X1[-1] + np.array([1, 2]) * np.diff(X1[-2:])])
    
    Y2 = np.concatenate([
        Y1[0] + np.array([-2, -1]) * np.diff(Y1[:2]),
        Y1,
        Y1[-1] + np.array([1, 2]) * np.diff(Y1[-2:])])

    fig.add_trace(
        go.Contour(
            z=H2.T, x=X2, y=Y2,
            name=f'{labels[0]}&{labels[1]}', 
            showlegend=False, 
            contours=dict(
                start=min(V),
                end=max(V),
                size=max(V) - min(V) if max(V) > min(V) else 1),
            ncontours=len(V),
            colorscale='Blues',
            line=dict(width=2),
            showscale=False),
        row=i2 + 1, col=j2 + 1)

    return fig


def quantile(
    x: np.ndarray, 
    q: Union[float, List[float]], 
    weights: Optional[np.ndarray] = None
    ) -> List[float]:
    
    x = np.atleast_1d(x)
    q = np.atleast_1d(q)

    if np.any(q < 0.0) or np.any(q > 1.0):
        raise ValueError("Quantiles must be strictly between 0 and 1.")

    if weights is None:
        return np.percentile(x, 100.0 * q).tolist()
        
    weights = np.atleast_1d(weights)
    if len(x) != len(weights):
        raise ValueError("Dimension mismatch: len(weights) must equal len(x).")
        
    idx = np.argsort(x)
    sw = weights[idx]
    cdf = np.cumsum(sw)[:-1]
    cdf /= cdf[-1]
    cdf = np.insert(cdf, 0, 0.0)
    
    return np.interp(q, cdf, x[idx]).tolist()


def _parse_input(
    xs: Union[np.ndarray, Sequence]
    ) -> np.ndarray:
    
    xs = np.atleast_1d(xs)
    if xs.ndim == 1:
        xs = xs[np.newaxis, :]
    elif xs.ndim == 2:
        xs = xs.T
    else:
        raise ValueError("The input sample array must be 1- or 2-D.")
    
    return xs
