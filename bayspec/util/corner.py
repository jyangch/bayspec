import logging
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from scipy.ndimage import gaussian_filter


def corner_plotly(
    xs,
    bins=30,
    ranges=None,
    weights=None,
    color=None,
    smooth1d=1, 
    smooth=1, 
    labels=None,
    quantiles=None,
    levels=None
    ):
    
    xs = _parse_input(xs)
    K = len(xs)
    
    bins = [int(bins) for _ in range(K)]
    
    if ranges is None:
        ranges = [[x.min(), x.max()] for x in xs]
    
    if color is None:
        color = '#08519c'
        
    if levels is None:
        levels = 1.0 - np.exp(-0.5 * np.array([1, 2]) ** 2)
        
    if quantiles is None:
        quantiles = []
        
    if labels is None:
        labels = [f'label{i}' for i in range(K)]

    fig = make_subplots(rows=K, cols=K, 
                        vertical_spacing=0.02, horizontal_spacing=0.02, 
                        shared_xaxes=False, shared_yaxes=False)

    for i, x in enumerate(xs):
        
        n_bins_1d = bins[i]
        bins_1d = np.linspace(min(ranges[i]), max(ranges[i]), n_bins_1d + 1)

        n, _ = np.histogram(x, bins=bins_1d, weights=weights, density=True)
        
        if smooth1d is not None:
            n = gaussian_filter(n, smooth1d)
            
        x0 = np.array(list(zip(bins_1d[:-1], bins_1d[1:]))).flatten()
        y0 = np.array(list(zip(n, n))).flatten()

        fig.add_trace(
            go.Scatter(x=x0, y=y0, mode='lines', 
                       name=labels[i], showlegend=False, 
                       line=dict(width=2, color=color)), 
            row=i + 1, col=i + 1)

        if len(quantiles) > 0:
            qvalues = quantile(x, quantiles, weights=weights)
            for q in qvalues:
                yq = y0[np.argmin(np.abs(q - x0))]
                fig.add_shape(
                    go.layout.Shape(
                        type="line",
                        x0=q, y0=0, x1=q, y1=yq,
                        name=labels[i], showlegend=False, 
                        line=dict(color=color, dash="dash")),
                    row=i + 1, col=i + 1)
            
        for j, y in enumerate(xs):
            if j >= i:
                continue

            fig = plot_hist2d(
                y, x, [bins[j], bins[i]], [ranges[j], ranges[i]],
                weights, smooth, [labels[j], labels[i]], levels, fig, (i, j))

    fig.update_layout(template='plotly_white', height=200*K, width=200*K)
    
    for i in range(K):
        for j in range(K):
            if j > i:
                continue
            else:
                fig.update_xaxes(tickangle=-45, row=i + 1, col=j + 1, showticklabels=False)
                fig.update_yaxes(tickangle=-45, row=i + 1, col=j + 1, showticklabels=False)

    for i in range(K):
        fig.update_xaxes(title_text=labels[i], row=K, col=i + 1, showticklabels=True)
        
    for i in range(1, K):
        fig.update_yaxes(title_text=labels[i], row=i + 1, col=1, showticklabels=True)

    return fig


def plot_hist2d(x, y, bins, ranges, weights, smooth, labels, levels, fig, subfig_idx):
    
    i2, j2 = subfig_idx
    
    bins_2d = []
    bins_2d.append(np.linspace(min(ranges[0]), max(ranges[0]), bins[0] + 1))
    bins_2d.append(np.linspace(min(ranges[1]), max(ranges[1]), bins[1] + 1))
    
    H, X, Y = np.histogram2d(x.flatten(), y.flatten(), bins=bins_2d, weights=weights)
    
    if smooth is not None:
        H = gaussian_filter(H, smooth)

    Hflat = H.flatten()
    inds = np.argsort(Hflat)[::-1]
    Hflat = Hflat[inds]
    
    sm = np.cumsum(Hflat)
    sm /= sm[-1]
    
    V = np.empty(len(levels))
    for i, v0 in enumerate(levels):
        try:
            V[i] = Hflat[sm <= v0][-1]
        except IndexError:
            V[i] = Hflat[0]
    V.sort()
    m = np.diff(V) == 0
    if np.any(m):
        logging.warning("Too few points to create valid contours")
    while np.any(m):
        V[np.where(m)[0][0]] *= 1.0 - 1e-4
        m = np.diff(V) == 0
    V.sort()

    X1, Y1 = 0.5 * (X[1:] + X[:-1]), 0.5 * (Y[1:] + Y[:-1])

    H2 = H.min() + np.zeros((H.shape[0] + 4, H.shape[1] + 4))
    H2[2:-2, 2:-2] = H
    H2[2:-2, 1] = H[:, 0]
    H2[2:-2, -2] = H[:, -1]
    H2[1, 2:-2] = H[0]
    H2[-2, 2:-2] = H[-1]
    H2[1, 1] = H[0, 0]
    H2[1, -2] = H[0, -1]
    H2[-2, 1] = H[-1, 0]
    H2[-2, -2] = H[-1, -1]
    
    X2 = np.concatenate(
        [
            X1[0] + np.array([-2, -1]) * np.diff(X1[:2]),
            X1,
            X1[-1] + np.array([1, 2]) * np.diff(X1[-2:])
            ]
        )
    
    Y2 = np.concatenate(
        [
            Y1[0] + np.array([-2, -1]) * np.diff(Y1[:2]),
            Y1,
            Y1[-1] + np.array([1, 2]) * np.diff(Y1[-2:])
            ]
        )
        
    fig.add_trace(
        go.Contour(
            z=H2.T, x=X2, y=Y2,
            name=f'{labels[0]}&{labels[1]}', 
            showlegend=False, 
            contours=dict(
                start=min(V),
                end=max(V),
                size=max(V) - min(V)),
            ncontours=len(V),
            colorscale='Blues',
            line=dict(width=2),
            showscale=False),
        row=i2 + 1, col=j2 + 1)

    return fig


def quantile(x, q, weights=None):
    
    x = np.atleast_1d(x)
    q = np.atleast_1d(q)

    if np.any(q < 0.0) or np.any(q > 1.0):
        raise ValueError("Quantiles must be between 0 and 1")

    if weights is None:
        return np.percentile(x, list(100.0 * q))
    else:
        weights = np.atleast_1d(weights)
        if len(x) != len(weights):
            raise ValueError("Dimension mismatch: len(weights) != len(x)")
        idx = np.argsort(x)
        sw = weights[idx]
        cdf = np.cumsum(sw)[:-1]
        cdf /= cdf[-1]
        cdf = np.append(0, cdf)
        return np.interp(q, cdf, x[idx]).tolist()


def _parse_input(xs):
    xs = np.atleast_1d(xs)
    if len(xs.shape) == 1:
        xs = np.atleast_2d(xs)
    else:
        assert len(xs.shape) == 2, "The input sample array must be 1- or 2-D."
        xs = xs.T
    return xs
