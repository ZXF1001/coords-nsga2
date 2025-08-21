def _plot_region_boundary(ax, region):
    """Plot the optimization region boundary"""
    if hasattr(region, 'exterior'):
        # Shapely polygon
        x, y = region.exterior.xy
        ax.plot(x, y, 'k', alpha=0.7)
        ax.fill(x, y, alpha=0.1, color='gray')