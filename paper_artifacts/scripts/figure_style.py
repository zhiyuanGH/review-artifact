"""Shared visual style for MOBench data figures.

General figure colors are loaded from ``figure_colors.json``. Paper data
bar-chart colors are loaded from ``data_figure_bar_colors.json`` so the
editable legend colors are grouped by figure, away from diagram/background
style settings. Paper data line-chart styles are loaded from
``data_figure_line_styles.json`` for the same reason.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


COLOR_CONFIG_PATH = Path(__file__).with_name("figure_colors.json")
BAR_COLOR_CONFIG_PATH = Path(__file__).with_name("data_figure_bar_colors.json")
LINE_STYLE_CONFIG_PATH = Path(__file__).with_name("data_figure_line_styles.json")


def load_json_config(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


COLOR_CONFIG = load_json_config(COLOR_CONFIG_PATH)
BAR_COLOR_CONFIG = load_json_config(BAR_COLOR_CONFIG_PATH)
LINE_STYLE_CONFIG = load_json_config(LINE_STYLE_CONFIG_PATH)
BASE_COLORS: dict[str, str] = dict(COLOR_CONFIG["base"])
SEMANTIC_COLORS: dict[str, dict[str, str]] = {
    key: dict(value) for key, value in COLOR_CONFIG["semantic"].items()
}
DATA_COLORS: dict[str, str] = dict(COLOR_CONFIG["data"])
DESIGN_COLORS: dict[str, str] = dict(COLOR_CONFIG["design"])
MOTIVATION_BACKGROUND_COLORS: dict[str, str] = dict(COLOR_CONFIG["motivation_background"])
SERIES_FALLBACK_COLORS: list[str] = list(COLOR_CONFIG["series_fallback"])


def semantic_color(category: str, key: str, *, fallback_index: int | None = None) -> str:
    colors = SEMANTIC_COLORS.get(category, {})
    if key in colors:
        return colors[key]
    if fallback_index is not None:
        return SERIES_FALLBACK_COLORS[fallback_index % len(SERIES_FALLBACK_COLORS)]
    return BASE_COLORS["gray"]


def semantic_series(category: str, keys: list[str]) -> list[str]:
    return [semantic_color(category, key, fallback_index=index) for index, key in enumerate(keys)]


METHOD_COLORS = SEMANTIC_COLORS["method"]
BACKEND_COLORS = SEMANTIC_COLORS["backend"]
DEVICE_COLORS = SEMANTIC_COLORS["device"]
MODEL_COLORS = SEMANTIC_COLORS["model"]
FRAMEWORK_COLORS = SEMANTIC_COLORS["framework"]

OURS = METHOD_COLORS["odabench"]
BASELINE = METHOD_COLORS["naive"]
ABLATION = METHOD_COLORS["ablation"]
BASELINE_2 = BASE_COLORS["baseline_2"]
NEUTRAL = BASE_COLORS["neutral"]
LIGHT_BLUE = DATA_COLORS["blue_light"]
PALE_BLUE = DATA_COLORS["blue_light"]
DIAGNOSTIC_BLUE = DATA_COLORS["diagnostic_blue"]
DIAGNOSTIC_BLUE_DARK = DATA_COLORS["diagnostic_blue_dark"]
COMPONENT_GREEN = DATA_COLORS["component_green"]
COMPONENT_KHAKI = DATA_COLORS["component_khaki"]
LIGHT_ORANGE = DATA_COLORS["light_orange"]
INK = BASE_COLORS["ink"]
GRID = BASE_COLORS["grid"]
WHITE = BASE_COLORS["white"]

PALETTE: dict[str, str] = {
    "ours": OURS,
    "baseline": BASELINE,
    "ablation": ABLATION,
    "baseline_2": BASELINE_2,
    "neutral": NEUTRAL,
    "light_blue": LIGHT_BLUE,
    "pale_blue": PALE_BLUE,
    "diagnostic_blue": DIAGNOSTIC_BLUE,
    "diagnostic_blue_dark": DIAGNOSTIC_BLUE_DARK,
    "component_green": COMPONENT_GREEN,
    "component_khaki": COMPONENT_KHAKI,
    "light_orange": LIGHT_ORANGE,
    "ink": INK,
    "grid": GRID,
    "white": WHITE,
}

SOLID_BAR_HATCH = ""
BAR_CONFIG: dict[str, Any] = dict(BAR_COLOR_CONFIG)
BAR_COLORS: dict[str, str] = dict(BAR_CONFIG["colors"])
BAR_FIGURES: dict[str, dict[str, Any]] = dict(BAR_CONFIG["figures"])


def _resolve_bar_color(method: str, group_or_color: str) -> str:
    if group_or_color in BAR_COLORS:
        return BAR_COLORS[group_or_color]
    if group_or_color.startswith("#"):
        return group_or_color
    raise ValueError(
        f"Unknown bar color group {group_or_color!r} for method {method!r} "
        f"in {BAR_COLOR_CONFIG_PATH}"
    )


def _iter_bar_figure_entries() -> Any:
    for figure_name, figure_config in BAR_FIGURES.items():
        entries = figure_config.get("entries", [])
        if isinstance(entries, dict):
            for method, entry_config in entries.items():
                if isinstance(entry_config, str):
                    yield figure_name, method, method, entry_config
                else:
                    yield figure_name, str(entry_config.get("legend", method)), method, str(entry_config["color"])
            continue
        for entry_config in entries:
            legend = str(entry_config["legend"])
            method_names = entry_config.get("methods", [entry_config.get("method")])
            group_or_color = str(entry_config["color"])
            for method in method_names:
                if method is None:
                    raise ValueError(
                        f"Bar entry {legend!r} in {figure_name} needs a method or methods field "
                        f"in {BAR_COLOR_CONFIG_PATH}."
                    )
                yield figure_name, legend, str(method), group_or_color


def _bar_method_groups_by_figure() -> dict[str, str]:
    groups: dict[str, str] = {}
    sources: dict[str, str] = {}
    for figure_name, legend, method, group_or_color in _iter_bar_figure_entries():
        color = _resolve_bar_color(method, group_or_color)
        source = f"{figure_name} / {legend}"
        if method in groups and groups[method] != color:
            raise ValueError(
                f"Conflicting bar colors for method {method!r}: "
                f"{sources[method]} uses {groups[method]}, but "
                f"{source} uses {color}. Keep each method identity "
                f"the same color across figures in {BAR_COLOR_CONFIG_PATH}."
            )
        groups[method] = color
        sources[method] = source
    return groups


BAR_IDENTITY_COLORS: dict[str, str] = _bar_method_groups_by_figure()
BAR_EDGE = str(BAR_CONFIG.get("edge", BASE_COLORS["gray_dark"]))
BAR_EDGE_WIDTH = float(BAR_CONFIG.get("edge_width", 0.35))


def bar_color(identity: str) -> str:
    """Return the configured semantic color for a bar-series identity."""
    if identity not in BAR_IDENTITY_COLORS:
        raise KeyError(
            f"Missing bar color for {identity!r}. Add it under the relevant "
            f"figure entry in {BAR_COLOR_CONFIG_PATH}."
        )
    return BAR_IDENTITY_COLORS[identity]


LINE_CONFIG: dict[str, Any] = dict(LINE_STYLE_CONFIG)
LINE_STYLE_GROUPS: dict[str, dict[str, str]] = dict(LINE_CONFIG["styles"])
LINE_FIGURES: dict[str, dict[str, Any]] = dict(LINE_CONFIG["figures"])
LINE_NORMAL_WIDTH = float(LINE_CONFIG.get("normal_linewidth", 1.8))
LINE_OURS_WIDTH = float(LINE_CONFIG.get("ours_linewidth", 2.3))
LINE_EMPHASIZE_OURS = bool(LINE_CONFIG.get("emphasize_ours", False))
LINE_MARKER_SIZE = float(LINE_CONFIG.get("markersize", 3.8))
LINE_MARKER_EDGE_COLOR = str(LINE_CONFIG.get("markeredgecolor", "#333333"))
LINE_MARKER_EDGE_WIDTH = float(LINE_CONFIG.get("markeredgewidth", 0.6))


def _resolve_line_style(method: str, style_name: str) -> dict[str, Any]:
    if style_name not in LINE_STYLE_GROUPS:
        raise ValueError(
            f"Unknown line style group {style_name!r} for method {method!r} "
            f"in {LINE_STYLE_CONFIG_PATH}"
        )
    style = dict(LINE_STYLE_GROUPS[style_name])
    style["style_name"] = style_name
    style["linewidth"] = (
        LINE_OURS_WIDTH if style_name == "Ours" and LINE_EMPHASIZE_OURS else LINE_NORMAL_WIDTH
    )
    style["markersize"] = LINE_MARKER_SIZE
    style["markeredgecolor"] = LINE_MARKER_EDGE_COLOR
    style["markeredgewidth"] = LINE_MARKER_EDGE_WIDTH
    return style


def _iter_line_figure_entries() -> Any:
    for figure_name, figure_config in LINE_FIGURES.items():
        for entry_config in figure_config.get("entries", []):
            legend = str(entry_config["legend"])
            method_names = entry_config.get("methods", [entry_config.get("method")])
            style_name = str(entry_config["style"])
            for method in method_names:
                if method is None:
                    raise ValueError(
                        f"Line entry {legend!r} in {figure_name} needs a method or methods field "
                        f"in {LINE_STYLE_CONFIG_PATH}."
                    )
                yield figure_name, legend, str(method), style_name


def _line_styles_by_figure() -> dict[str, dict[str, Any]]:
    styles: dict[str, dict[str, Any]] = {}
    sources: dict[str, str] = {}
    for figure_name, legend, method, style_name in _iter_line_figure_entries():
        style = _resolve_line_style(method, style_name)
        source = f"{figure_name} / {legend}"
        if method in styles and styles[method] != style:
            raise ValueError(
                f"Conflicting line styles for method {method!r}: "
                f"{sources[method]} and {source}. Keep each method identity "
                f"the same line style across figures in {LINE_STYLE_CONFIG_PATH}."
            )
        styles[method] = style
        sources[method] = source
    return styles


LINE_IDENTITY_STYLES: dict[str, dict[str, Any]] = _line_styles_by_figure()


def line_style(identity: str) -> dict[str, Any]:
    """Return the configured semantic line style for a data-series identity."""
    if identity not in LINE_IDENTITY_STYLES:
        raise KeyError(
            f"Missing line style for {identity!r}. Add it under the relevant "
            f"figure entry in {LINE_STYLE_CONFIG_PATH}."
        )
    return dict(LINE_IDENTITY_STYLES[identity])

METHOD_STYLES: dict[str, dict[str, Any]] = {
    "ours": {"color": OURS, "marker": "o", "linestyle": "-", "hatch": SOLID_BAR_HATCH},
    "baseline": {"color": BASELINE, "marker": "s", "linestyle": (0, (3, 2)), "hatch": SOLID_BAR_HATCH},
    "baseline_2": {"color": BASELINE_2, "marker": "D", "linestyle": (0, (1.5, 1.5)), "hatch": SOLID_BAR_HATCH},
    "ablation": {"color": ABLATION, "marker": "^", "linestyle": "-.", "hatch": SOLID_BAR_HATCH},
}

FONT: dict[str, Any] = {
    "family": "sans-serif",
    "sans_serif": ["Arial", "Helvetica", "DejaVu Sans", "sans-serif"],
    "base": 7.0,
    "title": 7.2,
    "label": 7.0,
    "tick": 6.2,
    "legend": 6.2,
    "annotation": 6.0,
}

STYLE: dict[str, float] = {
    "axes_linewidth": 0.7,
    "tick_width": 0.65,
    "grid_linewidth": 0.45,
    "bar_edge_width": 0.45,
    "bar_edge_width_small": 0.35,
    "line_width": 1.25,
    "marker_size": 3.1,
}


def configure_matplotlib(plt: Any) -> None:
    """Apply common Matplotlib rcParams for data figures."""
    plt.rcParams.update(
        {
            "font.family": FONT["family"],
            "font.sans-serif": FONT["sans_serif"],
            "font.size": FONT["base"],
            "axes.titlesize": FONT["title"],
            "axes.labelsize": FONT["label"],
            "xtick.labelsize": FONT["tick"],
            "ytick.labelsize": FONT["tick"],
            "legend.fontsize": FONT["legend"],
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.linewidth": STYLE["axes_linewidth"],
            "axes.edgecolor": BASELINE,
            "axes.labelcolor": INK,
            "axes.facecolor": WHITE,
            "xtick.major.width": STYLE["tick_width"],
            "ytick.major.width": STYLE["tick_width"],
            "xtick.color": INK,
            "ytick.color": INK,
            "grid.linewidth": STYLE["grid_linewidth"],
            "legend.frameon": True,
            "legend.facecolor": WHITE,
            "legend.edgecolor": NEUTRAL,
            "legend.framealpha": 1.0,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "svg.fonttype": "none",
            "savefig.facecolor": WHITE,
        }
    )


def box_axes(ax: Any, grid_axis: str = "none") -> None:
    """Use the shared four-sided axis style for data panels."""
    for side in ("top", "right", "left", "bottom"):
        ax.spines[side].set_visible(True)
        ax.spines[side].set_linewidth(STYLE["axes_linewidth"])
        ax.spines[side].set_color(BASELINE)
    ax.tick_params(
        top=False,
        right=False,
        direction="out",
        length=2.2,
        pad=1.2,
        colors=INK,
    )
    ax.grid(False)
    ax.set_axisbelow(True)


def add_panel_note(ax: Any, text: str, x: float, y: float) -> None:
    """Draw a consistent small annotation box inside a panel."""
    ax.text(
        x,
        y,
        text,
        transform=ax.transAxes,
        va="top",
        ha="left",
        color=INK,
        fontsize=FONT["annotation"],
        linespacing=1.15,
        bbox={
            "boxstyle": "round,pad=0.24,rounding_size=0.04",
            "facecolor": WHITE,
            "edgecolor": NEUTRAL,
            "linewidth": 0.45,
            "alpha": 0.96,
        },
    )


def boxed_legend(
    fig: Any,
    handles: list[Any],
    labels: list[str],
    *,
    ncol: int,
    anchor: tuple[float, float] = (0.5, 0.99),
    prop: dict[str, Any] | None = None,
    columnspacing: float = 0.8,
    handlelength: float = 0.9,
    handletextpad: float = 0.35,
    labelspacing: float = 0.25,
    markerscale: float = 0.85,
) -> Any:
    """Place a boxed legend above the plotting area."""
    legend = fig.legend(
        handles,
        labels,
        loc="upper center",
        bbox_to_anchor=anchor,
        ncol=ncol,
        frameon=True,
        fancybox=False,
        framealpha=1.0,
        borderaxespad=0.0,
        borderpad=0.22,
        columnspacing=columnspacing,
        handlelength=handlelength,
        handletextpad=handletextpad,
        labelspacing=labelspacing,
        markerscale=markerscale,
        prop=prop,
    )
    frame = legend.get_frame()
    frame.set_facecolor(WHITE)
    frame.set_edgecolor(NEUTRAL)
    frame.set_linewidth(0.45)
    return legend


def boxed_legend_above_axes(
    fig: Any,
    handles: list[Any],
    labels: list[str],
    *,
    ncol: int,
    axes: list[Any] | None = None,
    gap_points: float = 2.5,
    **style_kwargs: Any,
) -> Any:
    """Place a legend a small measured distance above the axes and titles."""
    legend = boxed_legend(fig, handles, labels, ncol=ncol, **style_kwargs)
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    plot_axes = list(fig.axes if axes is None else axes)
    axes_bounds = [ax.get_window_extent(renderer) for ax in plot_axes]
    content_bounds = [ax.get_tightbbox(renderer) for ax in plot_axes]
    legend_bounds = legend.get_window_extent(renderer)
    center_x = (min(b.x0 for b in axes_bounds) + max(b.x1 for b in axes_bounds)) / 2
    side_pad = 1.5 * fig.dpi / 72
    half_width = legend_bounds.width / 2
    center_x = max(fig.bbox.x0 + side_pad + half_width,
                   min(center_x, fig.bbox.x1 - side_pad - half_width))
    top_y = max(b.y1 for b in content_bounds if b is not None)
    anchor_y = top_y + gap_points * fig.dpi / 72 + legend_bounds.height
    anchor = fig.transFigure.inverted().transform((center_x, anchor_y))
    legend.set_bbox_to_anchor(anchor, transform=fig.transFigure)
    return legend


def save_pdf_svg(
    fig: Any,
    stem: Path,
    *,
    exact_canvas: bool = False,
    top_margin_points: float | None = None,
    tight_layout_pad: float = 0.15,
    pad_inches: float = 0.015,
    creator: str,
    preview_dir: Path | None = None,
) -> None:
    """Save a figure as PDF and SVG with shared metadata."""
    stem.parent.mkdir(parents=True, exist_ok=True)
    save_kwargs: dict[str, Any] = {
        "facecolor": WHITE,
        "metadata": {"Creator": creator},
        "pad_inches": pad_inches,
    }
    if exact_canvas:
        save_kwargs["bbox_inches"] = None
    else:
        fig.tight_layout(pad=tight_layout_pad)
        save_kwargs["bbox_inches"] = "tight"
    if exact_canvas and top_margin_points is not None:
        from matplotlib.transforms import Bbox

        fig.canvas.draw()
        top = fig.get_tightbbox(fig.canvas.get_renderer()).y1 + top_margin_points / 72
        save_kwargs["bbox_inches"] = Bbox.from_extents(0, 0, fig.get_figwidth(), top)
    fig.savefig(stem.with_suffix(".pdf"), **save_kwargs)
    fig.savefig(stem.with_suffix(".svg"), **save_kwargs)
    if preview_dir is not None:
        preview_dir.mkdir(parents=True, exist_ok=True)
        fig.savefig(
            preview_dir / f"{stem.name}.png",
            dpi=300,
            bbox_inches=save_kwargs.get("bbox_inches", "tight"),
            pad_inches=pad_inches,
            facecolor=WHITE,
        )
