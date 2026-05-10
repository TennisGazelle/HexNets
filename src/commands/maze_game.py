"""
Hex-disk mini-game helpers for `hexnet maze` (axial coords, bounded patch).
"""

from __future__ import annotations

import math

# Neighbor deltas in axial (q, r). hex_0 is screen-up (Δr = −1; smaller r = higher row).
# Indices 1–5 proceed clockwise for pointy-top hexes.
AXIAL_STEP: tuple[tuple[int, int], ...] = (
    (0, -1),
    (1, -1),
    (1, 0),
    (0, 1),
    (-1, 1),
    (-1, 0),
)


def axial_distance(q: int, r: int) -> int:
    return (abs(q) + abs(r) + abs(q + r)) // 2


def in_disk(q: int, r: int, radius: int) -> bool:
    return axial_distance(q, r) <= radius


def iter_disk_cells(radius: int) -> list[tuple[int, int]]:
    cells: list[tuple[int, int]] = []
    for q in range(-radius, radius + 1):
        for r in range(-radius, radius + 1):
            if in_disk(q, r, radius):
                cells.append((q, r))
    return cells


def oddq_col(q: int, r: int) -> int:
    """Column in odd-q vertical layout (for staggered ASCII rows)."""
    return q + (r - (r & 1)) // 2


def step_axial(q: int, r: int, direction: int) -> tuple[int, int]:
    dq, dr = AXIAL_STEP[direction]
    return q + dq, r + dr


def axial_to_plot_xy(q: int, r: int) -> tuple[float, float]:
    """
    Pointy-top axial hex center.

    Uses hex radius = 1.0, so neighboring hex centers are spaced exactly
    such that drawn RegularPolygon hexes touch edge-to-edge.
    """
    x = math.sqrt(3) * (q + r / 2)
    y = -1.5 * r
    return x, y


def plot_maze_hexbin(ax, ant_q: int, ant_r: int, radius: int) -> None:
    """
    Draw actual touching hex tiles instead of using matplotlib.hexbin.

    `hexbin` bins points into its own rectangular data grid, so your axial
    centers were being interpreted as samples, not as literal hex-cell centers.
    """
    from matplotlib.patches import RegularPolygon

    HEX_RADIUS = 1.0

    ax.clear()

    for q, r in iter_disk_cells(radius):
        x, y = axial_to_plot_xy(q, r)
        is_ant = (q, r) == (ant_q, ant_r)

        tile = RegularPolygon(
            (x, y),
            numVertices=6,
            radius=HEX_RADIUS,
            orientation=math.radians(30),  # pointy-top hex
            facecolor="orangered" if is_ant else "white",
            edgecolor="0.45",
            linewidth=1.0,
            zorder=2 if is_ant else 1,
        )
        ax.add_patch(tile)

    # Optional center dot for the ant, closer to your screenshot style.
    ant_x, ant_y = axial_to_plot_xy(ant_q, ant_r)
    ax.scatter(
        [ant_x],
        [ant_y],
        c=["orangered"],
        s=90,
        zorder=10,
        edgecolors="black",
        linewidths=0.8,
    )

    # Tight bounds around the disk.
    centers = [axial_to_plot_xy(q, r) for q, r in iter_disk_cells(radius)]
    xs = [x for x, _ in centers]
    ys = [y for _, y in centers]
    pad = 1.25

    ax.set_xlim(min(xs) - pad, max(xs) + pad)
    ax.set_ylim(min(ys) - pad, max(ys) + pad)
    ax.set_aspect("equal", adjustable="box")
    ax.axis("off")


def render_narrow(ant_q: int, ant_r: int, radius: int) -> str:
    lines = [
        "ASCII tiling needs ≥80 columns; neighbor map:",
        "Neighbors (hex_d → cell):",
    ]
    for d in range(6):
        nq, nr = step_axial(ant_q, ant_r, d)
        inside = in_disk(nq, nr, radius)
        status = "inside disk" if inside else "BLOCKED (edge)"
        lines.append(f"  m{d} / q{d}: hex_{d} → ({nq},{nr}) — {status}")
    lines.append("Commands: m0–m5 move, q0–q5 query (stub), help, quit.")
    return "\n".join(lines)


def render_wide(ant_q: int, ant_r: int, radius: int) -> str:
    """Place cells by pointy-top axial x; consecutive screen rows per axial r (no blank r-bands)."""
    cells = iter_disk_cells(radius)
    rs_sorted = sorted({r for _, r in cells})
    row_of_r = {r: i for i, r in enumerate(rs_sorted)}
    scale_x = 4.0
    staged: list[tuple[int, int, str]] = []
    for q, r in cells:
        xf = math.sqrt(3) * (q + r / 2)
        xi = int(round(xf * scale_x))
        yi = row_of_r[r]
        ch = "@" if (q, r) == (ant_q, ant_r) else "."
        staged.append((xi, yi, ch))
    staged.sort(key=lambda t: (t[1], t[0]))
    occupied: dict[tuple[int, int], str] = {}
    for xi, yi, ch in staged:
        while (xi, yi) in occupied:
            xi += 1
        occupied[(xi, yi)] = ch
    if not occupied:
        return ""
    min_x = min(x for x, _ in occupied)
    max_x = max(x for x, _ in occupied)
    min_y = min(y for _, y in occupied)
    max_y = max(y for _, y in occupied)
    lines: list[str] = []
    for y in range(min_y, max_y + 1):
        lines.append("".join(occupied.get((x, y), " ") for x in range(min_x, max_x + 1)))
    return "\n\n".join(lines)


def format_frame(
    ant_q: int,
    ant_r: int,
    radius: int,
    terminal_columns: int,
    successful_moves: int,
    last_line: str,
) -> str:
    header = (
        f"Hex maze · disk r={radius} · ant at ({ant_q},{ant_r}) · successful moves: {successful_moves}\n"
        f"hex_0 = up (−r), then clockwise · commands: m0–m5, q0–q5 (stub), help, quit\n"
        f"{last_line}\n"
    )
    if terminal_columns >= 80:
        body = render_wide(ant_q, ant_r, radius)
    else:
        body = render_narrow(ant_q, ant_r, radius)
    return header + body
