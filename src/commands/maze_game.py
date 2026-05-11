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


def axial_to_terminal_xy(q: int, r: int) -> tuple[int, int]:
    """
    Integer terminal coordinates for a flat-top hex grid.

    Direction behavior:
      m0: straight up
      m1: upper-right
      m2: lower-right
      m3: straight down
      m4: lower-left
      m5: upper-left

    This matches AXIAL_STEP:
      (0,-1), (1,-1), (1,0), (0,1), (-1,1), (-1,0)
    """
    x = 6 * q
    y = 4 * r + 2 * q
    return x, y


def step_axial(q: int, r: int, direction: int) -> tuple[int, int]:
    dq, dr = AXIAL_STEP[direction]
    return q + dq, r + dr


def axial_to_plot_xy(q: int, r: int) -> tuple[float, float]:
    """
    Pointy-top hex center coordinates matching the command directions:

        h0 / m0: straight up
        h1 / m1: upper-right
        h2 / m2: lower-right
        h3 / m3: straight down
        h4 / m4: lower-left
        h5 / m5: upper-left

    Hex radius is assumed to be 1.0 in plot_maze_hexbin.
    Adjacent centers are sqrt(3) apart, so hexes touch edge-to-edge.
    """
    x = 1.5 * q
    y = -math.sqrt(3) * (r + q / 2)
    return x, y


def plot_maze_hexbin(ax, ant_q: int, ant_r: int, radius: int) -> None:
    """
    Draw explicit touching pointy-top hex tiles.

    Do not use ax.hexbin here: hexbin rebins scattered sample points instead
    of treating your axial coordinates as literal cell centers.
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
            orientation=math.radians(30),  # flat top
            facecolor="orangered" if is_ant else "white",
            edgecolor="0.45",
            linewidth=1.0,
            zorder=2 if is_ant else 1,
        )
        ax.add_patch(tile)

    ant_x, ant_y = axial_to_plot_xy(ant_q, ant_r)
    ax.scatter(
        [ant_x],
        [ant_y],
        c=["orangered"],
        s=64,
        zorder=10,
        edgecolors="black",
        linewidths=0.8,
    )

    centers = [axial_to_plot_xy(q, r) for q, r in iter_disk_cells(radius)]
    xs = [x for x, _ in centers]
    ys = [y for _, y in centers]
    pad = 1.35

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
    """
    Render a real touching flat-top hex grid in the terminal.

    Split into:
      - draw_hex_edges: structural cell boundary
      - draw_cell_contents: normal cell marker
      - draw_ant: ant marker overlay
      - canvas_to_string: final terminal text
    """
    cells = iter_disk_cells(radius)
    canvas: dict[tuple[int, int], str] = {}

    def put(x: int, y: int, ch: str) -> None:
        existing = canvas.get((x, y))

        # Contents should win over edge/background characters.
        if ch in ("@", "."):
            canvas[(x, y)] = ch
            return

        # Do not overwrite cell contents with edges.
        if existing in ("@", "."):
            return

        canvas[(x, y)] = ch

    def draw_hex_edges(q: int, r: int) -> None:
        cx, cy = axial_to_terminal_xy(q, r)

        # Top and bottom horizontal edges.
        for dx in range(-2, 3):
            put(cx + dx, cy - 2, "_")
            put(cx + dx, cy + 2, "_")

        # Upper diagonals.
        put(cx - 3, cy - 1, "/")
        put(cx + 3, cy - 1, "\\")

        # Lower diagonals.
        put(cx - 3, cy + 1, "\\")
        put(cx + 3, cy + 1, "/")

    def draw_cell_contents(q: int, r: int) -> None:
        cx, cy = axial_to_terminal_xy(q, r)
        put(cx, cy, ".")

    def draw_ant(q: int, r: int) -> None:
        cx, cy = axial_to_terminal_xy(q, r)
        put(cx, cy, "@")

    def canvas_to_string() -> str:
        if not canvas:
            return ""

        min_x = min(x for x, _ in canvas)
        max_x = max(x for x, _ in canvas)
        min_y = min(y for _, y in canvas)
        max_y = max(y for _, y in canvas)

        lines: list[str] = []
        for y in range(min_y, max_y + 1):
            line = "".join(canvas.get((x, y), " ") for x in range(min_x, max_x + 1))
            lines.append(line.rstrip())

        return "\n".join(lines)

    # 1. Draw structure first.
    for q, r in cells:
        draw_hex_edges(q, r)

    # 2. Draw default contents.
    for q, r in cells:
        draw_cell_contents(q, r)

    # 3. Draw ant last so it overlays the normal marker.
    draw_ant(ant_q, ant_r)

    return canvas_to_string()


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
