from pathlib import Path
from typing import List
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patheffects as pe

from .geom2d import aabb_from_points, angle_cs
from .io_excel import load_template_from_csv, load_template_from_excel
from .periodic import periodic_shift_candidates, polygon_vertices_world
from .rve_types import PolygonShape, ShapeTemplate


def transform_vertices(vertices: np.ndarray, angle: float, translation: np.ndarray) -> np.ndarray:
    c, s = angle_cs(angle)
    x = vertices[:, 0]
    y = vertices[:, 1]
    X = c * x - s * y + translation[0]
    Y = s * x + c * y + translation[1]
    return np.column_stack([X, Y])


def save_rve_plot(
    shapes: List[PolygonShape],
    L: float,
    out_path: str = "rve.png",
    draw_parts: bool = False,
    alpha: float = 0.3,
    include_periodic_images: bool = True,
    periodic_alpha: float = 0.12,
    outline_vertices: np.ndarray = None,
    scaled_outline_vertices: np.ndarray = None,
    scaled_outline_color: str = "blue",
    scaled_outline_lw: float = 1.2,
    scaled_outline_ls: str = "--",
    scaled_outline_alpha: float = 1.0,
    overlay_vertices: List[np.ndarray] = None,
    overlay_styles: List[dict] = None,
    figsize=(7.0, 7.0),
    dpi: int = 600,
    pad_inches: float = 0.02,
) -> str:
    """
    直接保存 RVE 图（不显示）。
    """
    fig = plt.figure(figsize=figsize)
    ax = fig.gca()

    plot_polygons(
        ax=ax,
        shapes=shapes,
        L=L,
        draw_parts=draw_parts,
        alpha=alpha,
        include_periodic_images=include_periodic_images,
        periodic_alpha=periodic_alpha,
        outline_vertices=outline_vertices,
        scaled_outline_vertices=scaled_outline_vertices,
        scaled_outline_color=scaled_outline_color,
        scaled_outline_lw=scaled_outline_lw,
        scaled_outline_ls=scaled_outline_ls,
        scaled_outline_alpha=scaled_outline_alpha,
        overlay_vertices=overlay_vertices,
        overlay_styles=overlay_styles,
    )

    fig.savefig(out_path, dpi=dpi, bbox_inches="tight", pad_inches=pad_inches)
    plt.close(fig)
    return out_path


def plot_polygons(
    ax,
    shapes: List[PolygonShape],
    L: float,
    draw_parts: bool = False,
    alpha: float = 0.25,
    include_periodic_images: bool = False,
    periodic_alpha: float = None,
    outline_vertices: np.ndarray = None,
    scaled_outline_vertices: np.ndarray = None,
    scaled_outline_color: str = "blue",
    scaled_outline_lw: float = 1.2,
    scaled_outline_ls: str = "--",
    scaled_outline_alpha: float = 1.0,
    overlay_vertices: List[np.ndarray] = None,
    overlay_styles: List[dict] = None,
):
    ax.clear()
    colors = plt.cm.tab20.colors
    overlay_vertices = [v for v in (overlay_vertices or []) if v is not None]
    if overlay_styles is None:
        overlay_styles = [{} for _ in overlay_vertices]
    else:
        overlay_styles = list(overlay_styles)
    if scaled_outline_vertices is not None:
        overlay_vertices.append(scaled_outline_vertices)
        overlay_styles.append({
            "color": scaled_outline_color,
            "linestyle": scaled_outline_ls,
            "linewidth": scaled_outline_lw,
            "alpha": scaled_outline_alpha,
        })
    if len(overlay_styles) != len(overlay_vertices):
        raise ValueError("overlay_styles length must match overlay_vertices length.")
    for i, s in enumerate(shapes):
        t = s.translation
        if outline_vertices is None:
            base_vertices = polygon_vertices_world(s)
        else:
            base_vertices = transform_vertices(outline_vertices, s.angle, t)
        shifts = periodic_shift_candidates(base_vertices, L) if include_periodic_images else [(0.0, 0.0)]

        for dx, dy in shifts:
            is_periodic = dx != 0.0 or dy != 0.0
            # 主体更“实”，周期映像仍“淡”
            base_alpha = alpha
            if is_periodic:
                fill_alpha = periodic_alpha if periodic_alpha is not None else (0.35 * base_alpha)
            else:
                fill_alpha = base_alpha

            t_shift = (t[0] + dx, t[1] + dy)
            if draw_parts:
                for k, part in enumerate(s.template.parts):
                    part_vertices = transform_vertices(part, s.angle, np.array(t_shift))
                    X = part_vertices[:, 0]
                    Y = part_vertices[:, 1]
                    ax.fill(
                        X,
                        Y,
                        alpha=fill_alpha,
                        color=colors[(i + k) % 20],
                        edgecolor=colors[(i + k) % 20],
                        linewidth=0.6,
                    )
            else:
                V = s.template.vertices if outline_vertices is None else outline_vertices
                body_vertices = transform_vertices(V, s.angle, np.array(t_shift))
                X = body_vertices[:, 0]
                Y = body_vertices[:, 1]
                lw = 0.9 if not is_periodic else 0.6
                z = 30 if not is_periodic else 15

                # 1) 只填充：不画边界，避免边界被 alpha 影响而变“虚”
                patch = ax.fill(
                    X,
                    Y,
                    facecolor=colors[i % 20],
                    alpha=fill_alpha,  # 只影响填充透明度
                    edgecolor="none",  # 关键：不在 fill 里画边界
                    linewidth=0.0,
                    zorder=z,
                    joinstyle="round",
                    capstyle="round",
                )[0]

                # 2) 单独画轮廓：不透明纯黑，像红色边框一样清晰
                Xc = np.r_[X, X[0]]
                Yc = np.r_[Y, Y[0]]

                lw_edge = (lw if outline_vertices is None else max(lw, 0.9))
                lw_edge = max(lw_edge, 1.0)  # 让轮廓至少 1.6，更“硬朗”

                ax.plot(
                    Xc,
                    Yc,
                    color="black",
                    linewidth=lw_edge,
                    alpha=1.0,  # 关键：边界不透明
                    zorder=z + 200,  # 确保压在填充之上
                    solid_joinstyle="round",
                    solid_capstyle="round",
                )

                for ov, style in zip(overlay_vertices, overlay_styles):
                    overlay_vertices_world = transform_vertices(ov, s.angle, np.array(t_shift))
                    Xo = np.r_[overlay_vertices_world[:, 0], overlay_vertices_world[0, 0]]
                    Yo = np.r_[overlay_vertices_world[:, 1], overlay_vertices_world[0, 1]]
                    ax.plot(
                        Xo,
                        Yo,
                        color=style.get("color", "black"),
                        linestyle=style.get("linestyle", "--"),
                        linewidth=style.get("linewidth", 1.1),
                        alpha=style.get("alpha", 0.85),
                        zorder=style.get("zorder", z + 250),
                        solid_joinstyle="round",
                        solid_capstyle="round",
                    )

                # 可选：如果你还想再硬一点（像红框那样），开启下面这行
                # ax.plot(Xc, Yc, color="black", linewidth=lw_edge + 0.4, alpha=1.0, zorder=z + 201)

                if outline_vertices is None:
                    # 关键：白色描边 + 黑色轮廓，叠加再多也清晰
                    patch.set_path_effects([
                        pe.Stroke(linewidth=lw + 1.6, foreground="white"),
                        pe.Stroke(linewidth=lw, foreground="black"),
                        pe.Normal(),
                    ])

    ax.plot([0, L, L, 0, 0], [0, 0, L, L, 0], color="r", linewidth=3.6, zorder=1000)
    ax.set_aspect("equal")
    ax.set_xlim(0, L)
    ax.set_ylim(0, L)
    ax.axis("off")


def generate_shape_template(
    shape: str,
    template_xlsx: str,
    sheet_name: str = "Convex_Parts",
) -> ShapeTemplate:
    template_path = Path(template_xlsx)

    if template_path.suffix.lower() == ".csv":
        template = load_template_from_csv(template_path)
    else:
        template = load_template_from_excel(template_path, sheet_name=sheet_name)

    # ---- normalize dtype ----
    local_center = np.asarray(template.local_center, dtype=np.float64)
    vertices = np.asarray(template.vertices, dtype=np.float64)
    parts = [np.asarray(part, dtype=np.float64) for part in template.parts]
    part_centers = np.asarray(template.part_centers, dtype=np.float64)
    part_radii = np.asarray(template.part_radii, dtype=np.float64)

    # ---- rebuild aabbs (in case template file changed) ----
    aabb_parts = np.array([aabb_from_points(part) for part in parts], dtype=np.float64)
    aabb_whole = aabb_from_points(vertices)

    # ---- radius ----
    radius = float(template.radius)
    if shape == "pea":
        # 保持你之前的 pea 逻辑：用顶点到 local_center 的最大距离作为包围半径
        radius = float(np.max(np.linalg.norm(vertices - local_center, axis=1)))

    return ShapeTemplate(
        aabb_parts=aabb_parts,
        aabb_whole=aabb_whole,
        local_center=local_center,
        part_centers=part_centers,
        part_normals=template.part_normals,
        part_radii=part_radii,
        parts=parts,
        radius=radius,
        vertices=vertices,
        is_convex=template.is_convex,
    )
