from dataclasses import dataclass
from pathlib import Path
import time
from typing import Optional

import numpy as np

from intersection_seperation import reorientation as ro
from intersection_seperation.io_excel import (
    load_curve_outline_and_area,
    load_polygon_vertices_from_excel,
    save_polygon_data_txt,
)
from intersection_seperation.rve_types import PolygonShape
from intersection_seperation.solver import clamp_shapes_inside, separation_step, shapes_from_template, update_shapes
from intersection_seperation.viz import generate_shape_template, save_rve_plot
from shape_decomposition.concave_decomp_concave_poly import concave_decomp_concave_poly
from shape_decomposition.concave_decomp_l2 import concave_decomp_lobular2
from shape_decomposition.concave_decomp_pea import concave_decomp_pea


@dataclass(frozen=True)
class RVEConfig:
    demo_shape: str
    rve_size: float
    vol_frac_inc: float
    ori_ten2: np.ndarray
    step_scale: np.ndarray
    rand_ori: bool
    scale_ratio_collision: float
    max_iter: int
    circumscribe_mode: str = "symmetric"
    general_num_points: int = 240
    show_unscaled_curve: bool = True
    show_outer_polygon: bool = True
    enable_plots: bool = False
    seed: Optional[int] = 42
    lobular2_num_vertices: int = 16
    pea_num_vertices: int = 24


def build_orientation_tensor(ori_component_1, ori_component_2):
    return np.array(
        [[ori_component_1, ori_component_2, 0],
         [ori_component_2, 1 - ori_component_1, 0],
         [0, 0, 0],])


def get_data_dir() -> Path:
    data_dir = Path(__file__).resolve().parents[1] / "intermediate_final_files"
    data_dir.mkdir(parents=True, exist_ok=True)
    return data_dir


def normalize_circumscribe_mode(demo_shape: str, mode: Optional[str]) -> str:
    if demo_shape == "concave_poly":
        return "external"
    if mode is None:
        return "symmetric"
    m = str(mode).strip().lower()
    if m in ("general", "gen", "generic"):
        return "general"
    if m in ("symmetric", "sym", "duicheng", "对称"):
        return "symmetric"
    raise ValueError("circumscribe_mode must be 'symmetric' or 'general'.")


def circumscribing_method(demo_shape: str, circumscribe_mode: str) -> str:
    mode = normalize_circumscribe_mode(demo_shape, circumscribe_mode)
    if demo_shape == "pea":
        return "circum_concave_general" if mode == "general" else "circum_pea"
    if demo_shape == "lobular2":
        return "circum_concave_general" if mode == "general" else "circum_lobular2"
    if demo_shape == "concave_poly":
        return "circum_concave_poly"
    return "unknown"


def circumscribing_strategy(demo_shape: str, circumscribe_mode: str) -> str:
    mode = normalize_circumscribe_mode(demo_shape, circumscribe_mode)
    if demo_shape == "pea":
        if mode == "general":
            return "general concave polygon (circum_concave_general)"
        return "symmetry-preserving (x-axis, circum_pea)"
    if demo_shape == "lobular2":
        if mode == "general":
            return "general concave polygon (circum_concave_general)"
        return "symmetry-preserving (x/y-axis, circum_lobular2)"
    if demo_shape == "concave_poly":
        return "external polygon (circum_concave_poly)"
    return "unknown"


def ensure_convex_partition(
    demo_shape,
    scale_ratio_collision,
    data_dir: Optional[Path] = None,
    lobular2_num_vertices: int = 16,
    pea_num_vertices: int = 24,
    circumscribe_mode: str = "symmetric",
    general_num_points: int = 240,
    plot: bool = False,
):
    data_dir = data_dir or get_data_dir()
    mode = normalize_circumscribe_mode(demo_shape, circumscribe_mode)
    if demo_shape == "pea":
        concave_decomp_pea(
            scale_ratio_collision,
            num_vertices=pea_num_vertices,
            plot=plot,
            circumscribe_mode=mode,
            general_num_points=general_num_points,
        )
    elif demo_shape == "lobular2":
        concave_decomp_lobular2(
            scale_ratio_collision,
            num_vertices=lobular2_num_vertices,
            plot=plot,
            circumscribe_mode=mode,
            general_num_points=general_num_points,
        )
    elif demo_shape == "concave_poly":
        concave_decomp_concave_poly(scale_ratio_collision, plot=plot)
    else:
        raise ValueError("demo_shape must be 'pea', 'lobular2', or 'concave_poly'")


def run_rve(
    rve_size,
    vol_frac_inc,
    ori_ten2,
    scale_ratio_collision,
    step_scale=1.0,
    demo_shape: str = "lobular2",
    circumscribe_mode: str = "symmetric",
    circum_num_vertices: Optional[int] = None,
    circum_num_points: Optional[int] = None,
    max_iter=10000,
    include_periodic_images=True,
    rand_ori=False,
    show_unscaled_curve: bool = True,
    show_outer_polygon: bool = True,
    enable_plots: bool = False,
    seed: Optional[int] = 42,
    return_metrics: bool = False,
):

    if demo_shape in ("pea", "lobular2", "concave_poly"):
        data_dir = get_data_dir()
        template_xlsx = data_dir / f"convex_partition_{demo_shape}.xlsx"
        sheet_name = f"outer_polygon_{demo_shape}"
    else:
        raise ValueError("demo_shape must be 'pea', 'lobular2', or 'concave_poly'")

    template = generate_shape_template(demo_shape, template_xlsx, sheet_name=sheet_name)
    n_vertices = int(template.vertices.shape[0])
    n_convex_parts = int(len(template.parts))
    outline_vertices_curve, area_inc = load_curve_outline_and_area(demo_shape, template, data_dir)
    scaled_outline_vertices = None
    scaled_outline_xlsx = data_dir / f"outer_polygon_vertices_{demo_shape}.xlsx"
    scaled_outline_sheet = f"outer_polygon_{demo_shape}"
    if scaled_outline_xlsx.exists():
        try:
            scaled_outline_vertices = load_polygon_vertices_from_excel(
                scaled_outline_xlsx,
                sheet_name=scaled_outline_sheet,
            )
        except (FileNotFoundError, ValueError) as exc:
            print(f"Warning: failed to load scaled outline for {demo_shape}: {exc}")
    if scaled_outline_vertices is None and outline_vertices_curve is not None:
        scaled_outline_vertices = outline_vertices_curve * float(scale_ratio_collision)
    outline_vertices_to_plot = outline_vertices_curve if show_unscaled_curve else None
    scaled_outline_vertices_to_plot = scaled_outline_vertices if show_outer_polygon else None
    template_collision = template
    n_polygons = int(np.ceil(vol_frac_inc * rve_size * rve_size / (area_inc + 1e-30)))
    ini_shapes = shapes_from_template(template_collision, n_polygons, rve_size, seed=seed)

    angles = np.zeros((len(ini_shapes), 3), dtype=np.float64)
    angles[:, 1] = np.pi / 2.0
    ori_tolerance = 1e-4
    ori_optimized = False
    k_ori = None
    ori_max_iter = 1000
    if rand_ori:
        rng = np.random.default_rng(seed)
        angles[:, 0] = rng.uniform(0.0, 2.0 * np.pi, len(ini_shapes))
    else:
        if np.isclose(ori_ten2[0, 0], 1.0):
            angles[:, 0] = 0.0
        elif np.isclose(ori_ten2[1, 1], 1.0):
            angles[:, 0] = np.pi / 2.0
        else:
            beta = 1.0
            angles[:, 0] = np.array([s.angle for s in ini_shapes], dtype=np.float64)
            ori_vecs = np.column_stack(
                (np.sin(angles[:, 1]) * np.cos(angles[:, 0]), np.sin(angles[:, 1]) * np.sin(angles[:, 0]),
                 np.cos(angles[:, 1])))
            pred_ori_ten4 = ro.ori_tensor4_recon(ori_ten2)
            inc_ori_vecs, k_ori = ro.orivector_optimization(
                ori_vecs,
                pred_ori_ten4,
                beta,
                max_iter=ori_max_iter,
                tolerance=ori_tolerance,
            )
            ori_optimized = True
            angles = ro.optimized_ori_angles(inc_ori_vecs, angles)

    shapes = [
        PolygonShape(s.template, float(angles[i, 0]), s.translation.copy())
        for i, s in enumerate(ini_shapes)
    ]
    shapes = clamp_shapes_inside(shapes, rve_size)

    # The position-optimization stage starts from the orientation-optimized state.
    before = [PolygonShape(s.template, s.angle, s.translation.copy()) for s in shapes]

    overlay_vertices = []
    overlay_styles = []
    if scaled_outline_vertices_to_plot is None:
        overlay_vertices = [template.vertices]
        overlay_styles = [
            {"color": "black", "linestyle": "-", "linewidth": 1.2, "alpha": 0.9},
        ]

    if enable_plots:
        save_rve_plot(
            before,
            rve_size,
            out_path=data_dir / f"initial_rve_{demo_shape}.png",
            draw_parts=False,
            include_periodic_images=include_periodic_images,
            outline_vertices=outline_vertices_to_plot,
            scaled_outline_vertices=scaled_outline_vertices_to_plot,
            overlay_vertices=overlay_vertices,
            overlay_styles=overlay_styles,
        )

    cell_size = max(2.0 * template_collision.radius, 1e-6)

    threshold = 1.e-3
    step_min = 0.1
    step_max = 1.0
    prev_max_depth = None
    report_every = 100
    gjk_params = {
        "eps_gjk": 1e-8,
        "itmax_gjk": 64,
        "tol_epa": 1e-7,
        "itmax_epa": 48,
        "n_jobs": 0,
        "use_numba": True,
    }

    step_scale_init = float(step_scale)
    max_depth = 0.0
    it = 0
    iters_run = 0
    total_broad = 0.0
    total_narrow = 0.0
    total_gjk = 0
    total_epa = 0
    total_pairs = 0
    mig_t0 = time.perf_counter()
    for it in range(max_iter):
        if return_metrics:
            moves, max_depth, stats = separation_step(
                shapes,
                rve_size,
                cell_size,
                **gjk_params,
                return_stats=True,
            )
            total_broad += float(stats.get("t_broad", 0.0))
            total_narrow += float(stats.get("t_narrow", 0.0))
            total_gjk += int(stats.get("n_gjk", 0))
            total_epa += int(stats.get("n_epa", 0))
            total_pairs += int(stats.get("n_pairs", 0))
        else:
            moves, max_depth = separation_step(shapes, rve_size, cell_size, **gjk_params)
        if prev_max_depth is not None:
            if max_depth > prev_max_depth * 1.02:
                step_scale *= 0.5
            elif max_depth < prev_max_depth * 0.9:
                step_scale *= 1.05
        step_scale = min(step_max, max(step_min, step_scale))
        prev_max_depth = float(max_depth)

        shapes = update_shapes(shapes, moves, rve_size, step_scale=step_scale, cap_scale=0.75)

        iters_run = it + 1
        if (it) % report_every == 0:
            print(f"iter={it}, max_depth={max_depth:.6e}, step_scale={step_scale:.3f}")

        if max_depth < threshold:
            break
    t_mig = time.perf_counter() - mig_t0

    print(f"iters={iters_run}, max_depth={max_depth:.6e}, step_scale={step_scale:.3f}")

    if enable_plots:
        save_rve_plot(
            shapes,
            rve_size,
            out_path=data_dir / f"final_rve_{demo_shape}.png",
            draw_parts=False,
            include_periodic_images=include_periodic_images,
            outline_vertices=outline_vertices_to_plot,
            scaled_outline_vertices=scaled_outline_vertices_to_plot,
            overlay_vertices=overlay_vertices,
            overlay_styles=overlay_styles,
        )

    save_polygon_data_txt(
        before,
        data_dir / f"{demo_shape}_polygon_positions_initial.txt",
        include_periodic_images=False,
    )
    save_polygon_data_txt(
        shapes,
        data_dir / f"{demo_shape}_polygon_positions_after.txt",
        include_periodic_images=False,
    )
    save_polygon_data_txt(
        shapes,
        data_dir / f"{demo_shape}_polygon_positions_after_periodic.txt",
        include_periodic_images=True,
        rve_size=rve_size,
    )

    if not return_metrics:
        return None

    converged = bool(max_depth < threshold)
    v1_max = float(vol_frac_inc) if converged else None
    mode = normalize_circumscribe_mode(demo_shape, circumscribe_mode)
    metrics = {
        "demo_shape": demo_shape,
        "circumscribing_strategy": circumscribing_strategy(demo_shape, mode),
        "circum_mode": mode,
        "circum_method": circumscribing_method(demo_shape, mode),
        "circum_num_vertices": int(circum_num_vertices) if circum_num_vertices is not None else None,
        "circum_num_points": int(circum_num_points) if circum_num_points is not None else None,
        "v1_target": float(vol_frac_inc),
        "v1_max": v1_max,
        "converged": converged,
        "k_max": int(max_iter),
        "k_mig": int(iters_run),
        "N_v": n_vertices,
        "N_c": n_convex_parts,
        "delta_sep": float(threshold),
        "scale_ratio_collision": float(scale_ratio_collision),
        "N_GJK": int(total_gjk),
        "N_EPA": int(total_epa),
        "t_broad": float(total_broad),
        "t_narrow": float(total_narrow),
        "t_mig": float(t_mig),
        "tau_init": float(step_scale_init),
        "tau_final": float(step_scale),
        "eps": float(threshold),
        "eps_ori": float(ori_tolerance) if ori_optimized else None,
        "k_ori": int(k_ori) if ori_optimized and k_ori is not None else None,
        "n_pairs": int(total_pairs),
    }
    return metrics
