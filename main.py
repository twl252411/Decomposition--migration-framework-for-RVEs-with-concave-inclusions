import time

from intersection_seperation.rve_runner import (
    build_orientation_tensor,
    ensure_convex_partition,
    get_data_dir,
    normalize_circumscribe_mode,
    RVEConfig,
    run_rve)
from intersection_seperation.io_excel import save_run_metrics_txt


def build_sections(metrics):
    return [
        ("Run", [
            ("demo_shape", metrics.get("demo_shape")),
            ("circumscribing_strategy", metrics.get("circumscribing_strategy")),
            ("converged", metrics.get("converged")),
            ("run_id", metrics.get("run_id")),
            ("n_runs", metrics.get("n_runs")),
            ("attempts_total", metrics.get("attempts_total")),
        ]),
        ("Circumscribing", [
            ("circum_mode", metrics.get("circum_mode")),
            ("circum_method", metrics.get("circum_method")),
            ("circum_num_vertices", metrics.get("circum_num_vertices")),
            ("circum_num_points", metrics.get("circum_num_points")),
        ]),
        ("Packing and Convergence", [
            ("v1_max", metrics.get("v1_max")),
            ("v1_target", metrics.get("v1_target")),
            ("k_max", metrics.get("k_max")),
            ("k_mig", metrics.get("k_mig")),
            ("k_ori", metrics.get("k_ori")),
        ]),
        ("Geometric Surrogate", [
            ("N_v", metrics.get("N_v")),
            ("N_c", metrics.get("N_c")),
        ]),
        ("Contact Tolerance and Workload", [
            ("scale_ratio_collision", metrics.get("scale_ratio_collision")),
            ("N_GJK", metrics.get("N_GJK")),
            ("N_EPA", metrics.get("N_EPA")),
            ("n_pairs", metrics.get("n_pairs")),
        ]),
        ("Runtime", [
            ("t_dec", metrics.get("t_dec")),
            ("t_broad", metrics.get("t_broad")),
            ("t_narrow", metrics.get("t_narrow")),
            ("t_mig", metrics.get("t_mig")),
            ("t_tot", metrics.get("t_tot")),
        ]),
        ("Step-size and Tolerances", [
            ("tau_init", metrics.get("tau_init")),
            ("tau_final", metrics.get("tau_final")),
            ("eps", metrics.get("eps")),
            ("eps_ori", metrics.get("eps_ori")),
        ]),
    ]


def average_metrics(metrics_list):
    if not metrics_list:
        return {}
    avg = {}
    keys = set()
    for m in metrics_list:
        keys.update(m.keys())

    for key in keys:
        if key == "run_id":
            avg[key] = "avg"
            continue
        if key == "n_runs":
            avg[key] = len(metrics_list)
            continue

        values = [m.get(key) for m in metrics_list]
        values = [v for v in values if v is not None]
        if not values:
            avg[key] = None
            continue

        if all(isinstance(v, (bool, int, float)) for v in values):
            avg[key] = float(sum(float(v) for v in values) / float(len(values)))
            continue

        if all(isinstance(v, str) for v in values):
            avg[key] = values[0] if all(v == values[0] for v in values) else "mixed"
            continue

        avg[key] = values[0]

    return avg


def build_default_config(demo_shape="lobular2", step_scale=0.25):
    rve_size = 200.0
    if demo_shape == "concave_poly":
        vol_frac_inc = 0.67
    else:
        vol_frac_inc = 0.71
    circumscribe_mode = ["symmetric", "general"][0]
    general_num_points = 240
    ori_com_1, ori_com_2 = 0.5, 0
    ori_ten2 = build_orientation_tensor(ori_com_1, ori_com_2)
    rand_ori = False  # False=optimize to ori_ten2, True=random orientation
    scale_ratio_collision = 1.0
    max_iter = 5000
    show_unscaled_curve = True
    show_outer_polygon = True
    enable_plots = False
    seed = None  # 固定随机种子；设为 None 则每次随机
    lobular2_num_vertices = 16  # 必须是4的倍数
    pea_num_vertices = 16      # 必须是偶数
    return RVEConfig(
        demo_shape=demo_shape,
        rve_size=rve_size,
        vol_frac_inc=vol_frac_inc,
        ori_ten2=ori_ten2,
        step_scale=step_scale,
        rand_ori=rand_ori,
        scale_ratio_collision=scale_ratio_collision,
        max_iter=max_iter,
        circumscribe_mode=circumscribe_mode,
        general_num_points=general_num_points,
        show_unscaled_curve=show_unscaled_curve,
        show_outer_polygon=show_outer_polygon,
        enable_plots=enable_plots,
        seed=seed,
        lobular2_num_vertices=lobular2_num_vertices,
        pea_num_vertices=pea_num_vertices,
    )


def run_case(config, n_runs=20, max_attempts=None):
    data_dir = get_data_dir()
    mode = normalize_circumscribe_mode(config.demo_shape, config.circumscribe_mode)
    if config.demo_shape == "lobular2":
        circum_num_vertices = config.lobular2_num_vertices
    elif config.demo_shape == "pea":
        circum_num_vertices = config.pea_num_vertices
    else:
        circum_num_vertices = None
    circum_num_points = config.general_num_points if mode == "general" else None
    if mode == "symmetric":
        mode_tag = "symmetric"
    elif mode == "general":
        mode_tag = "general"
    else:
        mode_tag = mode

    if max_attempts is None:
        max_attempts = max(int(n_runs) * 10, int(n_runs))

    all_metrics = []
    success_id = 0
    attempt_id = 0
    while success_id < n_runs and attempt_id < max_attempts:
        attempt_id += 1
        t0 = time.time()
        try:
            dec_t0 = time.time()
            ensure_convex_partition(
                config.demo_shape,
                config.scale_ratio_collision,
                data_dir,
                lobular2_num_vertices=config.lobular2_num_vertices,
                pea_num_vertices=config.pea_num_vertices,
                circumscribe_mode=mode,
                general_num_points=config.general_num_points,
                plot=config.enable_plots,
            )
            t_dec = time.time() - dec_t0
            metrics = run_rve(
                config.rve_size,
                config.vol_frac_inc,
                config.ori_ten2,
                config.scale_ratio_collision,
                config.step_scale,
                demo_shape=config.demo_shape,
                circumscribe_mode=mode,
                circum_num_vertices=circum_num_vertices,
                circum_num_points=circum_num_points,
                max_iter=config.max_iter,
                include_periodic_images=True,
                rand_ori=config.rand_ori,
                show_unscaled_curve=config.show_unscaled_curve,
                show_outer_polygon=config.show_outer_polygon,
                enable_plots=config.enable_plots,
                seed=config.seed,
                return_metrics=True,
            )
            if metrics is None:
                raise RuntimeError("run_rve returned no metrics")
        except Exception as exc:
            print(f"run_failed attempt={attempt_id}: {exc}")
            continue

        if not bool(metrics.get("converged", False)):
            print(f"run_not_converged attempt={attempt_id}, skip")
            continue

        success_id += 1
        metrics["t_dec"] = float(t_dec)
        metrics["t_tot"] = float(time.time() - t0)
        metrics["run_id"] = int(success_id)
        metrics["n_runs"] = int(n_runs)

        sections = build_sections(metrics)
        metrics_path = data_dir / f"run_metrics_{config.demo_shape}_{mode_tag}_{success_id}.txt"
        save_run_metrics_txt(metrics_path, sections, start_time=None)

        all_metrics.append(metrics)
        print(f"run={success_id}, total_time={metrics['t_tot']:.2f}s")

    if success_id < n_runs:
        print(
            f"run_case_stopped shape={config.demo_shape}, mode={mode_tag}, "
            f"successes={success_id}/{n_runs}, attempts={attempt_id}/{max_attempts}"
        )

    if not all_metrics:
        raise RuntimeError(
            f"No converged runs collected for shape={config.demo_shape}, "
            f"mode={mode_tag} after {attempt_id} attempts."
        )

    avg_metrics = average_metrics(all_metrics)
    avg_metrics["demo_shape"] = config.demo_shape
    avg_metrics["circumscribing_strategy"] = all_metrics[0].get("circumscribing_strategy")
    avg_metrics["attempts_total"] = int(attempt_id)
    avg_sections = build_sections(avg_metrics)
    avg_path = (
        data_dir
        / f"run_metrics_{config.demo_shape}_{mode_tag}_avg{len(all_metrics)}_{config.step_scale}.txt"
    )
    save_run_metrics_txt(avg_path, avg_sections, start_time=None)
    print(f"avg_written={avg_path}")


def main():
    demo_shapes = ["lobular2", "pea", "concave_poly"]
    base_step_scales = [0.75, 0.5, 0.25]
    for demo_shape in demo_shapes:
        if demo_shape == "concave_poly":
            step_scales = [1.0 if s == 0.25 else s for s in base_step_scales]
        else:
            step_scales = base_step_scales
        for step_scale in step_scales:
            config = build_default_config(demo_shape=demo_shape, step_scale=step_scale)
            run_case(config)

if __name__ == "__main__":
    main()
