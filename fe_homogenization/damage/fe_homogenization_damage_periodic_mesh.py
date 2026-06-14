# -*- coding: utf-8 -*-
# Single-RVE Abaqus/CAE preprocessing script for damage periodic meshing.
# It builds one periodic mesh CAE and writes the boundary self-check summary.

from __future__ import print_function

import csv
import math
import os

from abaqus import *
from abaqusConstants import *
from caeModules import *
from driverUtils import executeOnCaeStartup
import mesh


executeOnCaeStartup()
session.journalOptions.setValues(replayGeometry=INDEX, recoverGeometry=INDEX)

try:
    script_dir = os.path.dirname(os.path.abspath(__file__))
except Exception:
    script_dir = os.getcwd()
os.chdir(script_dir)


# -----------------------------------------------------------------------------
# User settings
# -----------------------------------------------------------------------------
SCRIPT_TAG = "fe_homogenization_damage_periodic_mesh"
shape_name = os.environ.get("FE_HOMOG_DAMAGE_SHAPE", "lobular2")
rve_id = int(os.environ.get("FE_HOMOG_DAMAGE_RVE_ID", "1"))
target_n = 120
target_eq_radius = 5.0
target_vf = 0.50
num_ele = int(os.environ.get("FE_HOMOG_DAMAGE_NUM_ELE", "80"))

L_default = math.sqrt(target_n * math.pi * target_eq_radius * target_eq_radius / target_vf)
face_probe_tol = max(1.0e-8 * L_default, 1.0e-6)

mesh_deviation_factor = 0.05
mesh_min_size_factor = 0.20
mesh_algorithm = ADVANCING_FRONT
mesh_min_transition = ON
periodic_mesh_self_check = True
periodic_mesh_repair_iterations = 3
boundary_tol_factor = 1.0e-6
min_boundary_tol = 1.0e-6

src_dir = os.path.abspath(os.path.join(script_dir, "abaqus_rve"))
out_dir = os.path.abspath(
    os.environ.get(
        "FE_HOMOG_DAMAGE_PERIODIC_DIR",
        os.path.join(script_dir, "concave_periodic_mesh_cae_%d" % int(target_n)),
    )
)

if not os.path.isdir(out_dir):
    os.makedirs(out_dir)


def _resolve_src_cae(shape_name, rve_id):
    cands = [
        os.path.join(src_dir, "%s_rve_%02d_n%d.cae" % (shape_name, int(rve_id), int(target_n))),
        os.path.join(src_dir, "%s_rve_%d_n%d.cae" % (shape_name, int(rve_id), int(target_n))),
        os.path.join(src_dir, "%s_rve_%02d.cae" % (shape_name, int(rve_id))),
        os.path.join(src_dir, "%s_rve_%d.cae" % (shape_name, int(rve_id))),
    ]
    for path in cands:
        if os.path.isfile(path):
            return path
    return None

def _unwrap_face(face_hit):
    if face_hit is None:
        return None
    try:
        _ = face_hit.pointOn
        return face_hit
    except Exception:
        pass
    try:
        if len(face_hit) > 0:
            return face_hit[0]
    except Exception:
        pass
    return None


def _find_face_by_point(face_array, px, py, eps):
    probes = (
        (px, py, 0.0),
        (px + eps, py, 0.0),
        (px - eps, py, 0.0),
        (px, py + eps, 0.0),
        (px, py - eps, 0.0),
    )
    for q in probes:
        try:
            return _unwrap_face(face_array.findAt((q,)))
        except Exception:
            pass
        try:
            return _unwrap_face(face_array.findAt(q))
        except Exception:
            pass
    return None


def _face_index(face_obj):
    try:
        return int(face_obj.index)
    except Exception:
        return int(face_obj.index())


def _append_face(face_seq, part_faces, face_index):
    add_seq = part_faces[int(face_index):int(face_index) + 1]
    if face_seq is None:
        return add_seq
    return face_seq + add_seq


def _build_face_sets_on_merged_part(model, merged_part, matrix_part_name):
    if matrix_part_name not in model.parts.keys():
        raise RuntimeError("Matrix part %s not found when building face sets." % matrix_part_name)

    pmat = model.parts[matrix_part_name]
    if len(pmat.faces) == 0:
        raise RuntimeError("Matrix part has no faces.")

    pt0 = pmat.faces[0].pointOn[0]
    probe_x = float(pt0[0])
    probe_y = float(pt0[1])
    matrix_face = _find_face_by_point(merged_part.faces, probe_x, probe_y, face_probe_tol)
    if matrix_face is None:
        raise RuntimeError("Cannot identify merged matrix face from probe point (%.12g, %.12g)." % (probe_x, probe_y))

    matrix_idx = _face_index(matrix_face)
    matrix_seq = merged_part.faces[matrix_idx:matrix_idx + 1]
    fiber_seq = None

    for face in merged_part.faces:
        idx = _face_index(face)
        if idx == matrix_idx:
            continue
        fiber_seq = _append_face(fiber_seq, merged_part.faces, idx)

    if "Matrix-Faces" in merged_part.sets.keys():
        del merged_part.sets["Matrix-Faces"]
    merged_part.Set(name="Matrix-Faces", faces=matrix_seq)

    if fiber_seq is None:
        raise RuntimeError("Merged part has no fiber faces.")
    if "Fiber-Faces" in merged_part.sets.keys():
        del merged_part.sets["Fiber-Faces"]
    merged_part.Set(name="Fiber-Faces", faces=fiber_seq)


def _mesh_merged_part(merged_part, mesh_size):
    elem_quad = mesh.ElemType(elemCode=CPE4R, elemLibrary=EXPLICIT, hourglassControl=DEFAULT)
    elem_tri = mesh.ElemType(elemCode=CPE3, elemLibrary=EXPLICIT)
    merged_part.seedPart(
        size=float(mesh_size),
        deviationFactor=mesh_deviation_factor,
        minSizeFactor=mesh_min_size_factor
    )
    if len(merged_part.edges) > 0:
        merged_part.seedEdgeBySize(
            edges=merged_part.edges[:],
            size=float(mesh_size),
            deviationFactor=mesh_deviation_factor,
            minSizeFactor=mesh_min_size_factor,
            constraint=FIXED,
        )
    merged_part.setMeshControls(
        regions=merged_part.faces[:],
        elemShape=QUAD_DOMINATED,
        technique=FREE,
        algorithm=mesh_algorithm,
        minTransition=mesh_min_transition,
    )
    merged_part.setElementType(regions=(merged_part.faces[:],), elemTypes=(elem_quad, elem_tri))
    merged_part.generateMesh()


def _is_close(value, target, tol):
    return abs(float(value) - float(target)) <= tol


def _edge_index(edge_obj):
    try:
        return int(edge_obj.index)
    except Exception:
        return -1


def _edge_endpoints(part, edge_obj):
    vtags = edge_obj.getVertices()
    if len(vtags) < 2:
        return None
    p0 = part.vertices[int(vtags[0])].pointOn[0]
    p1 = part.vertices[int(vtags[-1])].pointOn[0]
    return (
        (float(p0[0]), float(p0[1])),
        (float(p1[0]), float(p1[1])),
    )


def _project_param_and_offline(x, y, p0, p1):
    dx = float(p1[0] - p0[0])
    dy = float(p1[1] - p0[1])
    ll = dx * dx + dy * dy
    if ll <= 1.0e-30:
        return 0.0, math.sqrt((x - p0[0]) ** 2 + (y - p0[1]) ** 2)
    rx = float(x - p0[0])
    ry = float(y - p0[1])
    t = (rx * dx + ry * dy) / ll
    px = float(p0[0] + t * dx)
    py = float(p0[1] + t * dy)
    off = math.sqrt((x - px) ** 2 + (y - py) ** 2)
    return float(t), float(off)


def _point_segment_distance(x, y, p0, p1):
    tt, off = _project_param_and_offline(x, y, p0, p1)
    if tt < 0.0:
        return math.sqrt((x - p0[0]) ** 2 + (y - p0[1]) ** 2)
    if tt > 1.0:
        return math.sqrt((x - p1[0]) ** 2 + (y - p1[1]) ** 2)
    return float(off)


def _get_edge_seed_value(part, edge_obj, attr_name):
    attr = globals().get(attr_name, None)
    if attr is None:
        return None
    try:
        return part.getEdgeSeeds(edge=edge_obj, attribute=attr)
    except Exception:
        try:
            return part.getEdgeSeeds(edge=edge_obj, attribute=attr_name)
        except Exception:
            return None


def _get_edge_length(part, edge_obj):
    endpoints = _edge_endpoints(part, edge_obj)
    if endpoints is None:
        return 0.0
    p0, p1 = endpoints
    return math.sqrt((p1[0] - p0[0]) ** 2 + (p1[1] - p0[1]) ** 2)


def _get_edge_target_segments(part, edge_obj, mesh_size):
    seed_number = _get_edge_seed_value(part, edge_obj, "NUMBER")
    if seed_number is not None:
        try:
            seed_number = int(seed_number)
        except Exception:
            seed_number = None
    if seed_number is not None and seed_number > 0:
        return int(seed_number)

    length = _get_edge_length(part, edge_obj)
    avg_size = _get_edge_seed_value(part, edge_obj, "AVERAGE_SIZE")
    try:
        avg_size = float(avg_size)
    except Exception:
        avg_size = None
    if avg_size is not None and avg_size > 0.0:
        return max(1, int(round(length / avg_size)))
    if mesh_size is not None and float(mesh_size) > 0.0 and length > 0.0:
        return max(1, int(round(length / float(mesh_size))))
    return None


def _detect_edge_seed_mismatches(part, mesh_size):
    mismatches = []
    line_tol_factor = 1.0e-4
    for edge_obj in part.edges:
        edge_idx = _edge_index(edge_obj)
        endpoints = _edge_endpoints(part, edge_obj)
        if endpoints is None:
            continue
        p0, p1 = endpoints
        length = math.sqrt((p1[0] - p0[0]) ** 2 + (p1[1] - p0[1]) ** 2)
        if length <= 1.0e-12:
            continue
        try:
            raw_nodes = edge_obj.getNodes()
        except Exception:
            raw_nodes = []
        edge_nodes = []
        seen = set()
        for nd in raw_nodes:
            lab = int(nd.label)
            if lab in seen:
                continue
            seen.add(lab)
            x = float(nd.coordinates[0])
            y = float(nd.coordinates[1])
            tt, off = _project_param_and_offline(x, y, p0, p1)
            edge_nodes.append((lab, x, y, tt, off))
        if len(edge_nodes) < 2:
            continue
        max_off = max(row[4] for row in edge_nodes)
        if max_off > max(line_tol_factor * length, 1.0e-8):
            continue
        actual_segments = len(edge_nodes) - 1
        expected_segments = None
        seed_mode = "UNKNOWN"
        seed_number = _get_edge_seed_value(part, edge_obj, "NUMBER")
        if seed_number is not None:
            try:
                seed_number = int(seed_number)
            except Exception:
                seed_number = None
        if seed_number is not None and seed_number > 0:
            expected_segments = int(seed_number)
            seed_mode = "NUMBER"
        else:
            avg_size = _get_edge_seed_value(part, edge_obj, "AVERAGE_SIZE")
            try:
                avg_size = float(avg_size)
            except Exception:
                avg_size = None
            if avg_size is not None and avg_size > 0.0:
                expected_segments = max(1, int(round(length / avg_size)))
                seed_mode = "SIZE"
        if expected_segments is None:
            continue
        if actual_segments != expected_segments:
            mismatches.append(
                {
                    "edge_index": edge_idx,
                    "edge_obj": edge_obj,
                    "seed_mode": seed_mode,
                    "expected_segments": int(expected_segments),
                    "actual_segments": int(actual_segments),
                    "length": float(length),
                }
            )
    return mismatches


def _collect_outer_boundary_edges(part):
    if len(part.vertices) == 0:
        return {"left": [], "right": [], "bottom": [], "top": [], "tol": min_boundary_tol}
    xvals = [float(v.pointOn[0][0]) for v in part.vertices]
    yvals = [float(v.pointOn[0][1]) for v in part.vertices]
    xmin = min(xvals)
    xmax = max(xvals)
    ymin = min(yvals)
    ymax = max(yvals)
    Lx = xmax - xmin
    Ly = ymax - ymin
    tol = max(boundary_tol_factor * max(Lx, Ly), min_boundary_tol)

    out = {"left": [], "right": [], "bottom": [], "top": [], "tol": tol}
    seen = {"left": set(), "right": set(), "bottom": set(), "top": set()}
    for edge_obj in part.edges:
        edge_idx = _edge_index(edge_obj)
        endpoints = _edge_endpoints(part, edge_obj)
        if endpoints is None:
            continue
        p0, p1 = endpoints
        if _is_close(p0[0], xmin, tol) and _is_close(p1[0], xmin, tol):
            if edge_idx not in seen["left"]:
                out["left"].append(edge_obj)
                seen["left"].add(edge_idx)
        if _is_close(p0[0], xmax, tol) and _is_close(p1[0], xmax, tol):
            if edge_idx not in seen["right"]:
                out["right"].append(edge_obj)
                seen["right"].add(edge_idx)
        if _is_close(p0[1], ymin, tol) and _is_close(p1[1], ymin, tol):
            if edge_idx not in seen["bottom"]:
                out["bottom"].append(edge_obj)
                seen["bottom"].add(edge_idx)
        if _is_close(p0[1], ymax, tol) and _is_close(p1[1], ymax, tol):
            if edge_idx not in seen["top"]:
                out["top"].append(edge_obj)
                seen["top"].add(edge_idx)
    return out


def _edge_midpoint(part, edge_obj):
    endpoints = _edge_endpoints(part, edge_obj)
    if endpoints is None:
        return None
    p0, p1 = endpoints
    return (0.5 * (p0[0] + p1[0]), 0.5 * (p0[1] + p1[1]))


def _match_periodic_boundary_edge_pairs(part, src_edges, dst_edges, axis, period, tol):
    def _edge_interval(edge_obj, axis_name):
        endpoints = _edge_endpoints(part, edge_obj)
        if endpoints is None:
            return None
        p0, p1 = endpoints
        if axis_name == "x":
            a0 = float(min(p0[1], p1[1]))
            a1 = float(max(p0[1], p1[1]))
        else:
            a0 = float(min(p0[0], p1[0]))
            a1 = float(max(p0[0], p1[0]))
        return (a0, a1)

    def _interval_overlap(int_a, int_b):
        if int_a is None or int_b is None:
            return -1.0
        lo = max(float(int_a[0]), float(int_b[0]))
        hi = min(float(int_a[1]), float(int_b[1]))
        return float(hi - lo)

    pairs = []
    used = set()
    for src_edge in src_edges:
        src_mid = _edge_midpoint(part, src_edge)
        if src_mid is None:
            continue
        if axis == "x":
            target_pt = (src_mid[0] + period, src_mid[1])
        else:
            target_pt = (src_mid[0], src_mid[1] + period)

        best = None
        best_dist = None
        for dst_edge in dst_edges:
            dst_idx = _edge_index(dst_edge)
            if dst_idx in used:
                continue
            endpoints = _edge_endpoints(part, dst_edge)
            if endpoints is None:
                continue
            dist = _point_segment_distance(target_pt[0], target_pt[1], endpoints[0], endpoints[1])
            src_int = _edge_interval(src_edge, axis)
            dst_int = _edge_interval(dst_edge, axis)
            overlap = _interval_overlap(src_int, dst_int)
            score = (1 if overlap >= -tol else 0, overlap, -dist)
            best_score = None
            if best is not None:
                best_endpoints = _edge_endpoints(part, best)
                best_int = _edge_interval(best, axis)
                best_overlap = _interval_overlap(src_int, best_int)
                best_dist_now = _point_segment_distance(target_pt[0], target_pt[1], best_endpoints[0], best_endpoints[1])
                best_score = (1 if best_overlap >= -tol else 0, best_overlap, -best_dist_now)
            if best is None or score > best_score:
                best = dst_edge
                best_dist = dist
        if best is None:
            continue
        if best_dist is None or best_dist > tol:
            continue
        used.add(_edge_index(best))
        pairs.append((src_edge, best, float(best_dist)))
    return pairs


def _repair_edge_seed_mismatches(part, mesh_size, report_prefix):
    repaired = []
    for rec in _detect_edge_seed_mismatches(part, mesh_size):
        edge_obj = rec["edge_obj"]
        nseg = int(rec["expected_segments"])
        if nseg <= 0:
            continue
        part.seedEdgeByNumber(edges=(edge_obj,), number=nseg, constraint=FIXED)
        repaired.append((int(rec["edge_index"]), int(rec["actual_segments"]), int(nseg)))
    if len(repaired) > 0:
        print("[repair] %s reseed mismatched edges: %s" % (
            report_prefix,
            ", ".join(["edge=%d actual=%d expected=%d" % row for row in repaired]),
        ))
    return repaired


def _repair_failed_boundary_edges(part, mesh_size, report_prefix, check_result):
    boundary = _collect_outer_boundary_edges(part)
    if len(part.vertices) == 0:
        return []
    xvals = [float(v.pointOn[0][0]) for v in part.vertices]
    yvals = [float(v.pointOn[0][1]) for v in part.vertices]
    xmin = min(xvals)
    xmax = max(xvals)
    ymin = min(yvals)
    ymax = max(yvals)
    tol = max(5.0 * float(boundary["tol"]), 0.25 * float(mesh_size))

    repairs = []
    pair_jobs = []
    if not check_result["tb_ok"]:
        pair_jobs.append(("bottom", "top", "y", ymax - ymin))
    if not check_result["lr_ok"]:
        pair_jobs.append(("left", "right", "x", xmax - xmin))

    for src_name, dst_name, axis, period in pair_jobs:
        src_edges = list(boundary[src_name])
        dst_edges = list(boundary[dst_name])
        pairs = _match_periodic_boundary_edge_pairs(part, src_edges, dst_edges, axis, period, tol)
        pair_map = {}
        for src_edge, dst_edge, dist in pairs:
            src_nseg = _get_edge_target_segments(part, src_edge, mesh_size)
            dst_nseg = _get_edge_target_segments(part, dst_edge, mesh_size)
            if src_nseg is None or src_nseg <= 0:
                try:
                    src_nseg = max(1, len(src_edge.getNodes()) - 1)
                except Exception:
                    src_nseg = None
            if dst_nseg is None or dst_nseg <= 0:
                try:
                    dst_nseg = max(1, len(dst_edge.getNodes()) - 1)
                except Exception:
                    dst_nseg = None
            pair_nseg = max(int(src_nseg or 0), int(dst_nseg or 0))
            if pair_nseg <= 0:
                continue
            part.seedEdgeByNumber(edges=(src_edge,), number=pair_nseg, constraint=FIXED)
            part.seedEdgeByNumber(edges=(dst_edge,), number=pair_nseg, constraint=FIXED)
            src_idx = _edge_index(src_edge)
            dst_idx = _edge_index(dst_edge)
            pair_map[src_idx] = (dst_idx, dist, pair_nseg)
            pair_map[dst_idx] = (src_idx, dist, pair_nseg)
            repairs.append(
                {
                    "edge_index": int(src_idx),
                    "edge_name": str(src_name),
                    "segments": int(pair_nseg),
                    "paired_edge_index": int(dst_idx),
                    "pair_distance": float(dist),
                }
            )
            repairs.append(
                {
                    "edge_index": int(dst_idx),
                    "edge_name": str(dst_name),
                    "segments": int(pair_nseg),
                    "paired_edge_index": int(src_idx),
                    "pair_distance": float(dist),
                }
            )

        paired_ids = set(pair_map.keys())
        for edge_group_name, edge_group in ((src_name, src_edges), (dst_name, dst_edges)):
            for edge_obj in edge_group:
                edge_idx = _edge_index(edge_obj)
                if edge_idx in paired_ids:
                    continue
                nseg = _get_edge_target_segments(part, edge_obj, mesh_size)
                if nseg is None or nseg <= 0:
                    try:
                        nseg = max(1, len(edge_obj.getNodes()) - 1)
                    except Exception:
                        nseg = None
                if nseg is None or nseg <= 0:
                    continue
                part.seedEdgeByNumber(edges=(edge_obj,), number=int(nseg), constraint=FIXED)
                repairs.append(
                    {
                        "edge_index": int(edge_idx),
                        "edge_name": str(edge_group_name),
                        "segments": int(nseg),
                        "paired_edge_index": "",
                        "pair_distance": "",
                    }
                )
    if len(repairs) > 0:
        print("[boundary-repair] %s forced boundary edges=%d" % (report_prefix, len(repairs)))
    return repairs


def _check_periodic_boundary_nodes(part, report_prefix):
    xvals = [float(node.coordinates[0]) for node in part.nodes]
    yvals = [float(node.coordinates[1]) for node in part.nodes]
    xmin = min(xvals)
    xmax = max(xvals)
    ymin = min(yvals)
    ymax = max(yvals)
    Lx = xmax - xmin
    Ly = ymax - ymin
    tol = max(boundary_tol_factor * max(Lx, Ly), min_boundary_tol)

    top_nodes = []
    bottom_nodes = []
    left_nodes = []
    right_nodes = []
    corner_lb = []
    corner_rb = []
    corner_rt = []
    corner_lt = []
    for node in part.nodes:
        x = float(node.coordinates[0])
        y = float(node.coordinates[1])
        lab = int(node.label)
        on_left = _is_close(x, xmin, tol)
        on_right = _is_close(x, xmax, tol)
        on_bottom = _is_close(y, ymin, tol)
        on_top = _is_close(y, ymax, tol)
        if on_left and on_bottom:
            corner_lb.append((lab, x, y))
            continue
        if on_right and on_bottom:
            corner_rb.append((lab, x, y))
            continue
        if on_right and on_top:
            corner_rt.append((lab, x, y))
            continue
        if on_left and on_top:
            corner_lt.append((lab, x, y))
            continue
        if on_bottom:
            bottom_nodes.append((lab, x, y))
        if on_top:
            top_nodes.append((lab, x, y))
        if on_left:
            left_nodes.append((lab, x, y))
        if on_right:
            right_nodes.append((lab, x, y))

    tb_ok = len(top_nodes) == len(bottom_nodes)
    lr_ok = len(left_nodes) == len(right_nodes)
    corners_ok = (
        len(corner_lb) == 1 and len(corner_rb) == 1 and
        len(corner_rt) == 1 and len(corner_lt) == 1
    )
    overall_ok = tb_ok and lr_ok and corners_ok

    summary_csv = os.path.join(out_dir, report_prefix + "_self_check_summary.csv")
    with open(summary_csv, "w") as fp:
        wr = csv.writer(fp)
        wr.writerow(["item", "value", "expected", "status"])
        wr.writerow(["top_node_count", len(top_nodes), len(bottom_nodes), "OK" if tb_ok else "FAIL"])
        wr.writerow(["bottom_node_count", len(bottom_nodes), len(top_nodes), "OK" if tb_ok else "FAIL"])
        wr.writerow(["left_node_count", len(left_nodes), len(right_nodes), "OK" if lr_ok else "FAIL"])
        wr.writerow(["right_node_count", len(right_nodes), len(left_nodes), "OK" if lr_ok else "FAIL"])
        wr.writerow(["corner_LB_count", len(corner_lb), 1, "OK" if len(corner_lb) == 1 else "FAIL"])
        wr.writerow(["corner_RB_count", len(corner_rb), 1, "OK" if len(corner_rb) == 1 else "FAIL"])
        wr.writerow(["corner_RT_count", len(corner_rt), 1, "OK" if len(corner_rt) == 1 else "FAIL"])
        wr.writerow(["corner_LT_count", len(corner_lt), 1, "OK" if len(corner_lt) == 1 else "FAIL"])
        wr.writerow(["overall_periodic_mesh_check", "PASS" if overall_ok else "FAIL", "PASS", "OK" if overall_ok else "FAIL"])
    return {
        "ok": bool(overall_ok),
        "tb_ok": bool(tb_ok),
        "lr_ok": bool(lr_ok),
        "corners_ok": bool(corners_ok),
        "summary_csv": summary_csv,
        "top_count": len(top_nodes),
        "bottom_count": len(bottom_nodes),
        "left_count": len(left_nodes),
        "right_count": len(right_nodes),
    }


def _mesh_with_periodic_check_and_repair(part, mesh_size, report_prefix):
    _mesh_merged_part(part, mesh_size)
    if not periodic_mesh_self_check:
        return {"ok": True, "summary_csv": ""}
    repair_csv = os.path.join(out_dir, report_prefix + "_edge_seed_repair_log.csv")
    with open(repair_csv, "w") as fp:
        wr = csv.writer(fp)
        wr.writerow([
            "iteration", "repair_type", "edge_index", "actual_segments_before",
            "expected_segments", "boundary_name", "paired_edge_index", "pair_distance"
        ])
        for it in range(int(max(1, periodic_mesh_repair_iterations))):
            check = _check_periodic_boundary_nodes(part, report_prefix)
            mismatches = _detect_edge_seed_mismatches(part, mesh_size)
            boundary_repairs = []
            if not check["ok"]:
                boundary_repairs = _repair_failed_boundary_edges(part, mesh_size, report_prefix, check)
            if check["ok"] and len(mismatches) == 0 and len(boundary_repairs) == 0:
                print("[periodic-check] %s PASS | top=%d bottom=%d left=%d right=%d" % (
                    report_prefix, check["top_count"], check["bottom_count"], check["left_count"], check["right_count"]))
                return check
            if len(mismatches) == 0 and len(boundary_repairs) == 0:
                print("[periodic-check] %s FAIL but no seed/node mismatched edge found | summary=%s" % (
                    report_prefix, check["summary_csv"]))
                return check
            for rec in mismatches:
                wr.writerow([it + 1, "seed_mismatch", rec["edge_index"], rec["actual_segments"], rec["expected_segments"], "", "", ""])
            repaired = _repair_edge_seed_mismatches(part, mesh_size, report_prefix)
            for rec in boundary_repairs:
                wr.writerow([
                    it + 1, "boundary_force", rec["edge_index"], "", rec["segments"],
                    rec["edge_name"], rec["paired_edge_index"], rec["pair_distance"]
                ])
            if len(repaired) == 0 and len(boundary_repairs) == 0:
                return check
            part.deleteMesh(regions=part.faces[:])
            part.generateMesh()
        final_check = _check_periodic_boundary_nodes(part, report_prefix)
        print("[periodic-check] %s final status=%s | summary=%s" % (
            report_prefix, "PASS" if final_check["ok"] else "FAIL", final_check["summary_csv"]))
        return final_check


def _case_output_paths(report_prefix):
    return {
        "summary_csv": os.path.join(out_dir, report_prefix + "_self_check_summary.csv"),
        "repair_csv": os.path.join(out_dir, report_prefix + "_edge_seed_repair_log.csv"),
        "cae": os.path.join(out_dir, report_prefix + ".cae"),
    }


def _delete_case_outputs(report_prefix):
    if not report_prefix:
        return
    for path in _case_output_paths(report_prefix).values():
        if path and os.path.isfile(path):
            try:
                os.remove(path)
            except Exception as exc:
                print("[cleanup] failed to delete %s: %s" % (path, str(exc)))


def _make_one_case(shape_name, src_rve_id, out_rve_id):
    report_prefix = None
    src_cae = _resolve_src_cae(shape_name, src_rve_id)
    if src_cae is None:
        print("[skip] source CAE missing | shape=%s src_rve=%02d" % (shape_name, int(src_rve_id)))
        return {
            "shape": str(shape_name),
            "source_rve_id": int(src_rve_id),
            "rve_id": int(out_rve_id),
            "report_prefix": "",
            "check_ok": False,
            "summary_csv": "",
            "top_count": 0,
            "bottom_count": 0,
            "left_count": 0,
            "right_count": 0,
            "fail_reason": "source CAE missing",
        }

    try:
        openMdb(pathName=src_cae)
        session.journalOptions.setValues(replayGeometry=INDEX, recoverGeometry=INDEX)
        model = mdb.models["Model-1"]

        if "Part-1" not in model.parts.keys():
            raise RuntimeError("Part-1 (matrix) not found in %s" % src_cae)
        if "Part-2" not in model.parts.keys():
            raise RuntimeError("Part-2 (fiber) not found in %s" % src_cae)

        a = model.rootAssembly
        a.deleteAllFeatures()
        a.Instance(name="Matrix-1", part=model.parts["Part-1"], dependent=OFF)
        a.Instance(name="Fiber-1", part=model.parts["Part-2"], dependent=OFF)
        a.InstanceFromBooleanMerge(
            name="Part-Manual",
            instances=(a.instances["Matrix-1"], a.instances["Fiber-1"]),
            keepIntersections=ON,
            originalInstances=DELETE,
            domain=GEOMETRY
        )

        if "Part-Manual" not in model.parts.keys():
            raise RuntimeError("Boolean merge failed for shape=%s src_rve=%02d" % (shape_name, int(src_rve_id)))

        merged_part = model.parts["Part-Manual"]
        if len(merged_part.faces) < 2:
            raise RuntimeError(
                "Merged geometry has %d faces only; expected matrix + fiber regions." % len(merged_part.faces)
            )

        _build_face_sets_on_merged_part(model, merged_part, "Part-1")
        xvals = [float(v.pointOn[0][0]) for v in merged_part.vertices]
        yvals = [float(v.pointOn[0][1]) for v in merged_part.vertices]
        if len(xvals) == 0 or len(yvals) == 0:
            raise RuntimeError("Merged part has no geometry vertices for shape=%s src_rve=%02d" % (shape_name, int(src_rve_id)))
        local_L = max(max(xvals) - min(xvals), max(yvals) - min(yvals))
        mesh_size = float(local_L) / float(num_ele)
        report_prefix = "%s_rve_%02d_n%d_h%s_concave_periodic_mesh" % (
            shape_name,
            int(out_rve_id),
            int(target_n),
            ("%0.3f" % float(mesh_size)).replace(".", "p"))
        check_result = _mesh_with_periodic_check_and_repair(merged_part, mesh_size, report_prefix)
        if check_result is None or not bool(check_result.get("ok", False)):
            print("[drop] periodic mesh check failed | shape=%s src_rve=%02d out_rve=%02d" % (
                shape_name, int(src_rve_id), int(out_rve_id)))
            _delete_case_outputs(report_prefix)
            return {
                "shape": str(shape_name),
                "source_rve_id": int(src_rve_id),
                "rve_id": int(out_rve_id),
                "report_prefix": str(report_prefix),
                "check_ok": False,
                "summary_csv": "",
                "top_count": int(check_result.get("top_count", 0)) if check_result is not None else 0,
                "bottom_count": int(check_result.get("bottom_count", 0)) if check_result is not None else 0,
                "left_count": int(check_result.get("left_count", 0)) if check_result is not None else 0,
                "right_count": int(check_result.get("right_count", 0)) if check_result is not None else 0,
                "fail_reason": "periodic self-check failed",
            }

        a = model.rootAssembly
        a.deleteAllFeatures()
        a.Instance(name="Part-Manual-1", part=merged_part, dependent=ON)

        out_path = os.path.join(out_dir, report_prefix + ".cae")
        mdb.saveAs(pathName=out_path)
        print(
            "[saveAs] %s | merged_faces=%d matrix_faces=%d fiber_faces=%d | nodes=%d elements=%d mshsize=%.12g" % (
                out_path,
                len(merged_part.faces),
                len(merged_part.sets["Matrix-Faces"].faces),
                len(merged_part.sets["Fiber-Faces"].faces),
                len(merged_part.nodes),
                len(merged_part.elements),
                mesh_size,
            )
        )
        return {
            "shape": str(shape_name),
            "source_rve_id": int(src_rve_id),
            "rve_id": int(out_rve_id),
            "report_prefix": str(report_prefix),
            "out_path": str(out_path),
            "check_ok": True,
            "summary_csv": str(check_result.get("summary_csv", "")),
            "top_count": int(check_result.get("top_count", 0)),
            "bottom_count": int(check_result.get("bottom_count", 0)),
            "left_count": int(check_result.get("left_count", 0)),
            "right_count": int(check_result.get("right_count", 0)),
            "fail_reason": "",
        }
    except Exception as exc:
        _delete_case_outputs(report_prefix)
        return {
            "shape": str(shape_name),
            "source_rve_id": int(src_rve_id),
            "rve_id": int(out_rve_id),
            "report_prefix": str(report_prefix or ""),
            "check_ok": False,
            "summary_csv": "",
            "top_count": 0,
            "bottom_count": 0,
            "left_count": 0,
            "right_count": 0,
            "fail_reason": str(exc),
        }


def _write_run_summary_txt(ok_rows, fail_rows):
    txt_path = os.path.join(out_dir, "concave_periodic_mesh_generation_periodic_check_summary.txt")
    with open(txt_path, "w") as fp:
        fp.write("Concave periodic mesh generation periodic check summary\n")
        fp.write("output_dir=%s\n" % out_dir)
        fp.write("case_count=%d\n" % (len(ok_rows) + len(fail_rows)))
        fp.write("pass_count=%d\n" % len(ok_rows))
        fp.write("fail_count=%d\n" % len(fail_rows))
        fp.write("\n")
        fp.write("PASS cases:\n")
        if len(ok_rows) == 0:
            fp.write("none\n")
        else:
            for row in ok_rows:
                fp.write(
                    "- %s | src_rve=%02d | top=%d bottom=%d left=%d right=%d | summary=%s\n" % (
                        row["report_prefix"],
                        int(row.get("source_rve_id", 0)),
                        row["top_count"],
                        row["bottom_count"],
                        row["left_count"],
                        row["right_count"],
                        row["summary_csv"],
                    )
                )
        fp.write("\n")
        fp.write("FAIL cases:\n")
        if len(fail_rows) == 0:
            fp.write("none\n")
        else:
            for row in fail_rows:
                fp.write(
                    "- src_rve=%02d | out_rve=%02d | prefix=%s | reason=%s | top=%d bottom=%d left=%d right=%d\n" % (
                        int(row.get("source_rve_id", 0)),
                        int(row.get("rve_id", 0)),
                        row["report_prefix"],
                        row.get("fail_reason", ""),
                        row["top_count"],
                        row["bottom_count"],
                        row["left_count"],
                        row["right_count"],
                    )
                )
    print("[summary-txt] %s" % txt_path)
    return txt_path


def main():
    case_results = []
    fail_results = []
    print("[%s] shape=%s src_rve=%02d out_rve=%02d" % (SCRIPT_TAG, shape_name, int(rve_id), 1))
    result = _make_one_case(shape_name, rve_id, 1)
    if result is not None:
        if bool(result.get("check_ok", False)):
            case_results.append(result)
        else:
            fail_results.append(result)
    summary_path = _write_run_summary_txt(case_results, fail_results)
    print("[%s] wrote %s" % (SCRIPT_TAG, summary_path))


if __name__ == "__main__":
    main()
