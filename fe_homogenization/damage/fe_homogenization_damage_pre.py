# -*- coding: utf-8 -*-
# Single-RVE Abaqus/CAE preprocessing script for damage homogenization.
# Run only after fe_homogenization_damage_periodic_mesh.py.
# It keeps the recorded cohesive/PBC workflow but only for one shape, one RVE,
# and one load case.

import csv
import math
import os
import glob
import re
import numpy as np

from abaqus import *
from caeModules import *
from abaqusConstants import *
from driverUtils import executeOnCaeStartup
import mesh
#
executeOnCaeStartup()
Mdb()
#
#-------------------------------------------------------------------------------------
#
session.journalOptions.setValues(replayGeometry=INDEX, recoverGeometry=INDEX)

try:
    script_dir = os.path.dirname(os.path.abspath(__file__))
except NameError:
    script_dir = os.getcwd()
os.chdir(script_dir)
#
#-------------------------------------------------------------------------------------
# 1. User parameters
#-------------------------------------------------------------------------------------
#
SCRIPT_TAG = 'fe_homogenization_damage_pre'
load_case_name = os.environ.get('FE_HOMOG_DAMAGE_LOAD_CASE', 'tension').strip().lower()
shape_name_single = os.environ.get('FE_HOMOG_DAMAGE_SHAPE', 'lobular2').strip()
rve_id_single = int(os.environ.get('FE_HOMOG_DAMAGE_RVE_ID', '1'))
#
nfiber = 60
fiber_radius = 5.0
fiber_vf = 0.5
L = math.sqrt(nfiber * math.pi * fiber_radius * fiber_radius / fiber_vf)
num_ele = 60
mshsize = L / num_ele

center_dir = os.environ.get('FE_HOMOG_DAMAGE_CENTER_DIR', 'intermediate_final_files')
workdir = f'concave_rve_cae_jobs_{int(num_ele)}'
workdir = os.environ.get('FE_HOMOG_DAMAGE_WORKDIR', workdir)
periodic_cae_dir = os.environ.get('FE_HOMOG_DAMAGE_PERIODIC_DIR', f'concave_periodic_mesh_cae_{int(num_ele)}')
matrix_size = L
#
save_cae = True
write_input = True
compression_direct_inp_patch_only = True
#
num_cpus = 4
job_explicit_precision = DOUBLE
job_nodal_output_precision = FULL
#
# This script only creates CAE/INP files. It does not submit jobs.
#
#-------------------------------------------------------------------------------------
# 2. Yang2012 material and model parameters, MPa-um-s unit system
#-------------------------------------------------------------------------------------
#
fiber_E = 40000.0
fiber_nu = 0.25
#
matrix_E = 4000.0
matrix_nu = 0.35
dp_beta = 24.0
dp_k = 0.8
dp_dilation = 24.0
matrix_tension_yield = 60.0
matrix_compression_yield = 101.9
matrix_eps_damage_tension = 0.05
matrix_eps_damage_compression = 0.50
matrix_Gf = 0.5
#
interface_K = 1.0E5
interface_strength = 39.1
interface_G = 100.0
# Delete cohesive elements when SDEG reaches this maximum degradation.
cohesive_delete_at_sdeg = 0.95
#
density = 1.0E-15
thickness = 1.0
#
step_time = 1.0
history_dt = 1.0E-3
#
tension_strain = 0.016
compression_strain = -0.05
#
mass_scaling_method = 'VARIABLE'
use_variable_mass_scaling = True
mass_scaling_dt = 1.0E-5
mass_scaling_frequency = 1
#
keep_matrix_elements_after_damage = True
#
exclude_cohesive_interface_nodes_from_pbc = True
pair_all_periodic_boundary_nodes_together = True
include_fiber_side_duplicate_nodes_in_pbc = False
# Nearest-coordinate tolerance scale for periodic boundary node pairing.
# Concave-shape periodic meshes may have small geometric asymmetry at boundaries.
pbc_pair_tol_factor = 200.0
# Minimum nearest-pair tolerance as a fraction of current mesh size.
pbc_pair_tol_mesh_fraction = 0.08
#
#-------------------------------------------------------------------------------------
# 3. Working directories
#-------------------------------------------------------------------------------------
#
if not os.path.isdir(workdir):
    os.makedirs(workdir)
#
root_dir = os.getcwd()
center_dir_abs = os.path.abspath(center_dir)
workdir_abs = os.path.abspath(workdir)
periodic_cae_dir_abs = os.path.abspath(periodic_cae_dir)
#
print('[%s] single-case damage preprocessing script' % SCRIPT_TAG)
print('[info] fiber positions are read from existing CSV files only')
print('[info] no random packing, no overlap removal, no center-file writing, no job submission')
print('[info] PBC uses reference-point equations')
print('[info] mass scaling method = %s, target dt = %.12g' % (mass_scaling_method, mass_scaling_dt))
print('[info] job precision = DOUBLE, nodal output precision = FULL')
print('[info] matrix element deletion = OFF')
print('[info] cohesive element deletion = ON at SDEG/maxDegradation = %.3f' % cohesive_delete_at_sdeg)
print('[info] RVE side length = %.12g um' % L)
print('[info] mesh size = %.12g um' % mshsize)
print('[info] center_dir = %s' % center_dir_abs)
print('[info] workdir = %s' % workdir_abs)
print('[info] periodic_cae_dir = %s' % periodic_cae_dir_abs)
print('[info] periodic-inclusion matrix_size = %.12g um' % matrix_size)
#
os.chdir(workdir_abs)
#

def _resolve_position_center_file(shape_name, rve_id):
    explicit_path = os.environ.get('FE_HOMOG_DAMAGE_CENTER_FILE', '').strip()
    if explicit_path:
        return os.path.abspath(explicit_path)
    cands = [
        os.path.join(center_dir_abs, '%s_rve_%02d_final.txt' % (shape_name, int(rve_id))),
        os.path.join(center_dir_abs, '%s_rve_%d_final.txt' % (shape_name, int(rve_id))),
    ]
    for c in cands:
        if os.path.isfile(c):
            return c
    return cands[0]


def _resolve_periodic_mesh_cae(shape_name, rve_id):
    explicit_path = os.environ.get('FE_HOMOG_DAMAGE_PERIODIC_CAE', '').strip()
    if explicit_path:
        return os.path.abspath(explicit_path)
    pat1 = os.path.join(periodic_cae_dir_abs, '%s_rve_%02d_h*_concave_periodic_mesh.cae' % (shape_name, int(rve_id)))
    pat2 = os.path.join(periodic_cae_dir_abs, '%s_rve_%d_h*_concave_periodic_mesh.cae' % (shape_name, int(rve_id)))
    hits = sorted(glob.glob(pat1))
    if len(hits) == 0:
        hits = sorted(glob.glob(pat2))
    if len(hits) == 0:
        return None
    return hits[-1]


def _mesh_size_from_periodic_cae_name(periodic_cae_path):
    """
    Parse mesh size tag from file name:
      ..._h1p038_concave_periodic_mesh.cae -> 1.038
    """
    if periodic_cae_path is None:
        return None
    bname = os.path.basename(str(periodic_cae_path))
    m = re.search(r"_h([0-9p]+)_concave_periodic_mesh\.cae$", bname)
    if not m:
        return None
    try:
        return float(m.group(1).replace('p', '.'))
    except Exception:
        return None


def _periodic_mesh_self_check_status(periodic_cae_path):
    base, _ = os.path.splitext(periodic_cae_path)
    summary_path = base + '_self_check_summary.csv'
    if not os.path.isfile(summary_path):
        legacy_summary_path = base.replace('_concave_periodic_mesh', '_concave_periodic_mesh_self_check_summary') + '.csv'
        if os.path.isfile(legacy_summary_path):
            summary_path = legacy_summary_path
        else:
            return 'MISSING', summary_path
    try:
        with open(summary_path, 'r') as fp:
            rd = csv.reader(fp)
            for row in rd:
                if len(row) >= 2 and str(row[0]).strip() == 'overall_periodic_mesh_check':
                    return str(row[1]).strip().upper(), summary_path
    except Exception:
        pass
    return 'UNKNOWN', summary_path

#-------------------------------------------------------------------------------------
# 3.5 Direct INP patch helper for Drucker-Prager hardening
#-------------------------------------------------------------------------------------
#
def _patch_drucker_prager_hardening_inp(inp_path, load_case):

    if not os.path.isfile(inp_path):
        raise RuntimeError('INP file does not exist and cannot be patched directly: %s' % inp_path)

    with open(inp_path, 'r') as fp:
        lines = fp.readlines()

    if str(load_case).lower() == 'compression':
        dp_type = 'COMPRESSION'
        yield_value = matrix_compression_yield
    else:
        dp_type = 'TENSION'
        yield_value = matrix_tension_yield

    patched = []
    changed = 0
    i = 0
    while i < len(lines):
        line = lines[i]
        low = line.strip().lower()

        if low.startswith('*drucker prager hardening'):
            patched.append('*Drucker Prager Hardening, type=%s\n' % dp_type)
            changed += 1
            i += 1

            # Preserve blank lines immediately after the keyword card, if any.
            while i < len(lines) and lines[i].strip() == '':
                patched.append(lines[i])
                i += 1

            # Replace the first data row after the card.
            if i < len(lines) and not lines[i].lstrip().startswith('*'):
                patched.append('%.12g, 0.\n' % yield_value)
                i += 1
            else:
                patched.append('%.12g, 0.\n' % yield_value)
            continue

        patched.append(line)
        i += 1

    if changed == 0:
        raise RuntimeError('Cannot find *Drucker Prager Hardening card in %s' % inp_path)

    backup_path = inp_path + '.bak_dp_hardening'
    if not os.path.isfile(backup_path):
        with open(backup_path, 'w') as fp:
            fp.writelines(lines)

    with open(inp_path, 'w') as fp:
        fp.writelines(patched)

    print('[inp direct patch] %s' % inp_path)
    print('[inp direct patch] *Drucker Prager Hardening, type=%s' % dp_type)
    print('[inp direct patch] yield = %.12g MPa | cards patched = %d' % (yield_value, changed))
    print('[inp direct patch] backup = %s' % backup_path)

#
def _patch_reference_point_loading_values_inp(inp_path, source_disp, target_disp):

    if not os.path.isfile(inp_path):
        raise RuntimeError('INP file does not exist and cannot be patched directly: %s' % inp_path)

    target_signed = float(target_disp)
    src_abs = abs(float(source_disp))
    tgt_abs = abs(target_signed)
    tol = max(1.0E-9, 1.0E-6 * max(src_abs, tgt_abs, 1.0))

    with open(inp_path, 'r') as fp:
        lines = fp.readlines()

    rp_tokens = (
        'RP-1', 'RP-2', 'RP1', 'RP2',
        'SET-RP-1', 'SET-RP-2',
        'RP-RL', 'RP-TB',
        'SET-RP-RL', 'SET-RP-TB'
    )

    patched = []
    in_boundary = False
    changed = 0
    changed_rows = []

    for line in lines:
        stripped = line.strip()
        low = stripped.lower()

        if low.startswith('*boundary'):
            in_boundary = True
            patched.append(line)
            continue

        if in_boundary and stripped.startswith('*'):
            in_boundary = False
            patched.append(line)
            continue

        if in_boundary and stripped and not stripped.startswith('**'):
            parts = [p.strip() for p in line.split(',')]
            if len(parts) >= 4:
                set_name_upper = parts[0].upper()
                is_rp_line = False
                for tok in rp_tokens:
                    if tok in set_name_upper:
                        is_rp_line = True
                        break

                if is_rp_line:
                    try:
                        old_value = float(parts[3])
                    except Exception:
                        old_value = 0.0

                    # Keep exact zero constraints unchanged.  Only actual loading
                    # values are converted from tension to compression.
                    if abs(old_value) > 1.0E-30:
                        # Idempotent patch: force exact signed target value
                        # (compression should stay negative, tension positive).
                        new_value = target_signed
                        parts[3] = '%.12g' % new_value
                        patched.append(', '.join(parts) + '\n')
                        changed += 1
                        old_abs = abs(old_value)
                        if abs(old_abs - src_abs) <= tol:
                            note = 'from_tension_value'
                        elif abs(old_abs - tgt_abs) <= tol:
                            note = 'already_target_value'
                        else:
                            note = 'unexpected_old_magnitude_reset_to_target'
                        changed_rows.append((parts[0], parts[1], parts[2], old_value, new_value, note))
                        continue

        patched.append(line)

    if changed == 0:
        raise RuntimeError(
            'Cannot find non-zero RP-1/RP-2/RP-RL/RP-TB loading boundary values in %s' % inp_path
        )

    with open(inp_path, 'w') as fp:
        fp.writelines(patched)

    print('[inp direct patch] reference-point loading values reset to target magnitude')
    print('[inp direct patch] source tensile displacement (reference) = %.12g um' % float(source_disp))
    print('[inp direct patch] target compression displacement = %.12g um' % float(target_disp))
    for row in changed_rows:
        print('[inp direct patch] %s, dof %s-%s: %.12g -> %.12g | %s' % row)
    print('[inp direct patch] RP loading rows patched = %d' % changed)


def _make_compression_inp_from_tension_inp(tension_inp_path, compression_inp_path,
                                           tension_jobname, compression_jobname,
                                           source_disp, target_disp):

    if not os.path.isfile(tension_inp_path):
        raise RuntimeError(
            'Compression INP does not exist and the corresponding tension INP was not found either. '
            'Run the tension preprocessing once first. Missing tension INP: %s'
            % tension_inp_path
        )

    with open(tension_inp_path, 'r') as fp:
        data = fp.read()

    data = data.replace(tension_jobname, compression_jobname)
    data = data.replace(tension_jobname.upper(), compression_jobname.upper())
    data = data.replace(tension_jobname.lower(), compression_jobname.lower())

    with open(compression_inp_path, 'w') as fp:
        fp.write(data)

    print('[inp direct copy] %s -> %s' % (tension_inp_path, compression_inp_path))
    _patch_reference_point_loading_values_inp(compression_inp_path, source_disp, target_disp)
    _patch_drucker_prager_hardening_inp(compression_inp_path, 'compression')



#-------------------------------------------------------------------------------------
# 4. Main preprocessing loop
#-------------------------------------------------------------------------------------
#
for shape_name, load_case, seed in [
    (shape_name_single, load_case_name, rve_id_single)
]:
        #
        #---------------------------------------------------------------------------------
        # 4.1 Case information
        #---------------------------------------------------------------------------------
        #
        if load_case == 'tension':
            applied_strain = tension_strain
            hardening_type = TENSION
            hardening_yield = matrix_tension_yield
        else:
            applied_strain = compression_strain
            hardening_type = COMPRESSION
            hardening_yield = matrix_compression_yield
        #
        periodic_cae_for_tag = _resolve_periodic_mesh_cae(shape_name, seed)
        mshsize_for_name = _mesh_size_from_periodic_cae_name(periodic_cae_for_tag)
        if mshsize_for_name is not None:
            mshsize = float(mshsize_for_name)
        jobname = '%s_%s_rve%02d_h%0.3fum' % (shape_name, load_case, seed, mshsize)
        jobname = jobname.replace('.', 'p')
        pathname = jobname + '.cae'
        center_file = _resolve_position_center_file(shape_name, seed)
        tension_disp = tension_strain * matrix_size
        compression_disp = compression_strain * matrix_size
        #
        if load_case == 'compression' and compression_direct_inp_patch_only:
            compression_inp_path = os.path.join(workdir_abs, jobname + '.inp')
            tension_jobname = '%s_tension_rve%02d_h%0.3fum' % (shape_name, seed, mshsize)
            tension_jobname = tension_jobname.replace('.', 'p')
            tension_inp_path = os.path.join(workdir_abs, tension_jobname + '.inp')
            print(' ')
            print('#-------------------------------------------------------------------------------------')
            print('[case] %s' % jobname)
            print('[compression direct INP patch only]')
            print('[target compression inp] %s' % compression_inp_path)
            print('[source tension inp]     %s' % tension_inp_path)
            print('#-------------------------------------------------------------------------------------')
            if not os.path.isfile(compression_inp_path):
                _make_compression_inp_from_tension_inp(
                    tension_inp_path,
                    compression_inp_path,
                    tension_jobname,
                    jobname,
                    tension_disp,
                    compression_disp
                )
            else:
                _patch_reference_point_loading_values_inp(
                    compression_inp_path,
                    tension_disp,
                    compression_disp
                )
                _patch_drucker_prager_hardening_inp(compression_inp_path, 'compression')
            print('[skip] compression CAE/cohesive/PBC workflow is not rerun.')
            continue
        #
        print(' ')
        print('#-------------------------------------------------------------------------------------')
        print('[case] %s' % jobname)
        print('[read] %s' % center_file)
        print('#-------------------------------------------------------------------------------------')
        #
        if not os.path.isfile(center_file):
            raise RuntimeError('Fiber-position CSV does not exist: %s' % center_file)
        #
        #---------------------------------------------------------------------------------
        # 4.2 Reset CAE database for this case
        #---------------------------------------------------------------------------------
        #
        Mdb()
        session.journalOptions.setValues(replayGeometry=INDEX, recoverGeometry=INDEX)
        model = mdb.models['Model-1']
        #
        #---------------------------------------------------------------------------------
        # 4.3 Read already-generated fiber positions
        #---------------------------------------------------------------------------------
        #
        with open(center_file, 'r') as fp:
            first_line = fp.readline().strip().lower()
        #
        base_centers = []
        if 'x_um' in first_line or 'fiber' in first_line or 'radius' in first_line:
            with open(center_file, 'r') as fp:
                reader = csv.DictReader(fp)
                rows = []
                for irow, row in enumerate(reader):
                    keys = dict((str(k).strip().lower(), v) for k, v in row.items())
                    fid = int(float(keys.get('fiber_id', keys.get('id', irow + 1))))
                    if 'x_um' in keys:
                        x = float(keys['x_um'])
                    else:
                        x = float(keys['x'])
                    if 'y_um' in keys:
                        y = float(keys['y_um'])
                    else:
                        y = float(keys['y'])
                    rows.append((fid, x, y))
                rows.sort(key=lambda t: t[0])
                for fid, x, y in rows:
                    base_centers.append((x, y))
        else:
            try:
                raw = np.loadtxt(center_file, delimiter=',')
            except Exception:
                raw = np.loadtxt(center_file)
            if len(raw.shape) == 1:
                raw = raw.reshape((1, raw.shape[0]))
            if raw.shape[1] >= 3:
                first_col = raw[:, 0]
                if np.all(np.abs(first_col - np.round(first_col)) < 1.0E-8) and np.all(first_col >= 1) and np.all(first_col <= nfiber):
                    xy = raw[:, 1:3]
                else:
                    xy = raw[:, 0:2]
            else:
                xy = raw[:, 0:2]
            for row in xy:
                base_centers.append((float(row[0]), float(row[1])))
        #
        if len(base_centers) != nfiber:
            raise RuntimeError('Expected %d fiber centers, but read %d from %s' % (nfiber, len(base_centers), center_file))
        #
        # Concave workflow may use an RVE size larger than the historical Yang2012
        # circle-derived L. Infer box size from imported center coordinates when needed.
        x_vals = [float(v[0]) for v in base_centers]
        y_vals = [float(v[1]) for v in base_centers]
        x_min = min(x_vals)
        x_max = max(x_vals)
        y_min = min(y_vals)
        y_max = max(y_vals)
        inferred_L = max(x_max, y_max, x_max - x_min, y_max - y_min)
        if inferred_L > matrix_size + 1.0E-8:
            print('[warn] center extents exceed preset L: old_L=%.12g, inferred_L=%.12g | update matrix_size/L' % (
                matrix_size, inferred_L))
            matrix_size = inferred_L
            L = inferred_L
            mshsize = L / float(num_ele)
            print('[info] updated L=%.12g, mesh size tag basis=%.12g' % (L, mshsize))
        #
        # Periodic-center files may contain coordinates slightly outside [0, L)
        # after wrapping operations. Normalize by periodic modulo before checks.
        wrapped_count = 0
        for i in range(len(base_centers)):
            x0, y0 = base_centers[i]
            x = x0 % matrix_size
            y = y0 % matrix_size
            if abs(x - x0) > 1.0E-10 or abs(y - y0) > 1.0E-10:
                wrapped_count += 1
            base_centers[i] = (x, y)
        if wrapped_count > 0:
            print('[centers] periodic wrap applied to %d centers' % wrapped_count)
        #
        for i in range(len(base_centers)):
            x, y = base_centers[i]
            if x < -1.0E-8 or x > matrix_size + 1.0E-8 or y < -1.0E-8 or y > matrix_size + 1.0E-8:
                raise RuntimeError('Fiber center %d lies outside the RVE: (%.12g, %.12g)' % (i + 1, x, y))
        #
        print('[centers] read %d fixed fiber centers' % (len(base_centers), ))
        #
        #---------------------------------------------------------------------------------
        # 4.4 Create periodic image centers for phase classification
        #---------------------------------------------------------------------------------
        #
        # The mesh CAE has already been generated with real periodic inclusion
        # partitions in the same domain:
        #     matrix_size = RVE_size = L
        #     domain = [0, L] x [0, L]
        #
        matrix_size = L
        domain_xmin = 0.0
        domain_xmax = L
        domain_ymin = 0.0
        domain_ymax = L
        #
        image_centers = []
        image_base_ids = []
        for ibase in range(len(base_centers)):
            x0, y0 = base_centers[ibase]
            for sx in [-1, 0, 1]:
                for sy in [-1, 0, 1]:
                    x = x0 + sx * L
                    y = y0 + sy * L
                    if (x + fiber_radius > domain_xmin - 1.0E-10 and
                        x - fiber_radius < domain_xmax + 1.0E-10 and
                        y + fiber_radius > domain_ymin - 1.0E-10 and
                        y - fiber_radius < domain_ymax + 1.0E-10):
                        image_centers.append((x, y))
                        image_base_ids.append(ibase + 1)
        #
        print('[geometry] periodic image centers = %d | real fibers = %d | matrix_size=%.12g um' % (
            len(image_centers), len(base_centers), matrix_size))
        #
        rs = [domain_xmax, domain_xmin, domain_ymax, domain_ymin]
        disp = applied_strain * matrix_size
        #
        #---------------------------------------------------------------------------------
        # 4.5 Read pre-generated periodic-inclusion mesh CAE
        #---------------------------------------------------------------------------------
        #
        # The periodic base mesh is generated by:
        #     fe_homogenization_damage_periodic_mesh.py
        #
        # This CAE already contains the square matrix domain and periodic inclusion
        # partitions.  This script reads that mesh, duplicates fiber-side interface
        # nodes, inserts COH2D4 cohesive elements, then applies materials, PBC and loads.
        #
        def _mesh_tag_for_file(size):
            return ('%0.3f' % float(size)).replace('.', 'p')
        #
        periodic_cae = _resolve_periodic_mesh_cae(shape_name, seed)
        #
        if periodic_cae is None or (not os.path.isfile(periodic_cae)):
            raise RuntimeError(
                'Periodic-inclusion mesh CAE does not exist: %s. '
                'Run fe_homogenization_damage_periodic_mesh.py first.'
                % str(periodic_cae)
            )
        mesh_status, mesh_summary_csv = _periodic_mesh_self_check_status(periodic_cae)
        if mesh_status != 'PASS':
            raise RuntimeError(
                'Periodic mesh self-check is not PASS for this case: status=%s, summary=%s. '
                'Please regenerate periodic mesh and fix boundary node periodicity first.'
                % (mesh_status, mesh_summary_csv)
            )
        #
        print('[open periodic-inclusion mesh CAE] %s' % periodic_cae)
        openMdb(pathName=periodic_cae)
        session.journalOptions.setValues(replayGeometry=INDEX, recoverGeometry=INDEX)
        model = mdb.models['Model-1']
        #
        if 'Part-Native' in model.parts.keys():
            p = model.parts['Part-Native']
        else:
            p = None
            for _pname in model.parts.keys():
                if len(model.parts[_pname].elements) > 0:
                    p = model.parts[_pname]
                    break
            if p is None:
                raise RuntimeError('No meshed part was found in periodic-inclusion mesh CAE: %s' % periodic_cae)
        #
        if len(p.elements) == 0:
            raise RuntimeError('Periodic-inclusion mesh CAE part has no mesh elements: %s' % periodic_cae)
        #
        print('[periodic mesh] part=%s nodes=%d elements=%d' % (p.name, len(p.nodes), len(p.elements)))
        # Keep matrix size consistent with the imported periodic mesh CAE.
        x_vals = [float(nd.coordinates[0]) for nd in p.nodes]
        y_vals = [float(nd.coordinates[1]) for nd in p.nodes]
        if len(x_vals) > 0 and len(y_vals) > 0:
            domain_xmin = min(x_vals)
            domain_xmax = max(x_vals)
            domain_ymin = min(y_vals)
            domain_ymax = max(y_vals)
            matrix_size = max(domain_xmax - domain_xmin, domain_ymax - domain_ymin)
            rs = [domain_xmax, domain_xmin, domain_ymax, domain_ymin]
            disp = applied_strain * matrix_size
            print('[periodic mesh bbox] xmin=%.12g xmax=%.12g ymin=%.12g ymax=%.12g | matrix_size=%.12g' % (
                domain_xmin, domain_xmax, domain_ymin, domain_ymax, matrix_size))
        #
        #---------------------------------------------------------------------------------
        # 4.7 Classify native elements and collect interface edges
        #---------------------------------------------------------------------------------
        #
        node_coords = {}
        for n in p.nodes:
            node_coords[int(n.label)] = (float(n.coordinates[0]), float(n.coordinates[1]), 0.0)
        #
        # Phase split requested:
        # 1) read one point on matrix part before merge
        # 2) find which merged face contains that point -> matrix face
        # 3) all remaining merged faces -> fiber faces
        phase_probe_eps = max(1.0E-8 * matrix_size, 1.0E-4 * mshsize)
        matrix_probe_point = None
        if 'Part-1' in model.parts.keys():
            try:
                if len(model.parts['Part-1'].faces) > 0:
                    pmat = model.parts['Part-1']
                    pt0 = pmat.faces[0].pointOn[0]
                    matrix_probe_point = (float(pt0[0]), float(pt0[1]), 0.0)
            except Exception:
                matrix_probe_point = None

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
                    return face_array.findAt((q,))
                except Exception:
                    pass
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

        def _face_index(face_obj):
            try:
                return int(face_obj.index)
            except Exception:
                return int(face_obj.index())

        matrix_face_index = None
        if matrix_probe_point is not None:
            mface = _unwrap_face(_find_face_by_point(p.faces, matrix_probe_point[0], matrix_probe_point[1], phase_probe_eps))
            if mface is not None:
                matrix_face_index = _face_index(mface)
                print('[phase] matrix probe point -> merged face index %d' % matrix_face_index)
            else:
                print('[phase] matrix probe point did not hit any merged face; fallback enabled')
        else:
            print('[phase] matrix probe point unavailable; fallback enabled')

        def _element_phase_and_fid(ex, ey):
            # Preferred route: face ownership in merged geometry.
            if matrix_face_index is not None:
                eface = _unwrap_face(_find_face_by_point(p.faces, ex, ey, phase_probe_eps))
                if eface is not None:
                    if _face_index(eface) == matrix_face_index:
                        return 'matrix', -1
                    best_d2 = 1.0E99
                    best_ic = -1
                    for ic in range(len(image_centers)):
                        cx, cy = image_centers[ic]
                        dx = ex - cx
                        dy = ey - cy
                        d2 = dx * dx + dy * dy
                        if d2 < best_d2:
                            best_d2 = d2
                            best_ic = ic
                    return 'fiber', best_ic
            # Fallback: circular distance (for robustness)
            fid = -1
            rr = (fiber_radius * 1.001) ** 2
            for ic in range(len(image_centers)):
                cx, cy = image_centers[ic]
                dx = ex - cx
                dy = ey - cy
                if dx * dx + dy * dy <= rr:
                    fid = ic
                    break
            return ('fiber', fid) if fid >= 0 else ('matrix', -1)

        elements = []
        edge_to_elems = {}
        for elem in p.elements:
            conn = []
            for nd in elem.getNodes():
                conn.append(int(nd.label))
            #
            sx = 0.0
            sy = 0.0
            for lab in conn:
                sx += node_coords[lab][0]
                sy += node_coords[lab][1]
            ex = sx / float(len(conn))
            ey = sy / float(len(conn))
            #
            phase, fid = _element_phase_and_fid(ex, ey)
            rec = {'label': int(elem.label), 'conn': conn, 'phase': phase, 'fiber_id': fid}
            elements.append(rec)
            idx = len(elements) - 1
            #
            if len(conn) == 3:
                elem_edge_list = [(conn[0], conn[1]), (conn[1], conn[2]), (conn[2], conn[0])]
            elif len(conn) == 4:
                elem_edge_list = [(conn[0], conn[1]), (conn[1], conn[2]), (conn[2], conn[3]), (conn[3], conn[0])]
            else:
                raise RuntimeError('Only TRI3/QUAD4 elements are expected.')
            #
            for ea, eb in elem_edge_list:
                key = tuple(sorted((int(ea), int(eb))))
                if key not in edge_to_elems:
                    edge_to_elems[key] = []
                edge_to_elems[key].append(idx)
        #
        interface_edges = []
        fiber_boundary_nodes = {}
        for key in edge_to_elems.keys():
            adj = edge_to_elems[key]
            if len(adj) != 2:
                continue
            #
            e1 = elements[adj[0]]
            e2 = elements[adj[1]]
            if e1['phase'] == e2['phase']:
                continue
            #
            if e1['phase'] == 'fiber':
                fiber_elem = e1
            else:
                fiber_elem = e2
            #
            fid = fiber_elem['fiber_id']
            if fid < 0:
                continue
            #
            a0, b0 = key
            cx, cy = image_centers[fid]
            xa, ya = node_coords[int(a0)][0], node_coords[int(a0)][1]
            xb, yb = node_coords[int(b0)][0], node_coords[int(b0)][1]
            ta = math.atan2(ya - cy, xa - cx)
            tb = math.atan2(yb - cy, xb - cx)
            dang = (tb - ta) % (2.0 * math.pi)
            if dang > math.pi:
                a, b = int(b0), int(a0)
            else:
                a, b = int(a0), int(b0)
            #
            interface_edges.append((fid, a, b))
            if fid not in fiber_boundary_nodes:
                fiber_boundary_nodes[fid] = set()
            fiber_boundary_nodes[fid].add(a)
            fiber_boundary_nodes[fid].add(b)
        #
        print('[mesh] native elements = %d | interface edges = %d' % (len(elements), len(interface_edges)))
        #
        #---------------------------------------------------------------------------------
        # 4.8 Build one-part orphan mesh with duplicated fiber-side interface nodes
        #---------------------------------------------------------------------------------
        #
        if 'Part-1' in model.parts.keys():
            del model.parts['Part-1']
        model.Part(name='Part-1', dimensionality=TWO_D_PLANAR, type=DEFORMABLE_BODY)
        pfinal = model.parts['Part-1']
        #
        for lab in sorted(node_coords.keys()):
            pfinal.Node(coordinates=node_coords[lab], label=int(lab))
        #
        max_node_label = max(node_coords.keys())
        dup = {}
        next_node_label = max_node_label + 1
        for fid in sorted(fiber_boundary_nodes.keys()):
            for old_lab in sorted(fiber_boundary_nodes[fid]):
                dup[(fid, int(old_lab))] = next_node_label
                pfinal.Node(coordinates=node_coords[int(old_lab)], label=next_node_label)
                next_node_label += 1
        #
        matrix_labels = []
        fiber_labels = []
        cohesive_labels = []
        #
        for rec in elements:
            if rec['phase'] == 'fiber':
                fid = rec['fiber_id']
                conn2 = []
                for old_lab in rec['conn']:
                    if (fid, int(old_lab)) in dup:
                        conn2.append(dup[(fid, int(old_lab))])
                    else:
                        conn2.append(int(old_lab))
            else:
                conn2 = [int(v) for v in rec['conn']]
            #
            nodes = tuple(pfinal.nodes.sequenceFromLabels(tuple(conn2)))
            if len(conn2) == 4:
                pfinal.Element(nodes=nodes, elemShape=QUAD4, label=int(rec['label']))
            elif len(conn2) == 3:
                pfinal.Element(nodes=nodes, elemShape=TRI3, label=int(rec['label']))
            else:
                raise RuntimeError('Unexpected continuum element connectivity.')
            #
            if rec['phase'] == 'fiber':
                fiber_labels.append(int(rec['label']))
            else:
                matrix_labels.append(int(rec['label']))
        #
        #---------------------------------------------------------------------------------
        # 4.9 Insert zero-thickness COH2D4 interface elements
        #---------------------------------------------------------------------------------
        #
        next_elem_label = max([rec['label'] for rec in elements]) + 1
        for ie in range(len(interface_edges)):
            fid, a, b = interface_edges[ie]
            da = dup[(fid, int(a))]
            db = dup[(fid, int(b))]
            conn = (int(a), int(b), int(db), int(da))
            nodes = tuple(pfinal.nodes.sequenceFromLabels(tuple(conn)))
            pfinal.Element(nodes=nodes, elemShape=QUAD4, label=next_elem_label)
            cohesive_labels.append(next_elem_label)
            next_elem_label += 1
        #
        pfinal.Set(name='Matrix-Elements', elements=pfinal.elements.sequenceFromLabels(tuple(matrix_labels)))
        pfinal.Set(name='Fiber-Elements', elements=pfinal.elements.sequenceFromLabels(tuple(fiber_labels)))
        pfinal.Set(name='Cohesive-Elements', elements=pfinal.elements.sequenceFromLabels(tuple(cohesive_labels)))
        pfinal.Set(name='Continuum-Elements', elements=pfinal.elements.sequenceFromLabels(tuple(matrix_labels + fiber_labels)))
        print('[phase set] matrix=%d fiber=%d cohesive=%d' % (len(matrix_labels), len(fiber_labels), len(cohesive_labels)))
        #
        if len(matrix_labels) == 0 or len(fiber_labels) == 0:
            raise RuntimeError(
                'Phase split failed: matrix_labels=%d fiber_labels=%d. '
                'Check merged-face classification and periodic mesh geometry.' % (len(matrix_labels), len(fiber_labels))
            )
        elemType1_matrix = mesh.ElemType(elemCode=CPE4R, elemLibrary=EXPLICIT, hourglassControl=DEFAULT, elemDeletion=OFF, maxDegradation=0.98)
        elemType2_matrix = mesh.ElemType(elemCode=CPE3, elemLibrary=EXPLICIT, elemDeletion=OFF, maxDegradation=0.98)
        elemType1_fiber = mesh.ElemType(elemCode=CPE4R, elemLibrary=EXPLICIT, hourglassControl=DEFAULT)
        elemType2_fiber = mesh.ElemType(elemCode=CPE3, elemLibrary=EXPLICIT)
        elemType3 = mesh.ElemType(elemCode=COH2D4, elemLibrary=EXPLICIT)
        #
        pfinal.setElementType(regions=pfinal.sets['Matrix-Elements'], elemTypes=(elemType1_matrix, elemType2_matrix))
        pfinal.setElementType(regions=pfinal.sets['Fiber-Elements'], elemTypes=(elemType1_fiber, elemType2_fiber))
        if len(cohesive_labels) > 0:
            pfinal.setElementType(regions=pfinal.sets['Cohesive-Elements'], elemTypes=(elemType3, ))
        #
        interface_matrix_side_node_labels = sorted(set([int(a) for fid, a, b in interface_edges] + [int(b) for fid, a, b in interface_edges]))
        interface_fiber_side_node_labels = sorted([int(v) for v in dup.values()])
        # For PBC, keep the original continuum interface nodes and exclude only
        # the duplicated fiber-side nodes introduced by cohesive insertion.
        pbc_excluded_node_labels = sorted(set(interface_fiber_side_node_labels))
        #
        if len(interface_matrix_side_node_labels) > 0:
            pfinal.Set(name='Cohesive-BottomNodes', nodes=pfinal.nodes.sequenceFromLabels(tuple(interface_matrix_side_node_labels)))
        if len(interface_fiber_side_node_labels) > 0:
            pfinal.Set(name='Cohesive-TopNodes', nodes=pfinal.nodes.sequenceFromLabels(tuple(interface_fiber_side_node_labels)))
        if len(pbc_excluded_node_labels) > 0:
            pfinal.Set(name='PBC-Excluded-Cohesive-Nodes', nodes=pfinal.nodes.sequenceFromLabels(tuple(pbc_excluded_node_labels)))
        #
        #---------------------------------------------------------------------------------
        # 4.10 Materials and sections
        #---------------------------------------------------------------------------------
        #
        model.Material(name='Material-1')
        model.materials['Material-1'].Density(table=((density, ), ))
        model.materials['Material-1'].DruckerPrager(table=((dp_beta, dp_k, dp_dilation), ))
        model.materials['Material-1'].druckerPrager.DruckerPragerHardening(type=hardening_type, table=((hardening_yield, 0.0), ))
        model.materials['Material-1'].Elastic(table=((matrix_E, matrix_nu), ))
        model.materials['Material-1'].DuctileDamageInitiation(table=((matrix_eps_damage_compression, -0.333333333333, 0.0), (matrix_eps_damage_tension, 0.333333333333, 0.0), ))
        model.materials['Material-1'].ductileDamageInitiation.DamageEvolution(
            type=ENERGY, softening=LINEAR, table=((matrix_Gf, ), ))
        model.HomogeneousSolidSection(name='Section-1', material='Material-1', thickness=thickness)
        region = pfinal.sets['Matrix-Elements']
        pfinal.SectionAssignment(region=region, sectionName='Section-1', offset=0.0, offsetType=MIDDLE_SURFACE, offsetField='', thicknessAssignment=FROM_SECTION)
        model.Material(name='Material-2')
        model.materials['Material-2'].Density(table=((density, ), ))
        model.materials['Material-2'].Elastic(table=((fiber_E, fiber_nu), ))
        model.HomogeneousSolidSection(name='Section-2', material='Material-2', thickness=thickness)
        region = pfinal.sets['Fiber-Elements']
        pfinal.SectionAssignment(region=region, sectionName='Section-2', offset=0.0, offsetType=MIDDLE_SURFACE, offsetField='', thicknessAssignment=FROM_SECTION)
        #
        model.Material(name='Material-3')
        model.materials['Material-3'].Density(table=((density, ), ))
        model.materials['Material-3'].Elastic(type=TRACTION, table=((interface_K, interface_K, interface_K), ))
        model.materials['Material-3'].MaxsDamageInitiation(table=((interface_strength, interface_strength, interface_strength), ))
        model.materials['Material-3'].maxsDamageInitiation.DamageEvolution(type=ENERGY, softening=LINEAR, table=((interface_G, ), ))
        model.CohesiveSection(name='Section-3', material='Material-3', response=TRACTION_SEPARATION, outOfPlaneThickness=None)
        region = pfinal.sets['Cohesive-Elements']
        pfinal.SectionAssignment(region=region, sectionName='Section-3', offset=0.0, offsetType=MIDDLE_SURFACE, offsetField='', thicknessAssignment=FROM_SECTION)
        #
        #---------------------------------------------------------------------------------
        # 4.11 Assembly, step, mass scaling and loading amplitude
        #---------------------------------------------------------------------------------
        #
        a = model.rootAssembly
        for _iname in list(a.instances.keys()):
            del a.instances[_iname]
        for _pname in list(model.parts.keys()):
            if _pname != 'Part-1':
                del model.parts[_pname]
        a.DatumCsysByDefault(CARTESIAN)
        a.Instance(name='Part-1-1', part=pfinal, dependent=ON)
        a.regenerate()
        #
        if str(mass_scaling_method).upper() == 'VARIABLE':
            model.ExplicitDynamicsStep(name='Step-1', previous='Initial', timePeriod=step_time,
                massScaling=((SEMI_AUTOMATIC, MODEL, THROUGHOUT_STEP, 0.0, mass_scaling_dt, BELOW_MIN,
                              mass_scaling_frequency, 0, 0.0, 0.0, 0, None), ))
        elif str(mass_scaling_method).upper() == 'NONE' or not use_variable_mass_scaling:
            model.ExplicitDynamicsStep(name='Step-1', previous='Initial', timePeriod=step_time)
        else:
            raise RuntimeError('Unsupported mass_scaling_method: %s' % str(mass_scaling_method))
        #
        amp_name = 'Amp-SmoothStep-Loading'
        model.SmoothStepAmplitude(name=amp_name, timeSpan=STEP, data=((0.0, 0.0), (step_time, 1.0)))
        #
        try:
            model.fieldOutputRequests['F-Output-1'].setValues(variables=('S', 'E', 'PE', 'PEEQ', 'U', 'RF', 'STATUS', 'SDEG'), numIntervals=120)
        except Exception as exc:
            print('[warn] field output request was not changed: %s' % str(exc))
        #
        #---------------------------------------------------------------------------------
        # 4.12 Boundary node sets for PBC
        #---------------------------------------------------------------------------------
        #
        a = model.rootAssembly
        allnodes = a.instances['Part-1-1'].nodes
        node_by_label = {}
        for _node_obj in allnodes:
            node_by_label[int(_node_obj.label)] = _node_obj
        #
        def _node(lab):
            return node_by_label[int(lab)]
        #
        def _single_node_region(lab):
            return allnodes.sequenceFromLabels((int(lab), ))
        #
        tol = max(1.0E-6 * (rs[0] - rs[1]), 1.0E-5)
        pair_tol = max(
            pbc_pair_tol_factor * tol,
            1.0E-6 * (rs[0] - rs[1]),
            pbc_pair_tol_mesh_fraction * mshsize
        )
        #
        cohesive_bottom_labels_for_pbc = set()
        cohesive_top_labels_for_pbc = set(interface_fiber_side_node_labels) if exclude_cohesive_interface_nodes_from_pbc else set()
        excluded_debug_labels = set(pbc_excluded_node_labels) if exclude_cohesive_interface_nodes_from_pbc else set()
        #
        def _is_close_coord(value, target):
            return abs(float(value) - float(target)) < tol

        def _node_in_excluded_cohesive(lab):
            return int(lab) in cohesive_top_labels_for_pbc or int(lab) in cohesive_bottom_labels_for_pbc

        def _filter_boundary_node_labels(node_array, side_name):
            labels = []
            seen = set()
            for inode in node_array:
                lab = int(inode.label)
                if lab in seen:
                    continue
                seen.add(lab)
                x = float(inode.coordinates[0])
                y = float(inode.coordinates[1])
                on_left = _is_close_coord(x, rs[1])
                on_right = _is_close_coord(x, rs[0])
                on_top = _is_close_coord(y, rs[2])
                on_bottom = _is_close_coord(y, rs[3])
                if exclude_cohesive_interface_nodes_from_pbc and _node_in_excluded_cohesive(lab):
                    continue
                if side_name == 'top':
                    if on_top and (not on_left) and (not on_right):
                        labels.append(lab)
                elif side_name == 'bottom':
                    if on_bottom and (not on_left) and (not on_right):
                        labels.append(lab)
                elif side_name == 'right':
                    if on_right and (not on_top) and (not on_bottom):
                        labels.append(lab)
                elif side_name == 'left':
                    if on_left and (not on_top) and (not on_bottom):
                        labels.append(lab)
            return labels

        boundary_candidates = {
            'top': allnodes.getByBoundingBox(
                xMin=rs[1] - tol, xMax=rs[0] + tol,
                yMin=rs[2] - tol, yMax=rs[2] + tol,
                zMin=-tol, zMax=tol
            ),
            'bottom': allnodes.getByBoundingBox(
                xMin=rs[1] - tol, xMax=rs[0] + tol,
                yMin=rs[3] - tol, yMax=rs[3] + tol,
                zMin=-tol, zMax=tol
            ),
            'right': allnodes.getByBoundingBox(
                xMin=rs[0] - tol, xMax=rs[0] + tol,
                yMin=rs[3] - tol, yMax=rs[2] + tol,
                zMin=-tol, zMax=tol
            ),
            'left': allnodes.getByBoundingBox(
                xMin=rs[1] - tol, xMax=rs[1] + tol,
                yMin=rs[3] - tol, yMax=rs[2] + tol,
                zMin=-tol, zMax=tol
            ),
        }
        boundary_candidate_labels = set()
        for side_name in ['top', 'bottom', 'right', 'left']:
            for inode in boundary_candidates[side_name]:
                boundary_candidate_labels.add(int(inode.label))
        excluded_boundary_count = 0
        cohesive_boundary_count = 0
        for lab in boundary_candidate_labels:
            if lab in excluded_debug_labels:
                excluded_boundary_count += 1
            if _node_in_excluded_cohesive(lab):
                cohesive_boundary_count += 1
        #
        if pair_all_periodic_boundary_nodes_together:
            pbc_groups = [
                {'tag': 'All-Boundary', 'prefix': 'ALL', 'top': [], 'bottom': [], 'right': [], 'left': []},
            ]
            group_index = {'ALL': 0}
        else:
            pbc_groups = [
                {'tag': 'Regular', 'prefix': 'REG', 'top': [], 'bottom': [], 'right': [], 'left': []},
                {'tag': 'Cohesive-Bottom', 'prefix': 'CBOT', 'top': [], 'bottom': [], 'right': [], 'left': []},
                {'tag': 'Cohesive-Top', 'prefix': 'CTOP', 'top': [], 'bottom': [], 'right': [], 'left': []},
            ]
            group_index = {'REG': 0, 'CBOT': 1, 'CTOP': 2}
        #
        if pair_all_periodic_boundary_nodes_together:
            g = pbc_groups[group_index['ALL']]
            g['top'] = _filter_boundary_node_labels(boundary_candidates['top'], 'top')
            g['bottom'] = _filter_boundary_node_labels(boundary_candidates['bottom'], 'bottom')
            g['right'] = _filter_boundary_node_labels(boundary_candidates['right'], 'right')
            g['left'] = _filter_boundary_node_labels(boundary_candidates['left'], 'left')
        else:
            g = pbc_groups[group_index['REG']]
            g['top'] = _filter_boundary_node_labels(boundary_candidates['top'], 'top')
            g['bottom'] = _filter_boundary_node_labels(boundary_candidates['bottom'], 'bottom')
            g['right'] = _filter_boundary_node_labels(boundary_candidates['right'], 'right')
            g['left'] = _filter_boundary_node_labels(boundary_candidates['left'], 'left')
        #
        for g in pbc_groups:
            g['top'].sort(key=lambda lab: (float(_node(lab).coordinates[0]), int(lab)))
            g['bottom'].sort(key=lambda lab: (float(_node(lab).coordinates[0]), int(lab)))
            g['right'].sort(key=lambda lab: (float(_node(lab).coordinates[1]), int(lab)))
            g['left'].sort(key=lambda lab: (float(_node(lab).coordinates[1]), int(lab)))
            print('[pbc group] %s | top=%d bottom=%d right=%d left=%d' % (
                g['tag'], len(g['top']), len(g['bottom']), len(g['right']), len(g['left'])))
        #
        # Write boundary-node list for checking.
        #
        pbc_node_debug_path = jobname + '_pbc_boundary_nodes_grouped.csv'
        with open(pbc_node_debug_path, 'w') as fp:
            wr = csv.writer(fp)
            wr.writerow(['group', 'side', 'order', 'label', 'x', 'y'])
            for g in pbc_groups:
                for side_name in ['top', 'bottom', 'right', 'left']:
                    for ii, lab in enumerate(g[side_name]):
                        nd = _node(lab)
                        wr.writerow([g['tag'], side_name, ii + 1, int(lab),
                                     '%.16g' % nd.coordinates[0],
                                     '%.16g' % nd.coordinates[1]])
        print('[pbc debug] boundary node groups: %s' % pbc_node_debug_path)
        #
        # Helper: one-to-one nearest pairing along a scalar edge coordinate.
        #
        def _nearest_one_to_one_pairs(dep_labels, base_labels, dep_coord_func, base_coord_func,
                                      group_tag, pair_name, jobname):
            if len(dep_labels) != len(base_labels):
                raise RuntimeError(
                    '%s PBC node count mismatch for group %s in %s: %d vs %d'
                    % (pair_name, group_tag, jobname, len(dep_labels), len(base_labels))
                )
            #
            available = set([int(v) for v in base_labels])
            pairs = []
            max_diff = 0.0
            #
            dep_sorted = sorted([int(v) for v in dep_labels], key=lambda lab: (dep_coord_func(lab), lab))
            for dep_lab in dep_sorted:
                cdep = dep_coord_func(dep_lab)
                best_lab = None
                best_diff = 1.0E99
                for base_lab in available:
                    diff = abs(cdep - base_coord_func(base_lab))
                    if diff < best_diff:
                        best_diff = diff
                        best_lab = base_lab
                #
                if best_lab is None:
                    raise RuntimeError('Internal PBC pairing error for group %s in %s.' % (group_tag, jobname))
                #
                if best_diff > pair_tol:
                    raise RuntimeError(
                        '%s PBC nearest-coordinate mismatch for group %s in %s: '
                        'dependent node %d coord=%.12g, nearest base node %d coord=%.12g, '
                        'diff=%.12g, tol=%.12g'
                        % (pair_name, group_tag, jobname, dep_lab, cdep,
                           best_lab, base_coord_func(best_lab), best_diff, pair_tol)
                    )
                #
                available.remove(best_lab)
                pairs.append((dep_lab, best_lab, cdep, base_coord_func(best_lab), best_diff))
                max_diff = max(max_diff, best_diff)
            #
            return pairs, max_diff
        #
        # Corner nodes:
        # Vert-A = LB, Vert-B = RB, Vert-C = RT, Vert-D = LT.
        # Use non-cohesive corner nodes for macro-corner constraints.
        #
        best = [None, None, None, None]
        bestd = [1.0E99, 1.0E99, 1.0E99, 1.0E99]
        targets = [(rs[1], rs[3]), (rs[0], rs[3]), (rs[0], rs[2]), (rs[1], rs[2])]
        for inode in allnodes:
            lab = int(inode.label)
            if lab in cohesive_bottom_labels_for_pbc or lab in cohesive_top_labels_for_pbc:
                continue
            x = float(inode.coordinates[0])
            y = float(inode.coordinates[1])
            for it in range(4):
                d2 = (x - targets[it][0]) ** 2 + (y - targets[it][1]) ** 2
                if d2 < bestd[it]:
                    bestd[it] = d2
                    best[it] = inode
        #
        if best[0] is None or best[1] is None or best[2] is None or best[3] is None:
            raise RuntimeError('Cannot find four non-cohesive corner nodes for PBC.')
        #
        a.Set(nodes=_single_node_region(best[0].label), name='Vert-A')
        a.Set(nodes=_single_node_region(best[1].label), name='Vert-B')
        a.Set(nodes=_single_node_region(best[2].label), name='Vert-C')
        a.Set(nodes=_single_node_region(best[3].label), name='Vert-D')
        #
        right_output_labels = []
        pbc_equation_count = 0
        dependent_dofs = set()
        pbc_pair_debug_path = jobname + '_pbc_nearest_pairs.csv'
        #
        #---------------------------------------------------------------------------------
        # 4.13 Equation constraints (reference-point PBC)
        #---------------------------------------------------------------------------------
        #
        rp_rl = a.ReferencePoint(point=(rs[0] + 0.2 * (rs[0] - rs[1]), 0.5 * (rs[2] + rs[3]), 0.0))
        rp_tb = a.ReferencePoint(point=(0.5 * (rs[0] + rs[1]), rs[2] + 0.2 * (rs[2] - rs[3]), 0.0))
        a.Set(name='RP-RL', referencePoints=(a.referencePoints[rp_rl.id], ))
        a.Set(name='RP-TB', referencePoints=(a.referencePoints[rp_tb.id], ))
        #
        # Corner periodic equations:
        #   Vert-B - Vert-A = RP-RL
        #   Vert-D - Vert-A = RP-TB
        #   Vert-C - Vert-B = RP-TB
        #
        corner_defs = [
            ('B-A', 'Vert-B', int(best[1].label), 'Vert-A', 'RP-RL'),
            ('D-A', 'Vert-D', int(best[3].label), 'Vert-A', 'RP-TB'),
            ('C-B', 'Vert-C', int(best[2].label), 'Vert-B', 'RP-TB'),
        ]
        for ctag, dep_set, dep_lab, base_set, rp_set in corner_defs:
            for dof, dname in [(1, 'X'), (2, 'Y')]:
                dep_key = (dep_lab, dof)
                if dep_key in dependent_dofs:
                    raise RuntimeError('Over-constrained corner DOF: node=%d dof=%d in %s' % (dep_lab, dof, jobname))
                dependent_dofs.add(dep_key)
                model.Equation(
                    name='PBC-Corner-%s-%s' % (ctag, dname),
                    terms=((1.0, dep_set, dof), (-1.0, base_set, dof), (-1.0, rp_set, dof))
                )
                pbc_equation_count += 1
        #
        with open(pbc_pair_debug_path, 'w') as fp:
            wr = csv.writer(fp)
            wr.writerow(['group', 'pair_type', 'order',
                         'dependent_label', 'dependent_x', 'dependent_y',
                         'base_label', 'base_x', 'base_y',
                         'coord_difference', 'tolerance'])
            #
            for g in pbc_groups:
                # Right-left: dependent side = right, base side = left.
                rl_pairs, rl_maxdiff = _nearest_one_to_one_pairs(
                    g['right'],
                    g['left'],
                    lambda lab: float(_node(lab).coordinates[1]),
                    lambda lab: float(_node(lab).coordinates[1]),
                    g['tag'],
                    'Right/left',
                    jobname
                )
                #
                for ii, (lab_r, lab_l, c_r, c_l, diff) in enumerate(rl_pairs):
                    s1 = 'PBC-%s-BC-%d' % (g['prefix'], ii + 1)
                    s2 = 'PBC-%s-DA-%d' % (g['prefix'], ii + 1)
                    a.Set(nodes=_single_node_region(lab_r), name=s1)
                    a.Set(nodes=_single_node_region(lab_l), name=s2)
                    for dof in [1, 2]:
                        dep_key = (lab_r, dof)
                        if dep_key in dependent_dofs:
                            raise RuntimeError('Over-constrained right-edge DOF: node=%d dof=%d in %s' % (lab_r, dof, jobname))
                        dependent_dofs.add(dep_key)
                    model.Equation(name='PBC-%s-BC-DA-X-%d' % (g['prefix'], ii + 1),
                        terms=((1.0, s1, 1), (-1.0, s2, 1), (-1.0, 'RP-RL', 1)))
                    model.Equation(name='PBC-%s-BC-DA-Y-%d' % (g['prefix'], ii + 1),
                        terms=((1.0, s1, 2), (-1.0, s2, 2), (-1.0, 'RP-RL', 2)))
                    pbc_equation_count += 2
                    right_output_labels.append(lab_r)
                    nr = _node(lab_r)
                    nl = _node(lab_l)
                    wr.writerow([g['tag'], 'RL', ii + 1,
                                 lab_r, '%.16g' % nr.coordinates[0], '%.16g' % nr.coordinates[1],
                                 lab_l, '%.16g' % nl.coordinates[0], '%.16g' % nl.coordinates[1],
                                 '%.16g' % diff, '%.16g' % pair_tol])
                #
                # Top-bottom: dependent side = top, base side = bottom.
                tb_pairs, tb_maxdiff = _nearest_one_to_one_pairs(
                    g['top'],
                    g['bottom'],
                    lambda lab: float(_node(lab).coordinates[0]),
                    lambda lab: float(_node(lab).coordinates[0]),
                    g['tag'],
                    'Top/bottom',
                    jobname
                )
                #
                for ii, (lab_t, lab_b, c_t, c_b, diff) in enumerate(tb_pairs):
                    s1 = 'PBC-%s-CD-%d' % (g['prefix'], ii + 1)
                    s2 = 'PBC-%s-AB-%d' % (g['prefix'], ii + 1)
                    a.Set(nodes=_single_node_region(lab_t), name=s1)
                    a.Set(nodes=_single_node_region(lab_b), name=s2)
                    for dof in [1, 2]:
                        dep_key = (lab_t, dof)
                        if dep_key in dependent_dofs:
                            raise RuntimeError('Over-constrained top-edge DOF: node=%d dof=%d in %s' % (lab_t, dof, jobname))
                        dependent_dofs.add(dep_key)
                    model.Equation(name='PBC-%s-CD-AB-X-%d' % (g['prefix'], ii + 1),
                        terms=((1.0, s1, 1), (-1.0, s2, 1), (-1.0, 'RP-TB', 1)))
                    model.Equation(name='PBC-%s-CD-AB-Y-%d' % (g['prefix'], ii + 1),
                        terms=((1.0, s1, 2), (-1.0, s2, 2), (-1.0, 'RP-TB', 2)))
                    pbc_equation_count += 2
                    nt = _node(lab_t)
                    nb = _node(lab_b)
                    wr.writerow([g['tag'], 'TB', ii + 1,
                                 lab_t, '%.16g' % nt.coordinates[0], '%.16g' % nt.coordinates[1],
                                 lab_b, '%.16g' % nb.coordinates[0], '%.16g' % nb.coordinates[1],
                                 '%.16g' % diff, '%.16g' % pair_tol])
                #
                print('[pbc nearest] %s | RL pairs=%d maxdiff=%.6g | TB pairs=%d maxdiff=%.6g' % (
                    g['tag'], len(rl_pairs), rl_maxdiff, len(tb_pairs), tb_maxdiff))
        #
        right_output_labels.append(int(best[1].label))
        right_output_labels.append(int(best[2].label))
        right_output_labels = sorted(list(set([int(v) for v in right_output_labels])))
        if len(right_output_labels) > 0:
            a.Set(nodes=allnodes.sequenceFromLabels(tuple(right_output_labels)), name='Right-Edge-Output-Nodes')
        #
        print('[pbc summary] equations=%d | cohesive_boundary_nodes=%d | excluded_debug_boundary_nodes=%d' % (
            pbc_equation_count, cohesive_boundary_count, excluded_boundary_count))
        print('[pbc debug] nearest pairs: %s' % pbc_pair_debug_path)
        #
        #---------------------------------------------------------------------------------
        # 4.14 Boundary conditions and output
        #---------------------------------------------------------------------------------
        #
        model.DisplacementBC(name='BC-LB-fixed', createStepName='Initial', region=a.sets['Vert-A'],
            u1=0.0, u2=0.0, ur3=UNSET)
        model.DisplacementBC(name='BC-RP-RL-load', createStepName='Step-1', region=a.sets['RP-RL'],
            u1=disp, u2=0.0, ur3=0.0, amplitude=amp_name, fixed=OFF,
            distributionType=UNIFORM, fieldName='', localCsys=None)
        # Uniaxial loading condition:
        # keep macroscopic shear strain at zero via RP-TB U1 = 0,
        # but release transverse normal strain (RP-TB U2 is free).
        model.DisplacementBC(name='BC-RP-TB-shear-only', createStepName='Initial', region=a.sets['RP-TB'],
            u1=0.0, u2=UNSET, ur3=0.0)
        #
        try:
            model.HistoryOutputRequest(name='H-Output-1', createStepName='Step-1', variables=('U1', 'RF1'),
                region=a.sets['RP-RL'], sectionPoints=DEFAULT, rebar=EXCLUDE, timeInterval=history_dt)
        except Exception:
            model.HistoryOutputRequest(name='H-Output-1', createStepName='Step-1', variables=('U1', 'RF1'),
                region=a.sets['RP-RL'], sectionPoints=DEFAULT, rebar=EXCLUDE)
        #
        #---------------------------------------------------------------------------------
        # 4.15 Job, writeInput and saveAs
        #---------------------------------------------------------------------------------
        #
        mdb.Job(name=jobname, model='Model-1', description='Yang2012 transverse RVE, fixed imported fiber centers',
            type=ANALYSIS, atTime=None, waitMinutes=0, waitHours=0, queue=None,
            memory=90, memoryUnits=PERCENTAGE, getMemoryFromAnalysis=True,
            explicitPrecision=job_explicit_precision, nodalOutputPrecision=job_nodal_output_precision,
            echoPrint=OFF, modelPrint=OFF, contactPrint=OFF, historyPrint=OFF,
            userSubroutine='', scratch='', resultsFormat=ODB,
            multiprocessingMode=DEFAULT, parallelizationMethodExplicit=DOMAIN,
            numDomains=num_cpus, activateLoadBalancing=False, numCpus=num_cpus, numGPUs=0)
        #
        if write_input:
            mdb.jobs[jobname].writeInput(consistencyChecking=OFF)
            inp_path = jobname + '.inp'
            _patch_drucker_prager_hardening_inp(inp_path, load_case)
            with open(inp_path, 'r') as fp:
                inp_text_original = fp.read()
            inp_text = inp_text_original.lower()
            if '*equation' not in inp_text:
                raise RuntimeError('Generated inp has no *Equation: %s.inp' % jobname)
            if str(mass_scaling_method).upper() == 'VARIABLE' and '*variable mass scaling' not in inp_text:
                raise RuntimeError('Generated inp has no *Variable Mass Scaling: %s.inp' % jobname)
            if amp_name.lower() not in inp_text:
                raise RuntimeError('Generated inp has no smooth loading amplitude: %s.inp' % jobname)
            print('[writeInput] %s.inp' % jobname)
        #
        if save_cae:
            mdb.saveAs(pathName=pathname)
            print('[saveAs] %s' % pathname)
        #
        if 'excluded_boundary_count' not in locals():
            excluded_boundary_count = 0
        print('[done] %s | matrix=%d fiber=%d cohesive=%d | excluded boundary cohesive nodes=%d' % (
            jobname, len(matrix_labels), len(fiber_labels), len(cohesive_labels), excluded_boundary_count))
        #
        #mdb.jobs[jobname].submit(consistencyChecking=OFF)
        #mdb.jobs[jobname].waitForCompletion()
        #
#
os.chdir(root_dir)
print('[%s] complete | CAE and INP files were generated only. No job was submitted.' % SCRIPT_TAG)
