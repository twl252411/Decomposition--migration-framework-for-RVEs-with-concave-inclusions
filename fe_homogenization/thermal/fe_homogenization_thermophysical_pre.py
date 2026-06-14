# Single-RVE Abaqus/CAE preprocessing script for thermal homogenization.
# Expected phase names: Part-1 matrix, Part-2 inclusion.
from abaqus import *
from abaqusConstants import *
from caeModules import *
from driverUtils import executeOnCaeStartup
import mesh
import regionToolset
import os
import sys
import time

executeOnCaeStartup()


# ---------------------------------------------------------------------------
# User parameters
SCRIPT_TAG = 'fe_homogenization_thermophysical_pre'
RVE_SIZE = float(os.environ.get('FE_HOMOG_THERMAL_RVE_SIZE', os.environ.get('STAGE1_RVE_SIZE', '200.0')))
RVE_BOUNDS = [RVE_SIZE, 0.0, RVE_SIZE, 0.0]  # [xmax, xmin, ymax, ymin]
RVE_CAE_PATH = os.environ.get('FE_HOMOG_THERMAL_CAE_FILE', os.environ.get('STAGE1_CAE_FILE', 'RVE_1.cae'))
MESH_SIZE = float(os.environ.get('FE_HOMOG_THERMAL_MESH_SIZE', '1.0'))

# Mechanical properties for thermal-expansion homogenization.
# Units must be consistent with the generated geometry.
MATRIX_E = 0.00434
MATRIX_NU = 0.37
INCLUSION_E = 0.0231
INCLUSION_NU = 0.40
MATRIX_CTE = 43.92e-6
INCLUSION_CTE = 10.08e-6

# Conductivity properties for steady heat-transfer homogenization.
MATRIX_COND = 380
INCLUSION_COND = 0.8
DELTA_T = 1.0

TOL = 1.0e-2
BOUNDARY_TOL = 1.0e-3
MESH_MIN_SIZE_FACTOR = 0.1
LAST_PERIODIC_MISMATCH = {}

def _assign_mechanical_materials(model):
    model.Material(name='Matrix')
    model.materials['Matrix'].Elastic(table=((MATRIX_E, MATRIX_NU), ))
    model.materials['Matrix'].Expansion(table=((MATRIX_CTE, ), ))
    model.HomogeneousSolidSection(name='Section-Matrix', material='Matrix', thickness=None)
    p = model.parts['Part-1']
    p.SectionAssignment(
        region=regionToolset.Region(faces=p.faces),
        sectionName='Section-Matrix',
        offset=0.0,
        offsetType=MIDDLE_SURFACE,
        offsetField='',
        thicknessAssignment=FROM_SECTION,
    )

    model.Material(name='Inclusion')
    model.materials['Inclusion'].Elastic(table=((INCLUSION_E, INCLUSION_NU), ))
    model.materials['Inclusion'].Expansion(table=((INCLUSION_CTE, ), ))
    model.HomogeneousSolidSection(name='Section-Inclusion', material='Inclusion', thickness=None)
    p = model.parts['Part-2']
    p.SectionAssignment(
        region=regionToolset.Region(faces=p.faces),
        sectionName='Section-Inclusion',
        offset=0.0,
        offsetType=MIDDLE_SURFACE,
        offsetField='',
        thicknessAssignment=FROM_SECTION,
    )


def _assign_thermal_materials(model):
    model.Material(name='Matrix')
    model.materials['Matrix'].Conductivity(table=((MATRIX_COND, ), ))
    model.HomogeneousSolidSection(name='Section-Matrix', material='Matrix', thickness=None)
    p = model.parts['Part-1']
    p.SectionAssignment(
        region=regionToolset.Region(faces=p.faces),
        sectionName='Section-Matrix',
        offset=0.0,
        offsetType=MIDDLE_SURFACE,
        offsetField='',
        thicknessAssignment=FROM_SECTION,
    )

    model.Material(name='Inclusion')
    model.materials['Inclusion'].Conductivity(table=((INCLUSION_COND, ), ))
    model.HomogeneousSolidSection(name='Section-Inclusion', material='Inclusion', thickness=None)
    p = model.parts['Part-2']
    p.SectionAssignment(
        region=regionToolset.Region(faces=p.faces),
        sectionName='Section-Inclusion',
        offset=0.0,
        offsetType=MIDDLE_SURFACE,
        offsetField='',
        thicknessAssignment=FROM_SECTION,
    )


def _outer_edge_sequence(instance, bounds):
    xmax, xmin, ymax, ymin = bounds
    edges = instance.edges[0:0]
    for edge in instance.edges:
        pt = edge.pointOn[0]
        if (
            abs(pt[1] - ymin) < BOUNDARY_TOL
            or abs(pt[0] - xmax) < BOUNDARY_TOL
            or abs(pt[1] - ymax) < BOUNDARY_TOL
            or abs(pt[0] - xmin) < BOUNDARY_TOL
        ):
                edges += instance.edges[edge.index:edge.index + 1]
    return edges


def _edge_endpoint_coordinates(instance, edge):
    try:
        vertex_ids = edge.getVertices()
        points = [instance.vertices[idx].pointOn[0] for idx in vertex_ids]
        if len(points) >= 2:
            return points[0], points[-1]
    except Exception:
        pass
    try:
        bbox = edge.getBoundingBox()
        low = bbox['low']
        high = bbox['high']
        return low, high
    except Exception:
        pass
    pt = edge.pointOn[0]
    return pt, pt


def _boundary_edge_infos(instance, bounds, side):
    xmax, xmin, ymax, ymin = bounds
    infos = []
    for edge in instance.edges:
        pt = edge.pointOn[0]
        if side == 'bottom' and abs(pt[1] - ymin) >= BOUNDARY_TOL:
            continue
        if side == 'top' and abs(pt[1] - ymax) >= BOUNDARY_TOL:
            continue
        if side == 'left' and abs(pt[0] - xmin) >= BOUNDARY_TOL:
            continue
        if side == 'right' and abs(pt[0] - xmax) >= BOUNDARY_TOL:
            continue

        p0, p1 = _edge_endpoint_coordinates(instance, edge)
        if side in ('bottom', 'top'):
            a = min(float(p0[0]), float(p1[0]))
            b = max(float(p0[0]), float(p1[0]))
            p0_coord = float(p0[0])
            p1_coord = float(p1[0])
        else:
            a = min(float(p0[1]), float(p1[1]))
            b = max(float(p0[1]), float(p1[1]))
            p0_coord = float(p0[1])
            p1_coord = float(p1[1])
        if b - a < 1.0e-8:
            continue
        infos.append({
            'instance': instance.name,
            'edge_index': edge.index,
            'start': a,
            'end': b,
            'p0_coord': p0_coord,
            'p1_coord': p1_coord,
            'length': b - a,
            'edges': instance.edges[edge.index:edge.index + 1],
        })
    infos.sort(key=lambda item: (item['start'], item['end'], item['instance'], item['edge_index']))
    return infos


def _edge_interval_key(info):
    return (round(float(info['start']), 4), round(float(info['end']), 4))


def _pair_periodic_edge_infos(first, second, label):
    first_by_key = {}
    second_by_key = {}
    for info in first:
        first_by_key.setdefault(_edge_interval_key(info), []).append(info)
    for info in second:
        second_by_key.setdefault(_edge_interval_key(info), []).append(info)

    if set(first_by_key.keys()) != set(second_by_key.keys()):
        first_only = sorted(set(first_by_key.keys()) - set(second_by_key.keys()))
        second_only = sorted(set(second_by_key.keys()) - set(first_by_key.keys()))
        print('Unmatched periodic {} edge intervals first-only: {}'.format(label, first_only[:20]))
        print('Unmatched periodic {} edge intervals second-only: {}'.format(label, second_only[:20]))
        sys.stdout.flush()
        raise ValueError('Periodic {} boundary edge intervals differ before meshing'.format(label))

    pairs = []
    for key in sorted(first_by_key.keys()):
        lhs = first_by_key[key]
        rhs = second_by_key[key]
        if len(lhs) != len(rhs):
            raise ValueError(
                'Periodic {} boundary edge interval {} count differs: {} vs {}'.format(
                    label, key, len(lhs), len(rhs)
                )
            )
        for left_info, right_info in zip(lhs, rhs):
            pairs.append((left_info, right_info))
    return pairs


def _boundary_vertex_coordinates(instance, bounds, side):
    xmax, xmin, ymax, ymin = bounds
    coords = []
    for vertex in instance.vertices:
        pt = vertex.pointOn[0]
        x = float(pt[0])
        y = float(pt[1])
        if side == 'bottom' and abs(y - ymin) < BOUNDARY_TOL:
            coords.append(x)
        elif side == 'top' and abs(y - ymax) < BOUNDARY_TOL:
            coords.append(x)
        elif side == 'left' and abs(x - xmin) < BOUNDARY_TOL:
            coords.append(y)
        elif side == 'right' and abs(x - xmax) < BOUNDARY_TOL:
            coords.append(y)
    return coords


def _unique_sorted_coords(coords):
    values = []
    for coord in sorted(float(v) for v in coords):
        if not values or abs(coord - values[-1]) > 1.0e-6:
            values.append(coord)
    return values


def _partition_side_at_coords(assembly, instances, side, coords):
    eps = 1.0e-6
    coords = _unique_sorted_coords(coords)
    while True:
        changed = False
        infos = []
        for entity in instances:
            infos.extend(_boundary_edge_infos(entity, RVE_BOUNDS, side))
        for coord in coords:
            for info in infos:
                if coord <= info['start'] + eps or coord >= info['end'] - eps:
                    continue
                denom = float(info['p1_coord']) - float(info['p0_coord'])
                if abs(denom) < 1.0e-12:
                    continue
                param = (float(coord) - float(info['p0_coord'])) / denom
                if param <= eps or param >= 1.0 - eps:
                    continue
                assembly.PartitionEdgeByParam(edges=info['edges'], parameter=param)
                changed = True
                break
            if changed:
                break
        if not changed:
            break


def _partition_periodic_boundary_vertices(assembly, instances):
    side_coords = {'bottom': [], 'top': [], 'left': [], 'right': []}
    for inst in instances:
        for side in side_coords:
            side_coords[side].extend(_boundary_vertex_coordinates(inst, RVE_BOUNDS, side))
    horizontal = _unique_sorted_coords(side_coords['bottom'] + side_coords['top'])
    vertical = _unique_sorted_coords(side_coords['left'] + side_coords['right'])
    _partition_side_at_coords(assembly, instances, 'bottom', horizontal)
    _partition_side_at_coords(assembly, instances, 'top', horizontal)
    _partition_side_at_coords(assembly, instances, 'left', vertical)
    _partition_side_at_coords(assembly, instances, 'right', vertical)


def _seed_periodic_outer_edges_by_number(assembly, instances):
    side_infos = {'bottom': [], 'top': [], 'left': [], 'right': []}
    for entity in instances:
        for side in side_infos:
            side_infos[side].extend(_boundary_edge_infos(entity, RVE_BOUNDS, side))

    periodic_pairs = []
    periodic_pairs.extend(_pair_periodic_edge_infos(side_infos['bottom'], side_infos['top'], 'bottom/top'))
    periodic_pairs.extend(_pair_periodic_edge_infos(side_infos['left'], side_infos['right'], 'left/right'))

    for first, second in periodic_pairs:
        length = 0.5 * (float(first['length']) + float(second['length']))
        n_elem = max(1, int(round(length / float(MESH_SIZE))))
        assembly.seedEdgeByNumber(edges=first['edges'], number=n_elem, constraint=FIXED)
        assembly.seedEdgeByNumber(edges=second['edges'], number=n_elem, constraint=FIXED)


def _interior_edge_sequence(instance, bounds):
    xmax, xmin, ymax, ymin = bounds
    edges = instance.edges[0:0]
    for edge in instance.edges:
        pt = edge.pointOn[0]
        if (
            abs(pt[0] - xmax) > BOUNDARY_TOL
            and abs(pt[0] - xmin) > BOUNDARY_TOL
            and abs(pt[1] - ymax) > BOUNDARY_TOL
            and abs(pt[1] - ymin) > BOUNDARY_TOL
        ):
            edges += instance.edges[edge.index:edge.index + 1]
    return edges


def _rounded_point_key(point, ndigits=5):
    return (round(float(point[0]), ndigits), round(float(point[1]), ndigits))


def _interface_edge_key(instance, edge):
    p0, p1 = _edge_endpoint_coordinates(instance, edge)
    k0 = _rounded_point_key(p0)
    k1 = _rounded_point_key(p1)
    if k1 < k0:
        k0, k1 = k1, k0
    return (k0, k1)


def _edge_length_from_endpoints(instance, edge):
    p0, p1 = _edge_endpoint_coordinates(instance, edge)
    dx = float(p1[0]) - float(p0[0])
    dy = float(p1[1]) - float(p0[1])
    return (dx * dx + dy * dy) ** 0.5


def _seed_phase_interface_edges_by_number(assembly, instances):
    if len(instances) < 2:
        return
    matrix_inst, inclusion_inst = instances
    matrix_edges = _interior_edge_sequence(matrix_inst, RVE_BOUNDS)
    inclusion_edges = _interior_edge_sequence(inclusion_inst, RVE_BOUNDS)
    if len(matrix_edges) == 0 or len(inclusion_edges) == 0:
        return

    matrix_by_key = {}
    inclusion_by_key = {}
    for edge in matrix_edges:
        matrix_by_key.setdefault(_interface_edge_key(matrix_inst, edge), []).append(edge)
    for edge in inclusion_edges:
        inclusion_by_key.setdefault(_interface_edge_key(inclusion_inst, edge), []).append(edge)

    shared_keys = sorted(set(matrix_by_key.keys()).intersection(set(inclusion_by_key.keys())))
    if not shared_keys:
        return

    for key in shared_keys:
        matrix_list = matrix_by_key[key]
        inclusion_list = inclusion_by_key[key]
        if len(matrix_list) != len(inclusion_list):
            continue
        for matrix_edge, inclusion_edge in zip(matrix_list, inclusion_list):
            length = 0.5 * (
                _edge_length_from_endpoints(matrix_inst, matrix_edge)
                + _edge_length_from_endpoints(inclusion_inst, inclusion_edge)
            )
            n_elem = max(1, int(round(length / float(MESH_SIZE))))
            assembly.seedEdgeByNumber(
                edges=matrix_inst.edges[matrix_edge.index:matrix_edge.index + 1],
                number=n_elem,
                constraint=FIXED,
            )
            assembly.seedEdgeByNumber(
                edges=inclusion_inst.edges[inclusion_edge.index:inclusion_edge.index + 1],
                number=n_elem,
                constraint=FIXED,
            )


def _seed_and_mesh_mechanical(model):
    assembly = model.rootAssembly
    instances = (assembly.instances['Part-1-1'], assembly.instances['Part-2-1'])
    _partition_periodic_boundary_vertices(assembly, instances)
    assembly.seedPartInstance(
        regions=instances,
        size=MESH_SIZE,
        deviationFactor=0.1,
        minSizeFactor=MESH_MIN_SIZE_FACTOR,
    )
    _seed_periodic_outer_edges_by_number(assembly, instances)
    _seed_phase_interface_edges_by_number(assembly, instances)
    faces = assembly.instances['Part-1-1'].faces + assembly.instances['Part-2-1'].faces
    assembly.setMeshControls(regions=faces, elemShape=TRI, technique=FREE, algorithm=ADVANCING_FRONT)
    elem_type = mesh.ElemType(elemCode=CPE3, elemLibrary=STANDARD)
    assembly.setElementType(regions=(faces, ), elemTypes=(elem_type, ))
    assembly.generateMesh(regions=instances)


def _seed_and_mesh_thermal(model):
    assembly = model.rootAssembly
    instances = (assembly.instances['Part-1-1'], assembly.instances['Part-2-1'])
    _partition_periodic_boundary_vertices(assembly, instances)
    assembly.seedPartInstance(
        regions=instances,
        size=MESH_SIZE,
        deviationFactor=0.1,
        minSizeFactor=MESH_MIN_SIZE_FACTOR,
    )
    _seed_periodic_outer_edges_by_number(assembly, instances)
    _seed_phase_interface_edges_by_number(assembly, instances)
    faces = assembly.instances['Part-1-1'].faces + assembly.instances['Part-2-1'].faces
    assembly.setMeshControls(regions=faces, elemShape=TRI, technique=FREE, algorithm=ADVANCING_FRONT)
    elem_type = mesh.ElemType(elemCode=DC2D3, elemLibrary=STANDARD)
    assembly.setElementType(regions=(faces, ), elemTypes=(elem_type, ))
    assembly.generateMesh(regions=instances)


def _nodes_from_edge_set(assembly, name, edges):
    if name in assembly.sets:
        del assembly.sets[name]
    assembly.Set(edges=edges, name=name)
    return assembly.sets[name].nodes


def _non_outer_nodes(instance, bounds):
    xmax, xmin, ymax, ymin = bounds
    nodes = []
    for node in instance.nodes:
        x = float(node.coordinates[0])
        y = float(node.coordinates[1])
        if (
            abs(x - xmin) > BOUNDARY_TOL
            and abs(x - xmax) > BOUNDARY_TOL
            and abs(y - ymin) > BOUNDARY_TOL
            and abs(y - ymax) > BOUNDARY_TOL
        ):
            nodes.append(node)
    return nodes


def _match_interface_nodes(matrix_nodes, inclusion_nodes):
    if len(matrix_nodes) == 0:
        raise ValueError('Matrix interface has no nodes')

    matrix_by_coord = {}
    for node in matrix_nodes:
        key = _rounded_point_key(node.coordinates, ndigits=5)
        matrix_by_coord.setdefault(key, []).append(node)

    pairs = []
    max_distance = 0.0
    for node in inclusion_nodes:
        key = _rounded_point_key(node.coordinates, ndigits=5)
        candidates = matrix_by_coord.get(key, [])
        chosen = candidates[0] if candidates else None
        if chosen is None:
            best = None
            best_dist2 = None
            x = float(node.coordinates[0])
            y = float(node.coordinates[1])
            for candidate in matrix_nodes:
                dx = float(candidate.coordinates[0]) - x
                dy = float(candidate.coordinates[1]) - y
                dist2 = dx * dx + dy * dy
                if best is None or dist2 < best_dist2:
                    best = candidate
                    best_dist2 = dist2
            chosen = best
            max_distance = max(max_distance, best_dist2 ** 0.5)
        pairs.append((chosen, node))

    print(
        '[stage1-fe] interface node equations: matrix_nodes={}, inclusion_nodes={}, max_nearest_distance={:.6g}'.format(
            len(matrix_nodes), len(inclusion_nodes), max_distance
        ),
        flush=True,
    )
    return pairs


def _couple_phase_interfaces(model, dofs):
    assembly = model.rootAssembly
    matrix_inst = assembly.instances['Part-1-1']
    inclusion_inst = assembly.instances['Part-2-1']
    matrix_edges = _interior_edge_sequence(matrix_inst, RVE_BOUNDS)
    inclusion_edges = _interior_edge_sequence(inclusion_inst, RVE_BOUNDS)
    if len(matrix_edges) == 0 or len(inclusion_edges) == 0:
        return

    matrix_nodes = _nodes_from_edge_set(assembly, 'Interface-Matrix-Nodes', matrix_edges)
    inclusion_nodes = _nodes_from_edge_set(assembly, 'Common-Surf-1', inclusion_edges)
    if len(matrix_nodes) == 0:
        matrix_nodes = _non_outer_nodes(matrix_inst, RVE_BOUNDS)
        print(
            '[stage1-fe] matrix interface edge set has no nodes; using {} non-outer matrix nodes for nearest coupling'.format(
                len(matrix_nodes)
            ),
            flush=True,
        )
    if len(inclusion_nodes) == 0:
        inclusion_nodes = list(inclusion_inst.nodes)
        print(
            '[stage1-fe] inclusion interface edge set has no nodes; using all {} inclusion nodes for nearest coupling'.format(
                len(inclusion_nodes)
            ),
            flush=True,
        )
    pairs = _match_interface_nodes(matrix_nodes, inclusion_nodes)
    for i, pair in enumerate(pairs):
        idx = i + 1
        _make_single_node_set(assembly, 'Int-M-{}'.format(idx), ('Part-1-1', pair[0].label))
        _make_single_node_set(assembly, 'Int-I-{}'.format(idx), ('Part-2-1', pair[1].label))
        for dof in dofs:
            model.Equation(
                name='IntTie-{}-{}'.format(dof, idx),
                terms=((1.0, 'Int-I-{}'.format(idx), dof), (-1.0, 'Int-M-{}'.format(idx), dof)),
            )


def _couple_phase_interfaces_mechanical(model):
    _couple_phase_interfaces(model, (1, 2))


def _couple_phase_interfaces_thermal(model):
    _couple_phase_interfaces(model, (11, ))


def _boundary_nodes(assembly, inst_name, bounds, exclude_labels=None):
    xmax, xmin, ymax, ymin = bounds
    inst = assembly.instances[inst_name]
    exclude_labels = set() if exclude_labels is None else set(exclude_labels)
    bottom = []
    right = []
    top = []
    left = []
    corners = {'LB': None, 'RB': None, 'RT': None, 'LT': None}
    for node in inst.nodes:
        if node.label in exclude_labels:
            continue
        x, y = node.coordinates[0], node.coordinates[1]
        is_left = abs(x - xmin) < BOUNDARY_TOL
        is_right = abs(x - xmax) < BOUNDARY_TOL
        is_bottom = abs(y - ymin) < BOUNDARY_TOL
        is_top = abs(y - ymax) < BOUNDARY_TOL
        item = (inst_name, node.label, x, y)
        if is_left and is_bottom:
            corners['LB'] = item
            continue
        if is_right and is_bottom:
            corners['RB'] = item
            continue
        if is_right and is_top:
            corners['RT'] = item
            continue
        if is_left and is_top:
            corners['LT'] = item
            continue
        if is_bottom:
            bottom.append(item)
        elif is_right:
            right.append(item)
        elif is_top:
            top.append(item)
        elif is_left:
            left.append(item)
    corner_targets = {
        'LB': (xmin, ymin),
        'RB': (xmax, ymin),
        'RT': (xmax, ymax),
        'LT': (xmin, ymax),
    }
    corner_snap_tol = max(5.0 * BOUNDARY_TOL, 0.05 * float(MESH_SIZE))
    for key, target in corner_targets.items():
        if corners[key] is not None:
            continue
        best_item = None
        best_dist = None
        for node in inst.nodes:
            if node.label in exclude_labels:
                continue
            x, y = node.coordinates[0], node.coordinates[1]
            dist = ((float(x) - float(target[0])) ** 2 + (float(y) - float(target[1])) ** 2) ** 0.5
            if best_dist is None or dist < best_dist:
                best_dist = dist
                best_item = (inst_name, node.label, x, y)
        if best_item is not None and best_dist is not None and best_dist <= corner_snap_tol:
            print(
                '[stage1-fe] snapped {} corner node on {} with distance {:.6g}'.format(
                    key, inst_name, best_dist
                ),
                flush=True,
            )
            corners[key] = best_item
    bottom.sort(key=lambda v: v[2])
    top.sort(key=lambda v: v[2])
    left.sort(key=lambda v: v[3])
    right.sort(key=lambda v: v[3])
    return bottom, right, top, left, corners


def _make_single_node_set(assembly, name, item):
    inst_name, label = item[0], item[1]
    nodes = assembly.instances[inst_name].nodes.sequenceFromLabels(labels=(label, ))
    assembly.Set(nodes=nodes, name=name)


def _periodic_boundary_items(model):
    assembly = model.rootAssembly
    bottom = []
    right = []
    top = []
    left = []
    corner_candidates = {'LB': [], 'RB': [], 'RT': [], 'LT': []}
    for inst_name in ('Part-1-1', 'Part-2-1'):
        btm, rgt, tp, lft, crn = _boundary_nodes(assembly, inst_name, RVE_BOUNDS)
        bottom.extend(btm)
        right.extend(rgt)
        top.extend(tp)
        left.extend(lft)
        for key in corner_candidates:
            if crn[key] is not None:
                corner_candidates[key].append(crn[key])

    bottom.sort(key=lambda v: v[2])
    top.sort(key=lambda v: v[2])
    left.sort(key=lambda v: v[3])
    right.sort(key=lambda v: v[3])
    return bottom, right, top, left, corner_candidates


def _record_periodic_mismatch(label, first_only, second_only):
    global LAST_PERIODIC_MISMATCH
    LAST_PERIODIC_MISMATCH = {
        'label': label,
        'first_only': [float(v) for v in first_only],
        'second_only': [float(v) for v in second_only],
    }


def _match_periodic_items(first, second, coord_index, label):
    pairs = []
    max_distance = 0.0

    def nearest(item, candidates):
        best_j = None
        best_d = None
        for j, cand in enumerate(candidates):
            dist = abs(float(item[coord_index]) - float(cand[coord_index]))
            if best_d is None or dist < best_d:
                best_d = dist
                best_j = j
        return best_j, best_d

    for item in first:
        best_j, best_d = nearest(item, second)
        if best_j is None:
            _record_periodic_mismatch(label, [float(item[coord_index])], [])
            raise ValueError('No periodic counterpart found for {} boundary node'.format(label))
        max_distance = max(max_distance, float(best_d))
        pairs.append((item, second[best_j]))

    covered_second = set((pair[1][0], pair[1][1]) for pair in pairs)
    for item in second:
        item_key = (item[0], item[1])
        if item_key in covered_second:
            continue
        best_j, best_d = nearest(item, first)
        if best_j is None:
            _record_periodic_mismatch(label, [], [float(item[coord_index])])
            raise ValueError('No periodic counterpart found for {} boundary node'.format(label))
        max_distance = max(max_distance, float(best_d))
        pairs.append((first[best_j], item))
    if max_distance > 1.0e-4:
        print(
            '[stage1-fe] periodic {} nearest-node pairing max coordinate mismatch={:.6g}'.format(
                label, max_distance
            ),
            flush=True,
        )
    return pairs


def _validate_periodic_boundary_match(model):
    bottom, right, top, left, corner_candidates = _periodic_boundary_items(model)
    _match_periodic_items(bottom, top, 2, 'bottom/top')
    _match_periodic_items(left, right, 3, 'left/right')
    for key in ('LB', 'RB', 'RT', 'LT'):
        if not corner_candidates[key]:
            raise ValueError('Missing corner node {}'.format(key))


def _cleanup_tie_and_mesh(model):
    assembly = model.rootAssembly
    for name in ('Tie-1', ):
        if name in model.interactions:
            del model.interactions[name]
    for name in ('Master-1', 'Slave-1'):
        if name in assembly.surfaces:
            del assembly.surfaces[name]
    for name in list(model.constraints.keys()):
        if name.startswith('IntTie-'):
            del model.constraints[name]
    for name in list(assembly.sets.keys()):
        if (
            name in ('Common-Surf-1', 'Interface-Matrix-Nodes')
            or name.startswith('Int-M-')
            or name.startswith('Int-I-')
        ):
            del assembly.sets[name]
    for name in ('Common-Surf-1', ):
        if name in assembly.sets:
            del assembly.sets[name]
    try:
        assembly.deleteMesh(regions=(assembly.instances['Part-1-1'], assembly.instances['Part-2-1']))
    except Exception:
        pass


def _partition_last_periodic_mismatch(model):
    mismatch = LAST_PERIODIC_MISMATCH
    if not mismatch:
        return
    coords = mismatch.get('first_only', []) + mismatch.get('second_only', [])
    if not coords:
        return
    assembly = model.rootAssembly
    instances = (assembly.instances['Part-1-1'], assembly.instances['Part-2-1'])
    label = mismatch.get('label')
    if label == 'bottom/top':
        _partition_side_at_coords(assembly, instances, 'bottom', coords)
        _partition_side_at_coords(assembly, instances, 'top', coords)
    elif label == 'left/right':
        _partition_side_at_coords(assembly, instances, 'left', coords)
        _partition_side_at_coords(assembly, instances, 'right', coords)


def _mesh_tie_until_periodic(model, mesh_builder, interface_coupler):
    last_error = None
    for attempt in range(12):
        mesh_builder(model)
        interface_coupler(model)
        try:
            _validate_periodic_boundary_match(model)
            return
        except ValueError as exc:
            last_error = exc
            print('[stage1-fe] periodic validation attempt {} failed: {}'.format(attempt + 1, exc), flush=True)
            _cleanup_tie_and_mesh(model)
            _partition_last_periodic_mismatch(model)
    if last_error is not None:
        raise last_error


def _collect_periodic_boundary_sets(model):
    assembly = model.rootAssembly
    bottom, right, top, left, corner_candidates = _periodic_boundary_items(model)

    def match_pairs(first, second, coord_index, label):
        pairs = []
        max_distance = 0.0

        def nearest(item, candidates):
            best_j = None
            best_d = None
            for j, cand in enumerate(candidates):
                dist = abs(float(item[coord_index]) - float(cand[coord_index]))
                if best_d is None or dist < best_d:
                    best_d = dist
                    best_j = j
            return best_j, best_d

        for item in first:
            best_j, best_d = nearest(item, second)
            if best_j is None:
                raise ValueError('No periodic counterpart found for {} boundary node'.format(label))
            max_distance = max(max_distance, float(best_d))
            pairs.append((item, second[best_j]))
        covered_second = set((pair[1][0], pair[1][1]) for pair in pairs)
        for item in second:
            item_key = (item[0], item[1])
            if item_key in covered_second:
                continue
            best_j, best_d = nearest(item, first)
            if best_j is None:
                raise ValueError('No periodic counterpart found for {} boundary node'.format(label))
            max_distance = max(max_distance, float(best_d))
            pairs.append((first[best_j], item))
        if max_distance > 1.0e-4:
            print(
                '[stage1-fe] periodic {} equation nearest-node pairing max coordinate mismatch={:.6g}'.format(
                    label, max_distance
                ),
                flush=True,
            )
        return pairs

    bottom_top_pairs = match_pairs(bottom, top, 2, 'bottom/top')
    left_right_pairs = match_pairs(left, right, 3, 'left/right')

    for i, pair in enumerate(bottom_top_pairs):
        _make_single_node_set(assembly, 'Edge-AB-{}'.format(i + 1), pair[0])
        _make_single_node_set(assembly, 'Edge-CD-{}'.format(i + 1), pair[1])
    for i, pair in enumerate(left_right_pairs):
        _make_single_node_set(assembly, 'Edge-DA-{}'.format(i + 1), pair[0])
        _make_single_node_set(assembly, 'Edge-BC-{}'.format(i + 1), pair[1])

    for key, set_name in (('LB', 'Vert-1'), ('RB', 'Vert-2'), ('RT', 'Vert-3'), ('LT', 'Vert-4')):
        if not corner_candidates[key]:
            raise ValueError('Missing corner node {}'.format(key))
        _make_single_node_set(assembly, set_name, corner_candidates[key][0])

    return len(bottom_top_pairs), len(left_right_pairs)


def _add_mechanical_periodic_equations(model):
    assembly = model.rootAssembly
    n_bottom_top, n_left_right = _collect_periodic_boundary_sets(model)

    xmax, xmin, ymax, ymin = RVE_BOUNDS
    assembly.ReferencePoint(point=(xmax * 1.05, 0.5 * (ymax + ymin), 0.0))
    rp1_id = assembly.features['RP-1'].id
    assembly.Set(referencePoints=(assembly.referencePoints[rp1_id], ), name='Set-RP-1')

    assembly.ReferencePoint(point=(0.5 * (xmax + xmin), ymax * 1.05, 0.0))
    rp2_id = assembly.features['RP-2'].id
    assembly.Set(referencePoints=(assembly.referencePoints[rp2_id], ), name='Set-RP-2')

    for i in range(n_bottom_top):
        model.Equation(
            name='EdgeEqs-AB-CD-X-{}'.format(i + 1),
            terms=((1.0, 'Edge-CD-{}'.format(i + 1), 1), (-1.0, 'Edge-AB-{}'.format(i + 1), 1),
                   (-1.0, 'Set-RP-2', 1)),
        )
        model.Equation(
            name='EdgeEqs-AB-CD-Y-{}'.format(i + 1),
            terms=((1.0, 'Edge-CD-{}'.format(i + 1), 2), (-1.0, 'Edge-AB-{}'.format(i + 1), 2),
                   (-1.0, 'Set-RP-2', 2)),
        )

    for i in range(n_left_right):
        model.Equation(
            name='EdgeEqs-BC-DA-X-{}'.format(i + 1),
            terms=((1.0, 'Edge-BC-{}'.format(i + 1), 1), (-1.0, 'Edge-DA-{}'.format(i + 1), 1),
                   (-1.0, 'Set-RP-1', 1)),
        )
        model.Equation(
            name='EdgeEqs-BC-DA-Y-{}'.format(i + 1),
            terms=((1.0, 'Edge-BC-{}'.format(i + 1), 2), (-1.0, 'Edge-DA-{}'.format(i + 1), 2),
                   (-1.0, 'Set-RP-1', 2)),
        )

    model.Equation(name='VertEqs-X-1', terms=((1.0, 'Vert-2', 1), (-1.0, 'Vert-1', 1), (-1.0, 'Set-RP-1', 1)))
    model.Equation(name='VertEqs-Y-1', terms=((1.0, 'Vert-2', 2), (-1.0, 'Vert-1', 2), (-1.0, 'Set-RP-1', 2)))
    model.Equation(name='VertEqs-X-2', terms=((1.0, 'Vert-3', 1), (-1.0, 'Vert-4', 1), (-1.0, 'Set-RP-1', 1)))
    model.Equation(name='VertEqs-Y-2', terms=((1.0, 'Vert-3', 2), (-1.0, 'Vert-4', 2), (-1.0, 'Set-RP-1', 2)))
    model.Equation(name='VertEqs-X-3', terms=((1.0, 'Vert-4', 1), (-1.0, 'Vert-1', 1), (-1.0, 'Set-RP-2', 1)))
    model.Equation(name='VertEqs-Y-3', terms=((1.0, 'Vert-4', 2), (-1.0, 'Vert-1', 2), (-1.0, 'Set-RP-2', 2)))


def _faces_region(model):
    assembly = model.rootAssembly
    faces = assembly.instances['Part-1-1'].faces + assembly.instances['Part-2-1'].faces
    return regionToolset.Region(faces=faces)


def _add_thermal_control_nodes(model):
    assembly = model.rootAssembly
    p = model.Part(name='Thermal-Control', dimensionality=TWO_D_PLANAR, type=DEFORMABLE_BODY)
    p.Node(coordinates=(RVE_BOUNDS[0] * 1.10, RVE_BOUNDS[2] * 0.45, 0.0))
    p.Node(coordinates=(RVE_BOUNDS[0] * 1.10, RVE_BOUNDS[2] * 0.55, 0.0))
    p.Set(nodes=p.nodes[0:1], name='Control-X')
    p.Set(nodes=p.nodes[1:2], name='Control-Y')
    assembly.Instance(name='Thermal-Control-1', part=p, dependent=OFF)
    assembly.Set(
        nodes=assembly.instances['Thermal-Control-1'].nodes[0:1],
        name='Set-Temp-Control-X',
    )
    assembly.Set(
        nodes=assembly.instances['Thermal-Control-1'].nodes[1:2],
        name='Set-Temp-Control-Y',
    )


def _add_thermal_periodic_equations(model):
    n_bottom_top, n_left_right = _collect_periodic_boundary_sets(model)
    _add_thermal_control_nodes(model)

    temp_dof = 11
    for i in range(n_bottom_top):
        model.Equation(
            name='TempEqs-AB-CD-{}'.format(i + 1),
            terms=((1.0, 'Edge-CD-{}'.format(i + 1), temp_dof),
                   (-1.0, 'Edge-AB-{}'.format(i + 1), temp_dof),
                   (-1.0, 'Set-Temp-Control-Y', temp_dof)),
        )

    for i in range(n_left_right):
        model.Equation(
            name='TempEqs-BC-DA-{}'.format(i + 1),
            terms=((1.0, 'Edge-BC-{}'.format(i + 1), temp_dof),
                   (-1.0, 'Edge-DA-{}'.format(i + 1), temp_dof),
                   (-1.0, 'Set-Temp-Control-X', temp_dof)),
        )

    model.Equation(
        name='TempVertEqs-1',
        terms=((1.0, 'Vert-2', temp_dof), (-1.0, 'Vert-1', temp_dof),
               (-1.0, 'Set-Temp-Control-X', temp_dof)),
    )
    model.Equation(
        name='TempVertEqs-2',
        terms=((1.0, 'Vert-3', temp_dof), (-1.0, 'Vert-4', temp_dof),
               (-1.0, 'Set-Temp-Control-X', temp_dof)),
    )
    model.Equation(
        name='TempVertEqs-3',
        terms=((1.0, 'Vert-4', temp_dof), (-1.0, 'Vert-1', temp_dof),
               (-1.0, 'Set-Temp-Control-Y', temp_dof)),
    )


def _repair_nan_nodes_in_inp(job_name):
    inp_path = job_name + '.inp'
    last_size = -1
    stable_count = 0
    for _ in range(120):
        if os.path.exists(inp_path):
            size = os.path.getsize(inp_path)
            if size > 0 and size == last_size:
                stable_count += 1
                if stable_count >= 2:
                    break
            else:
                stable_count = 0
                last_size = size
        time.sleep(1.0)
    if not os.path.exists(inp_path):
        return 0

    with open(inp_path, 'r') as f:
        lines = f.readlines()

    coords = {}
    nan_nodes = []
    elements = {}
    current_part = None
    mode = None

    for idx, raw in enumerate(lines):
        line = raw.strip()
        lower = line.lower()
        if not line:
            continue
        if lower.startswith('*part'):
            current_part = None
            for token in line.split(','):
                token = token.strip()
                if token.lower().startswith('name='):
                    current_part = token.split('=', 1)[1]
            coords.setdefault(current_part, {})
            elements.setdefault(current_part, [])
            mode = None
            continue
        if lower.startswith('*end part') or lower.startswith('*assembly'):
            current_part = None
            mode = None
            continue
        if lower.startswith('*instance'):
            for token in line.split(','):
                token = token.strip()
                if token.lower().startswith('part='):
                    current_part = token.split('=', 1)[1]
                    coords.setdefault(current_part, {})
                    elements.setdefault(current_part, [])
                    break
            mode = None
            continue
        if lower.startswith('*end instance'):
            current_part = None
            mode = None
            continue
        if line.startswith('*'):
            if lower.startswith('*node'):
                mode = 'node'
            elif lower.startswith('*element'):
                mode = 'element'
            else:
                mode = None
            continue
        if current_part is None:
            continue
        if mode == 'node':
            parts = [p.strip() for p in line.split(',')]
            if len(parts) < 3:
                continue
            try:
                label = int(parts[0])
            except ValueError:
                continue
            if 'nan' in lower:
                nan_nodes.append((current_part, label, idx))
                continue
            try:
                coords[current_part][label] = (float(parts[1]), float(parts[2]))
            except ValueError:
                pass
        elif mode == 'element':
            parts = [p.strip() for p in line.split(',') if p.strip()]
            if len(parts) < 4:
                continue
            try:
                conn = [int(p) for p in parts[1:]]
            except ValueError:
                continue
            elements.setdefault(current_part, []).append(conn)

    if not nan_nodes:
        return 0

    nan_by_part = {}
    for part, label, _ in nan_nodes:
        nan_by_part.setdefault(part, set()).add(label)

    repaired = 0
    for part, label, idx in nan_nodes:
        samples = []
        for conn in elements.get(part, []):
            if label not in conn:
                continue
            for other in conn:
                if other == label or other in nan_by_part.get(part, set()):
                    continue
                if other in coords.get(part, {}):
                    samples.append(coords[part][other])
        if not samples:
            print('[stage1-fe] could not repair NaN node {} in {}'.format(label, part), flush=True)
            continue
        sx = 0.0
        sy = 0.0
        for px, py in samples:
            sx += px
            sy += py
        x = sx / float(len(samples))
        y = sy / float(len(samples))
        coords.setdefault(part, {})[label] = (x, y)
        lines[idx] = '{:8d}, {:12.6f}, {:12.6f}\n'.format(label, x, y)
        repaired += 1

    if repaired:
        with open(inp_path, 'w') as f:
            f.writelines(lines)
        print('[stage1-fe] repaired {} NaN node coordinates in {}'.format(repaired, inp_path), flush=True)
    if repaired != len(nan_nodes):
        raise ValueError(
            'Unrepaired NaN node coordinates remain in {}: repaired {} of {}'.format(
                inp_path, repaired, len(nan_nodes)
            )
        )
    return repaired


def _parse_inp_tri_mesh(lines):
    data = {}
    current_part = None
    mode = None
    for idx, raw in enumerate(lines):
        line = raw.strip()
        lower = line.lower()
        if not line:
            continue
        if lower.startswith('*part'):
            current_part = None
            for token in line.split(','):
                token = token.strip()
                if token.lower().startswith('name='):
                    current_part = token.split('=', 1)[1]
            data.setdefault(current_part, {'coords': {}, 'node_lines': {}, 'elements': []})
            mode = None
            continue
        if lower.startswith('*end part') or lower.startswith('*assembly'):
            current_part = None
            mode = None
            continue
        if lower.startswith('*instance'):
            for token in line.split(','):
                token = token.strip()
                if token.lower().startswith('part='):
                    current_part = token.split('=', 1)[1]
                    data.setdefault(current_part, {'coords': {}, 'node_lines': {}, 'elements': []})
                    break
            mode = None
            continue
        if lower.startswith('*end instance'):
            current_part = None
            mode = None
            continue
        if line.startswith('*'):
            if lower.startswith('*node'):
                mode = 'node'
            elif lower.startswith('*element'):
                mode = 'element'
            else:
                mode = None
            continue
        if current_part is None:
            continue
        part_data = data.setdefault(current_part, {'coords': {}, 'node_lines': {}, 'elements': []})
        if mode == 'node':
            parts = [p.strip() for p in line.split(',')]
            if len(parts) < 3:
                continue
            try:
                label = int(parts[0])
                x = float(parts[1])
                y = float(parts[2])
            except ValueError:
                continue
            part_data['coords'][label] = (x, y)
            part_data['node_lines'][label] = idx
        elif mode == 'element':
            parts = [p.strip() for p in line.split(',') if p.strip()]
            if len(parts) != 4:
                continue
            try:
                elem_label = int(parts[0])
                conn = [int(p) for p in parts[1:]]
            except ValueError:
                continue
            part_data['elements'].append((elem_label, conn, idx))
    return data


def _triangle_signed_area(coords, conn):
    p1 = coords[conn[0]]
    p2 = coords[conn[1]]
    p3 = coords[conn[2]]
    return 0.5 * ((p2[0] - p1[0]) * (p3[1] - p1[1]) - (p3[0] - p1[0]) * (p2[1] - p1[1]))


def _regularize_triangles_in_inp(job_name):
    inp_path = job_name + '.inp'
    if not os.path.exists(inp_path):
        return 0, 0
    with open(inp_path, 'r') as f:
        lines = f.readlines()

    data = _parse_inp_tri_mesh(lines)
    moved_nodes = 0
    flipped_elements = 0
    duplicate_tol2 = 1.0e-16

    for part, part_data in data.items():
        coords = part_data['coords']
        elements = part_data['elements']
        node_lines = part_data['node_lines']
        neighbors = {}
        for _, conn, _ in elements:
            for node in conn:
                bucket = neighbors.setdefault(node, set())
                for other in conn:
                    if other != node:
                        bucket.add(other)

        for _, conn, _ in elements:
            if any(node not in coords for node in conn):
                continue
            area = _triangle_signed_area(coords, conn)
            if abs(area) > 1.0e-12:
                continue
            duplicate_pairs = []
            for a, b in ((conn[0], conn[1]), (conn[1], conn[2]), (conn[0], conn[2])):
                pa = coords[a]
                pb = coords[b]
                dist2 = (pa[0] - pb[0]) ** 2 + (pa[1] - pb[1]) ** 2
                if dist2 <= duplicate_tol2:
                    duplicate_pairs.append((a, b))
            if not duplicate_pairs:
                continue
            target = max(duplicate_pairs[0])
            px, py = coords[target]
            if (
                abs(px - RVE_BOUNDS[1]) < BOUNDARY_TOL
                or abs(px - RVE_BOUNDS[0]) < BOUNDARY_TOL
                or abs(py - RVE_BOUNDS[3]) < BOUNDARY_TOL
                or abs(py - RVE_BOUNDS[2]) < BOUNDARY_TOL
            ):
                continue
            samples = []
            for other in neighbors.get(target, set()):
                if other not in coords:
                    continue
                ox, oy = coords[other]
                if (ox - px) ** 2 + (oy - py) ** 2 <= duplicate_tol2:
                    continue
                samples.append((ox, oy))
            if not samples:
                continue
            sx = 0.0
            sy = 0.0
            for ox, oy in samples:
                sx += ox
                sy += oy
            nx = sx / float(len(samples))
            ny = sy / float(len(samples))
            coords[target] = (nx, ny)
            lines[node_lines[target]] = '{:8d}, {:12.6f}, {:12.6f}\n'.format(target, nx, ny)
            moved_nodes += 1

    data = _parse_inp_tri_mesh(lines)
    for part, part_data in data.items():
        coords = part_data['coords']
        for elem_label, conn, idx in part_data['elements']:
            if any(node not in coords for node in conn):
                continue
            if _triangle_signed_area(coords, conn) < 0.0:
                fixed = [conn[0], conn[2], conn[1]]
                lines[idx] = '{:8d}, {:8d}, {:8d}, {:8d}\n'.format(elem_label, fixed[0], fixed[1], fixed[2])
                flipped_elements += 1

    if moved_nodes or flipped_elements:
        with open(inp_path, 'w') as f:
            f.writelines(lines)
        print(
            '[stage1-fe] regularized inp triangles in {}: moved_nodes={}, flipped_elements={}'.format(
                inp_path, moved_nodes, flipped_elements
            ),
            flush=True,
        )
    return moved_nodes, flipped_elements


def _submit_job_from_repaired_input(job_name):
    inp_path = job_name + '.inp'
    try:
        del mdb.jobs[job_name]
    except Exception:
        pass
    mdb.JobFromInputFile(
        name=job_name,
        inputFileName=inp_path,
        type=ANALYSIS,
        memory=90,
        memoryUnits=PERCENTAGE,
        getMemoryFromAnalysis=True,
        explicitPrecision=SINGLE,
        nodalOutputPrecision=SINGLE,
        userSubroutine='',
        scratch='',
        resultsFormat=ODB,
        multiprocessingMode=DEFAULT,
        numCpus=1,
        numGPUs=0,
    )
    mdb.jobs[job_name].submit(consistencyChecking=OFF)
    mdb.jobs[job_name].waitForCompletion()


def _build_cte_model(cae_path, job_name, save_path):
    Mdb()
    print('[stage1-fe] openMdb start {}'.format(cae_path), flush=True)
    try:
        openMdb(pathName=cae_path, readOnly=True)
    except TypeError:
        openMdb(pathName=cae_path)
    print('[stage1-fe] openMdb done {}'.format(cae_path), flush=True)
    session.journalOptions.setValues(replayGeometry=INDEX, recoverGeometry=INDEX)
    model = mdb.models['Model-1']
    print('[stage1-fe] assign mechanical materials', flush=True)
    _assign_mechanical_materials(model)
    model.StaticStep(name='Step-CTE', previous='Initial', nlgeom=OFF)
    model.fieldOutputRequests['F-Output-1'].setValues(variables=('S', 'E', 'U', 'EVOL', 'IVOL'))
    print('[stage1-fe] mesh/interface/PBC validation start', flush=True)
    _mesh_tie_until_periodic(model, _seed_and_mesh_mechanical, _couple_phase_interfaces_mechanical)
    print('[stage1-fe] mesh/interface/PBC validation done', flush=True)
    print('[stage1-fe] add mechanical periodic equations', flush=True)
    _add_mechanical_periodic_equations(model)

    assembly = model.rootAssembly
    model.DisplacementBC(
        name='BC-Fix-Origin',
        createStepName='Initial',
        region=assembly.sets['Vert-1'],
        u1=0.0,
        u2=0.0,
        u3=0.0,
        ur1=0.0,
        ur2=0.0,
        ur3=0.0,
        amplitude=UNSET,
        fixed=OFF,
        distributionType=UNIFORM,
        fieldName='',
        localCsys=None,
    )
    # Suppress macro shear and let the two normal macro strains be solved freely.
    model.DisplacementBC(
        name='BC-No-Macro-Shear-1',
        createStepName='Initial',
        region=assembly.sets['Set-RP-1'],
        u2=0.0,
        ur3=0.0,
        amplitude=UNSET,
        fixed=OFF,
        distributionType=UNIFORM,
        fieldName='',
        localCsys=None,
    )
    model.DisplacementBC(
        name='BC-No-Macro-Shear-2',
        createStepName='Initial',
        region=assembly.sets['Set-RP-2'],
        u1=0.0,
        ur3=0.0,
        amplitude=UNSET,
        fixed=OFF,
        distributionType=UNIFORM,
        fieldName='',
        localCsys=None,
    )
    model.Temperature(
        name='Uniform-Delta-T',
        createStepName='Step-CTE',
        region=_faces_region(model),
        distributionType=UNIFORM,
        crossSectionDistribution=CONSTANT_THROUGH_THICKNESS,
        magnitudes=(DELTA_T, ),
    )
    mdb.Job(
        name=job_name,
        model='Model-1',
        description='2D RVE free thermal-expansion homogenization',
        type=ANALYSIS,
        memory=90,
        memoryUnits=PERCENTAGE,
        getMemoryFromAnalysis=True,
        explicitPrecision=SINGLE,
        nodalOutputPrecision=SINGLE,
        echoPrint=OFF,
        modelPrint=OFF,
        contactPrint=OFF,
        historyPrint=OFF,
        resultsFormat=ODB,
        multiprocessingMode=DEFAULT,
        numCpus=1,
        numGPUs=0,
    )
    print('[stage1-fe] write input start {}'.format(job_name), flush=True)
    mdb.jobs[job_name].writeInput(consistencyChecking=OFF)
    _repair_nan_nodes_in_inp(job_name)
    _regularize_triangles_in_inp(job_name)
    print('[stage1-fe] save CAE {}'.format(save_path), flush=True)
    mdb.saveAs(pathName=save_path)
    print('[stage1-fe] submit start {}'.format(job_name), flush=True)
    _submit_job_from_repaired_input(job_name)
    print('[stage1-fe] submit done {}'.format(job_name), flush=True)


def _build_conductivity_model(cae_path, job_name, save_path):
    Mdb()
    try:
        openMdb(pathName=cae_path, readOnly=True)
    except TypeError:
        openMdb(pathName=cae_path)
    session.journalOptions.setValues(replayGeometry=INDEX, recoverGeometry=INDEX)
    model = mdb.models['Model-1']
    _assign_thermal_materials(model)
    model.HeatTransferStep(name='Step-Kx', previous='Initial', response=STEADY_STATE)
    model.HeatTransferStep(name='Step-Ky', previous='Step-Kx', response=STEADY_STATE)
    model.fieldOutputRequests['F-Output-1'].setValues(variables=('NT', 'HFL', 'RFL', 'EVOL', 'IVOL'))
    _mesh_tie_until_periodic(model, _seed_and_mesh_thermal, _couple_phase_interfaces_thermal)
    _add_thermal_periodic_equations(model)

    assembly = model.rootAssembly
    model.TemperatureBC(
        name='BC-Temp-Gauge',
        createStepName='Step-Kx',
        region=assembly.sets['Vert-1'],
        magnitude=0.0,
        amplitude=UNSET,
        fixed=OFF,
        distributionType=UNIFORM,
        fieldName='',
    )
    model.TemperatureBC(
        name='BC-Control-X',
        createStepName='Step-Kx',
        region=assembly.sets['Set-Temp-Control-X'],
        magnitude=DELTA_T,
        amplitude=UNSET,
        fixed=OFF,
        distributionType=UNIFORM,
        fieldName='',
    )
    model.TemperatureBC(
        name='BC-Control-Y',
        createStepName='Step-Kx',
        region=assembly.sets['Set-Temp-Control-Y'],
        magnitude=0.0,
        amplitude=UNSET,
        fixed=OFF,
        distributionType=UNIFORM,
        fieldName='',
    )
    model.boundaryConditions['BC-Control-X'].setValuesInStep(stepName='Step-Ky', magnitude=0.0)
    model.boundaryConditions['BC-Control-Y'].setValuesInStep(stepName='Step-Ky', magnitude=DELTA_T)

    mdb.Job(
        name=job_name,
        model='Model-1',
        description='2D RVE steady heat-conduction homogenization',
        type=ANALYSIS,
        memory=90,
        memoryUnits=PERCENTAGE,
        getMemoryFromAnalysis=True,
        explicitPrecision=SINGLE,
        nodalOutputPrecision=SINGLE,
        echoPrint=OFF,
        modelPrint=OFF,
        contactPrint=OFF,
        historyPrint=OFF,
        resultsFormat=ODB,
        multiprocessingMode=DEFAULT,
        numCpus=1,
        numGPUs=0,
    )
    mdb.jobs[job_name].writeInput(consistencyChecking=OFF)
    _repair_nan_nodes_in_inp(job_name)
    _regularize_triangles_in_inp(job_name)
    mdb.saveAs(pathName=save_path)
    _submit_job_from_repaired_input(job_name)


def main():
    cae_path = RVE_CAE_PATH
    if not os.path.exists(cae_path):
        raise IOError('Cannot find {}'.format(cae_path))

    _build_cte_model(
        cae_path=cae_path,
        job_name='Job-1-CTE',
        save_path='FE_Model_CTE.cae',
    )
    _build_conductivity_model(
        cae_path=cae_path,
        job_name='Job-1-ETC',
        save_path='FE_Model_ETC.cae',
    )
    print('[%s] processed_cae=%s' % (SCRIPT_TAG, cae_path))
    print('[%s] wrote FE_Model_CTE.cae and FE_Model_ETC.cae' % SCRIPT_TAG)


if __name__ == '__main__':
    main()
