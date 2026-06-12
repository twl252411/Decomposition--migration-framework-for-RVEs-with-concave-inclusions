#
from abaqus import *
from abaqusConstants import *
from caeModules import *
from driverUtils import executeOnCaeStartup
from geometry_shape_2d import geometry_shape_2d
from pathlib import Path
import numpy as np

#------------------------------------------------------------------------------------
#
executeOnCaeStartup()
Mdb()

rve_size = 200.0
part_name, path_name = 'Part-1', 'RVE_1.cae'
base_dirs = []
if "__file__" in globals():
    base_dirs.append(Path(__file__).resolve().parents[1])
cwd = Path.cwd().resolve()
base_dirs.extend([cwd, cwd.parent])
data_dir = None
for base_dir in base_dirs:
    candidate = base_dir / "intermediate_final_files"
    if candidate.exists():
        data_dir = candidate
        break
if data_dir is None:
    data_dir = (base_dirs[0] if base_dirs else cwd) / "intermediate_final_files"
inc_type = ["lobular2", "pea", "concave_poly"][1]
file = data_dir / f"{inc_type}_polygon_positions_after_periodic.txt"
data = np.loadtxt(file)
user_vars = {}

#-----------------------------------------Circles------------------------------------------
#
if inc_type == "lobular2":
    radius = 4.17284
    user_vars.update({'radius':radius})
    part = geometry_shape_2d(user_vars).lobular2(part_name)
elif inc_type == "pea":
    inc_size = [9.84, 8.2, 12.3, 1.4]
    user_vars.update({'R1':inc_size[0], 'R2':inc_size[1],
                      'K1':inc_size[2], 'a':inc_size[3]})
    part = geometry_shape_2d(user_vars).pea_shape(part_name)
else:
    file = data_dir / "vertices_concave_poly.txt"
    vertices = np.loadtxt(file)
    user_vars.update({'vertices':vertices})
    part = geometry_shape_2d(user_vars).concave_poly(part_name)
#
#------------------------------------------------------------------------------------
#
p = mdb.models['Model-1'].parts['Part-1']
dic = p.getMassProperties( )
V1 = dic['area']

#------------------------------------------------------------------------------------
#
tran_vecs = data[:, 0:2]
angles = data[:, 2] * 180 / np.pi

#------------------------------------------------------------------------------------
#
ins_list = ()
for inc in range(len(tran_vecs)):
    a = mdb.models['Model-1'].rootAssembly
    p = mdb.models['Model-1'].parts['Part-1']
    ins_name= 'Part-1-' + str(inc+1)
    a.Instance(name=ins_name, part=p, dependent=OFF)
    a.rotate(instanceList=(ins_name, ), axisPoint=(0.0, 0.0, 0.0), axisDirection=(0.0, 0.0, 1.0), angle=angles[inc])
    a.translate(instanceList=(ins_name, ), vector=(tran_vecs[inc][0], tran_vecs[inc][1], 0.0))
    ins_list = ins_list + ((a.instances[ins_name], ))
#
a = mdb.models['Model-1'].rootAssembly
a.InstanceFromBooleanMerge(name='Part-2', instances=ins_list, originalInstances=DELETE, domain=GEOMETRY)
del a.instances['Part-2-1']
del mdb.models['Model-1'].parts['Part-1']
#
p = mdb.models['Model-1'].parts['Part-2']
volume = p.getMassProperties( )['area']
#
a = mdb.models['Model-1'].rootAssembly
p1 = mdb.models['Model-1'].parts['Part-2']
a.Instance(name='Part-2-1', part=p1, dependent=OFF)
a.Instance(name='Part-2-2', part=p1, dependent=OFF)
#
#------------------------------------------------------------------------------------
#
s = mdb.models['Model-1'].ConstrainedSketch(name='__profile__', sheetSize=200.0)
g, v, d, c = s.geometry, s.vertices, s.dimensions, s.constraints
s.setPrimaryObject(option=STANDALONE)
s.rectangle(point1=(0.0, 0.0), point2=(rve_size, rve_size))
p = mdb.models['Model-1'].Part(name='Part-1', dimensionality=TWO_D_PLANAR,
    type=DEFORMABLE_BODY)
p.BaseShell(sketch=s)
s.unsetPrimaryObject()
del mdb.models['Model-1'].sketches['__profile__']
#
a = mdb.models['Model-1'].rootAssembly
p = mdb.models['Model-1'].parts['Part-1']
a.Instance(name='Part-1-1', part=p, dependent=OFF)
a.Instance(name='Part-1-2', part=p, dependent=OFF)
#
a = mdb.models['Model-1'].rootAssembly
a.InstanceFromBooleanCut(name='Part-3',
    instanceToBeCut=mdb.models['Model-1'].rootAssembly.instances['Part-2-2'],
    cuttingInstances=(a.instances['Part-1-1'], ), originalInstances=DELETE)
a.InstanceFromBooleanCut(name='Part-4',
    instanceToBeCut=mdb.models['Model-1'].rootAssembly.instances['Part-2-1'],
    cuttingInstances=(a.instances['Part-3-1'], ), originalInstances=DELETE)
#
del mdb.models['Model-1'].parts['Part-2']
del mdb.models['Model-1'].parts['Part-3']
a = mdb.models['Model-1'].rootAssembly
a.deleteFeatures(('Part-4-1',))
mdb.models['Model-1'].parts.changeKey(fromName='Part-4', toName='Part-2')
p = mdb.models['Model-1'].parts['Part-2']
a.Instance(name='Part-2-1', part=p, dependent=OFF)
#
a = mdb.models['Model-1'].rootAssembly
a.InstanceFromBooleanCut(name='Part-3',
    instanceToBeCut=mdb.models['Model-1'].rootAssembly.instances['Part-1-2'],
    cuttingInstances=(a.instances['Part-2-1'], ), originalInstances=DELETE)
a.deleteFeatures(('Part-3-1',))
del mdb.models['Model-1'].parts['Part-1']
mdb.models['Model-1'].parts.changeKey(fromName='Part-3', toName='Part-1')
#
a = mdb.models['Model-1'].rootAssembly
p = mdb.models['Model-1'].parts['Part-1']
a.Instance(name='Part-1-1', part=p, dependent=OFF)
p = mdb.models['Model-1'].parts['Part-2']
a.Instance(name='Part-2-1', part=p, dependent=OFF)

#------------------------------------------------------------------------------------
#
p = mdb.models['Model-1'].parts['Part-1']
volume1 = p.getMassProperties( )['area']
p = mdb.models['Model-1'].parts['Part-2']
volume2 = p.getMassProperties( )['area']
print (f'Volume fraction: {volume2/(volume2+volume1)}')

#
#-------------------------------------------------------------------------------------------------------
#
mdb.saveAs(pathName=path_name)

# -------------------------------------------------------------------------
# Cleanup: remove files in abaqus_rve except .py, .cae, .jnl
keep_suffixes = {'.py', '.cae', '.jnl'}
cleanup_dir = Path(__file__).resolve().parent if "__file__" in globals() else Path.cwd().resolve()
for entry in cleanup_dir.iterdir():
    if entry.is_file() and entry.suffix not in keep_suffixes:
        try:
            entry.unlink()
        except Exception:
            pass
