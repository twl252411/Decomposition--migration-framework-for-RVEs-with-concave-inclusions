# Single-RVE Abaqus/CAE preprocessing script for elastic homogenization.
# Expected phase names: Part-1 matrix, Part-2 inclusion.
from abaqus import *
from abaqusConstants import *
from caeModules import *
from driverUtils import executeOnCaeStartup
from pathlib import Path
import os
import traceback

SCRIPT_TAG = "fe_homogenization_elastic_pre"
RVE_CAE_PATH = Path(
    os.environ.get("FE_HOMOG_ELASTIC_CAE_FILE", os.environ.get("FE_HOMOG_CAE_FILE", "RVE_1.cae"))
).expanduser().resolve()
OUTPUT_DIR = Path(
    os.environ.get("FE_HOMOG_ELASTIC_OUTPUT_DIR", os.environ.get("FE_HOMOG_OUTPUT_DIR", "."))
).expanduser().resolve()
BOUNDARY_TOL = float(
    os.environ.get("FE_HOMOG_ELASTIC_BOUNDARY_TOL", os.environ.get("FE_HOMOG_BOUNDARY_TOL", "1.0E-3"))
)
MATCH_TOL = float(
    os.environ.get("FE_HOMOG_ELASTIC_MATCH_TOL", os.environ.get("FE_HOMOG_MATCH_TOL", "1.0E-6"))
)
MESH_SIZE = float(
    os.environ.get("FE_HOMOG_ELASTIC_MESH_SIZE", os.environ.get("FE_HOMOG_MESH_SIZE", "1.0"))
)
JOB_NAME = os.environ.get("FE_HOMOG_ELASTIC_JOB_NAME", os.environ.get("FE_HOMOG_JOB_NAME", RVE_CAE_PATH.stem))


def _matrix_bounds_from_part(part):
    coords = []
    try:
        vertices = part.vertices
        for vertex in vertices:
            pt = vertex.pointOn[0]
            coords.append((float(pt[0]), float(pt[1])))
    except Exception:
        coords = []
    if len(coords) < 4:
        coords = []
        try:
            edges = part.edges
            for edge in edges:
                pt = edge.pointOn[0]
                coords.append((float(pt[0]), float(pt[1])))
        except Exception:
            coords = []
    if not coords:
        raise ValueError("Unable to determine matrix bounds from Part-1.")
    xs = [xy[0] for xy in coords]
    ys = [xy[1] for xy in coords]
    return [max(xs), min(xs), max(ys), min(ys)]


def _write_failure_report(report_path, cae_path, exc):
    report_lines = [
        "status\tfailed",
        "source_cae\t%s" % cae_path,
        "error_type\t%s" % exc.__class__.__name__,
        "error_message\t%s" % exc,
        "",
        "[traceback]",
        traceback.format_exc().strip(),
        "",
    ]
    with open(report_path, "w") as fp:
        fp.write("\n".join(report_lines))


def _save_preprocessed_cae(pathname):
    mdb.saveAs(pathName=str(pathname))


executeOnCaeStartup()
#
#------------------------------------------------------------------------------------
#
caename = RVE_CAE_PATH
boundary_tol = BOUNDARY_TOL
match_tol = MATCH_TOL
mesh_size = MESH_SIZE
if not caename.is_file():
    raise IOError("Input .cae file not found: %s" % caename)
if caename.is_file():
    #
    Mdb()
    #
    session.journalOptions.setValues(replayGeometry=INDEX, recoverGeometry=INDEX)
    #
    output_dir = OUTPUT_DIR
    output_dir.mkdir(parents=True, exist_ok=True)
    pathname = output_dir / (caename.stem + '.inp')
    jobname = JOB_NAME
    preprocessed_cae = output_dir / (caename.stem + '_preprocessed.cae')
    failed_cae = output_dir / (caename.stem + '_preprocessed_failed.cae')
    failure_report = output_dir / (caename.stem + '_preprocessing_failed.txt')
    old_cwd = os.getcwd()
    try:
        #
        openMdb(pathName=str(caename))
        model = mdb.models['Model-1']
        rs = _matrix_bounds_from_part(model.parts['Part-1'])
        RLength = float(rs[0] - rs[1])
        mshsize = mesh_size
        disp = [(rs[0]-rs[1])*1.0, (rs[2]-rs[3])*1.0]
        #
        #-------------------------------------------------------------------------------------
        #
        model.Material(name='Material-1')
        model.materials['Material-1'].Elastic(table=((0.00335, 0.35), ))
        model.HomogeneousSolidSection(name='Section-1', material='Material-1', thickness=None)
        p = model.parts['Part-1']
        region = regionToolset.Region(faces=p.faces)
        p.SectionAssignment(region=region, sectionName='Section-1', offset=0.0, offsetType=MIDDLE_SURFACE,
            offsetField='', thicknessAssignment=FROM_SECTION)
        #
        model.Material(name='Material-2')
        model.materials['Material-2'].Elastic(table=((0.074, 0.2), ))
        model.HomogeneousSolidSection(name='Section-2', material='Material-2', thickness=None)
        p = model.parts['Part-2']
        region = regionToolset.Region(faces=p.faces)
        p.SectionAssignment(region=region, sectionName='Section-2', offset=0.0, offsetType=MIDDLE_SURFACE,
            offsetField='', thicknessAssignment=FROM_SECTION)
        #
        #-------------------------------------------------------------------------------------
        #
        mdb.models['Model-1'].StaticLinearPerturbationStep(name='Step-1', previous='Initial')
        mdb.models['Model-1'].fieldOutputRequests['F-Output-1'].setValues(variables=('S', 'E', 'U', 'EVOL', 'IVOL'))
        mdb.models['Model-1'].StaticLinearPerturbationStep(name='Step-2', previous='Step-1')
        mdb.models['Model-1'].StaticLinearPerturbationStep(name='Step-3', previous='Step-2')
        #
        #-------------------------------------------------------------------------------------
        #
        a = mdb.models['Model-1'].rootAssembly
        partInstances =(a.instances['Part-1-1'], a.instances['Part-2-1'], )
        a.seedPartInstance(regions=partInstances, size=mshsize, deviationFactor=0.1, minSizeFactor=0.1)
        #
        a = mdb.models['Model-1'].rootAssembly
        alledges = a.instances['Part-1-1'].edges
        e = alledges[0:0]
        for iedge in alledges:
            pt = iedge.pointOn[0]
            if abs(pt[1]-rs[3]) < boundary_tol or abs(pt[0]-rs[0]) < boundary_tol or abs(pt[1]-rs[2]) < boundary_tol or abs(pt[0]-rs[1]) < boundary_tol:
                e += alledges[iedge.index:iedge.index+1]
        a.seedEdgeBySize(edges=e, size=mshsize, deviationFactor=0.1, minSizeFactor=0.1, constraint=FIXED)
        #
        a = mdb.models['Model-1'].rootAssembly
        alledges = a.instances['Part-2-1'].edges
        e = alledges[0:0]
        for iedge in alledges:
            pt = iedge.pointOn[0]
            if abs(pt[1]-rs[3]) < boundary_tol or abs(pt[0]-rs[0]) < boundary_tol or abs(pt[1]-rs[2]) < boundary_tol or abs(pt[0]-rs[1]) < boundary_tol:
                e += alledges[iedge.index:iedge.index+1]
        a.seedEdgeBySize(edges=e, size=mshsize, deviationFactor=0.1, minSizeFactor=0.1, constraint=FIXED)
        #
        a = mdb.models['Model-1'].rootAssembly
        f = a.instances['Part-1-1'].faces + a.instances['Part-2-1'].faces
        a.setMeshControls(regions=f, elemShape=TRI)
        elemType1 = mesh.ElemType(elemCode=CPE4, elemLibrary=STANDARD)
        elemType2 = mesh.ElemType(elemCode=CPE3, elemLibrary=STANDARD)
        a.setElementType(regions=(f, ), elemTypes=(elemType1, elemType2))
        a.generateMesh(regions=(a.instances['Part-1-1'],a.instances['Part-2-1'],))
        #
        #-------------------------------------------------------------------------------------
        #
        a = mdb.models['Model-1'].rootAssembly
        alledges = a.instances['Part-1-1'].edges
        selectedges = alledges[0:0]
        #
        for iedge in alledges:
            pt = iedge.pointOn[0]
            if abs(pt[0]-rs[0]) > boundary_tol and abs(pt[0]-rs[1]) > boundary_tol and abs(pt[1]-rs[2]) > boundary_tol and abs(pt[1]-rs[3]) > boundary_tol:
                selectedges += alledges[iedge.index:iedge.index+1]
        #
        a = mdb.models['Model-1'].rootAssembly
        a.Surface(side1Edges=selectedges, name='Master-1')
        a.Set(edges=selectedges, name='Common-Surf-1')
        #
        a = mdb.models['Model-1'].rootAssembly
        alledges = a.instances['Part-2-1'].edges
        selectedges = alledges[0:0]
        for iedge in alledges:
            pt = iedge.pointOn[0]
            if abs(pt[0]-rs[0]) > boundary_tol and abs(pt[0]-rs[1]) > boundary_tol and abs(pt[1]-rs[2]) > boundary_tol and abs(pt[1]-rs[3]) > boundary_tol:
                selectedges += alledges[iedge.index:iedge.index+1]
        #
        a = mdb.models['Model-1'].rootAssembly
        a.Surface(side1Edges=selectedges, name='Slave-1')
        #
        a = mdb.models['Model-1'].rootAssembly
        region1, region2 = a.surfaces['Master-1'], a.surfaces['Slave-1']
        mdb.models['Model-1'].Tie(name='Tie-1', main=region2, secondary=region1, positionToleranceMethod=COMPUTED, adjust=OFF, 
            tieRotations=OFF, thickness=ON)
        #
        #-------------------------------------------------------------------------------------
        #
        a = mdb.models['Model-1'].rootAssembly
        allnodes = a.instances['Part-1-1'].nodes
        setnds = a.sets['Common-Surf-1'].nodes
        selcnds = setnds[0:0]
        for isetn in setnds:
            pt = isetn.coordinates
            if abs(pt[0]-rs[0]) < boundary_tol or abs(pt[0]-rs[1]) < boundary_tol or abs(pt[1]-rs[2]) < boundary_tol or abs(pt[1]-rs[3]) < boundary_tol:
                selcnds += allnodes[isetn.label-1:isetn.label]
        #
        a = mdb.models['Model-1'].rootAssembly
        allnodes = a.instances['Part-1-1'].nodes
        EABnds, EBCnds, ECDnds, EDAnds = allnodes[0:0], allnodes[0:0], allnodes[0:0], allnodes[0:0]
        #
        for inode in allnodes:
            #
            pt = inode.coordinates
            #
            if abs(pt[1]-rs[3]) < boundary_tol and abs(pt[0]-rs[0]) > boundary_tol and abs(pt[0]-rs[1]) > boundary_tol:
                if inode not in selcnds:
                    EABnds += allnodes[inode.label-1:inode.label]
            if abs(pt[0]-rs[0]) < boundary_tol and abs(pt[1]-rs[2]) > boundary_tol and abs(pt[1]-rs[3]) > boundary_tol:
                if inode not in selcnds:
                    EBCnds += allnodes[inode.label-1:inode.label]
            if abs(pt[1]-rs[2]) < boundary_tol and abs(pt[0]-rs[0]) > boundary_tol and abs(pt[0]-rs[1]) > boundary_tol:
                if inode not in selcnds:
                    ECDnds += allnodes[inode.label-1:inode.label]
            if abs(pt[0]-rs[1]) < boundary_tol and abs(pt[1]-rs[2]) > boundary_tol and abs(pt[1]-rs[3]) > boundary_tol:
                if inode not in selcnds:
                    EDAnds += allnodes[inode.label-1:inode.label]
        #
        TmpCoords = []
        for i in range(len(EABnds)):
            TmpCoords += [EABnds[i].coordinates[0]]
        for i in range(0,len(EABnds)):
            for j in range(len(EABnds)-1,i,-1):
                if TmpCoords[j] < TmpCoords[j-1]:
                    Temp = TmpCoords[j-1]
                    TmpCoords[j-1] = TmpCoords[j]
                    TmpCoords[j] = Temp
        for i in range(0,len(EABnds)):
            for j in range(0,len(EABnds)):
                if abs(TmpCoords[i]-EABnds[j].coordinates[0]) < match_tol:
                    node = (allnodes[EABnds[j].label-1:EABnds[j].label], )
                    a = mdb.models['Model-1'].rootAssembly
                    a.Set(nodes=node, name='Edge-AB-' + str(i+1))
        #
        TmpCoords = []
        for i in range(len(ECDnds)):
            TmpCoords += [ECDnds[i].coordinates[0]]
        for i in range(0,len(ECDnds)):
            for j in range(len(ECDnds)-1,i,-1):
                if TmpCoords[j] < TmpCoords[j-1]:
                    Temp = TmpCoords[j-1]
                    TmpCoords[j-1] = TmpCoords[j]
                    TmpCoords[j] = Temp
        for i in range(0,len(ECDnds)):
            for j in range(0,len(ECDnds)):
                if abs(TmpCoords[i]-ECDnds[j].coordinates[0]) < match_tol:
                    node = (allnodes[ECDnds[j].label-1:ECDnds[j].label], )
                    a = mdb.models['Model-1'].rootAssembly
                    a.Set(nodes=node, name='Edge-CD-' + str(i+1))
        #
        TmpCoords = []
        for i in range(len(EBCnds)):
            TmpCoords += [EBCnds[i].coordinates[1]]
        for i in range(0,len(EBCnds)):
            for j in range(len(EBCnds)-1,i,-1):
                if TmpCoords[j] < TmpCoords[j-1]:
                    Temp = TmpCoords[j-1]
                    TmpCoords[j-1] = TmpCoords[j]
                    TmpCoords[j] = Temp
        for i in range(0,len(EBCnds)):
            for j in range(0,len(EBCnds)):
                if abs(TmpCoords[i]-EBCnds[j].coordinates[1]) < match_tol:
                    node = (allnodes[EBCnds[j].label-1:EBCnds[j].label], )
                    a = mdb.models['Model-1'].rootAssembly
                    a.Set(nodes=node, name='Edge-BC-' + str(i+1))
        #
        TmpCoords = []
        for i in range(len(EDAnds)):
            TmpCoords += [EDAnds[i].coordinates[1]]
        for i in range(0,len(EDAnds)):
            for j in range(len(EDAnds)-1,i,-1):
                if TmpCoords[j] < TmpCoords[j-1]:
                    Temp = TmpCoords[j-1]
                    TmpCoords[j-1] = TmpCoords[j]
                    TmpCoords[j] = Temp
        for i in range(0,len(EDAnds)):
            for j in range(0,len(EDAnds)):
                if abs(TmpCoords[i]-EDAnds[j].coordinates[1]) < match_tol:
                    node = (allnodes[EDAnds[j].label-1:EDAnds[j].label], )
                    a = mdb.models['Model-1'].rootAssembly
                    a.Set(nodes=node, name='Edge-DA-' + str(i+1))
        #
        #-------------------------------------------------------------------------------------
        #
        a = mdb.models['Model-1'].rootAssembly
        allnodes = a.instances['Part-2-1'].nodes
        EABnds1, EBCnds1, ECDnds1, EDAnds1 = allnodes[0:0], allnodes[0:0], allnodes[0:0], allnodes[0:0]
        #
        for inode in allnodes:
            pt = inode.coordinates
            #
            if abs(pt[1]-rs[3]) < boundary_tol and abs(pt[0]-rs[0]) > boundary_tol and abs(pt[0]-rs[1]) > boundary_tol:
                EABnds1 += allnodes[inode.label-1:inode.label]
            if abs(pt[0]-rs[0]) < boundary_tol and abs(pt[1]-rs[2]) > boundary_tol and abs(pt[1]-rs[3]) > boundary_tol:
                EBCnds1 += allnodes[inode.label-1:inode.label]
            if abs(pt[1]-rs[2]) < boundary_tol and abs(pt[0]-rs[0]) > boundary_tol and abs(pt[0]-rs[1]) > boundary_tol:
                ECDnds1 += allnodes[inode.label-1:inode.label]
            if abs(pt[0]-rs[1]) < boundary_tol and abs(pt[1]-rs[2]) > boundary_tol and abs(pt[1]-rs[3]) > boundary_tol:
                EDAnds1 += allnodes[inode.label-1:inode.label]
        #
        TmpCoords = []
        for i in range(len(EABnds1)):
            TmpCoords += [EABnds1[i].coordinates[0]]
        for i in range(0,len(EABnds1)):
            for j in range(len(EABnds1)-1,i,-1):
                if TmpCoords[j] < TmpCoords[j-1]:
                    Temp = TmpCoords[j-1]
                    TmpCoords[j-1] = TmpCoords[j]
                    TmpCoords[j] = Temp	
        for i in range(0,len(EABnds1)):
            for j in range(0,len(EABnds1)):
                if abs(TmpCoords[i]-EABnds1[j].coordinates[0]) < match_tol:
                    node = (allnodes[EABnds1[j].label-1:EABnds1[j].label], )
                    a = mdb.models['Model-1'].rootAssembly
                    a.Set(nodes=node, name='Edge-AB-' + str(i+1+len(EABnds)))
        #
        TmpCoords = []
        for i in range(len(ECDnds1)):
            TmpCoords += [ECDnds1[i].coordinates[0]]
        for i in range(0,len(ECDnds1)):
            for j in range(len(ECDnds1)-1,i,-1):
                if TmpCoords[j] < TmpCoords[j-1]:
                    Temp = TmpCoords[j-1]
                    TmpCoords[j-1] = TmpCoords[j]
                    TmpCoords[j] = Temp	
        for i in range(0,len(ECDnds1)):
            for j in range(0,len(ECDnds1)):
                if abs(TmpCoords[i]-ECDnds1[j].coordinates[0]) < match_tol:
                    node = (allnodes[ECDnds1[j].label-1:ECDnds1[j].label], )
                    a = mdb.models['Model-1'].rootAssembly
                    a.Set(nodes=node, name='Edge-CD-' + str(i+1+len(ECDnds)))
        #
        TmpCoords = []
        for i in range(len(EBCnds1)):
            TmpCoords += [EBCnds1[i].coordinates[1]]
        for i in range(0,len(EBCnds1)):
            for j in range(len(EBCnds1)-1,i,-1):
                if TmpCoords[j] < TmpCoords[j-1]:
                    Temp = TmpCoords[j-1]
                    TmpCoords[j-1] = TmpCoords[j]
                    TmpCoords[j] = Temp
        for i in range(0,len(EBCnds1)):
            for j in range(0,len(EBCnds1)):
                if abs(TmpCoords[i]-EBCnds1[j].coordinates[1]) < match_tol:
                    node = (allnodes[EBCnds1[j].label-1:EBCnds1[j].label], )
                    a = mdb.models['Model-1'].rootAssembly
                    a.Set(nodes=node, name='Edge-BC-' + str(i+1+len(EBCnds)))
        #
        TmpCoords = []
        for i in range(len(EDAnds1)):
            TmpCoords += [EDAnds1[i].coordinates[1]]
        for i in range(0,len(EDAnds1)):
            for j in range(len(EDAnds1)-1,i,-1):
                if TmpCoords[j] < TmpCoords[j-1]:
                    Temp = TmpCoords[j-1]
                    TmpCoords[j-1] = TmpCoords[j]
                    TmpCoords[j] = Temp
        for i in range(0,len(EDAnds1)):
            for j in range(0,len(EDAnds1)):
                if abs(TmpCoords[i]-EDAnds1[j].coordinates[1]) < match_tol:
                    node = (allnodes[EDAnds1[j].label-1:EDAnds1[j].label], )
                    a = mdb.models['Model-1'].rootAssembly
                    a.Set(nodes=node, name='Edge-DA-' + str(i+1+len(EDAnds)))
        #
        #-------------------------------------------------------------------------------------
        #
        a = mdb.models['Model-1'].rootAssembly
        allnodes = a.instances['Part-1-1'].nodes
        Vertnd1, Vertnd2, Vertnd3, Vertnd4 = allnodes[0:0], allnodes[0:0], allnodes[0:0], allnodes[0:0]
        #
        for i in range(2):
            instname = 'Part-'+str(i+1)+'-1'
            a = mdb.models['Model-1'].rootAssembly
            allnodes = a.instances[instname].nodes
            #
            for inode in allnodes:
                pt = inode.coordinates
                #
                if abs(pt[0]-rs[1]) < boundary_tol and abs(pt[1]-rs[3]) < boundary_tol:
                    Vertnd1 += allnodes[inode.label-1:inode.label]
                if abs(pt[0]-rs[0]) < boundary_tol and abs(pt[1]-rs[3]) < boundary_tol:
                    Vertnd2 += allnodes[inode.label-1:inode.label]
                if abs(pt[0]-rs[0]) < boundary_tol and abs(pt[1]-rs[2]) < boundary_tol:
                    Vertnd3 += allnodes[inode.label-1:inode.label]
                if abs(pt[0]-rs[1]) < boundary_tol and abs(pt[1]-rs[2]) < boundary_tol:
                    Vertnd4 += allnodes[inode.label-1:inode.label]
        #
        for i in range(4):
            if i == 0:
                node = (Vertnd1[0:1],)
            if i == 1:
                node = (Vertnd2[0:1],)
            if i == 2:
                node = (Vertnd3[0:1],)
            if i == 3:
                node = (Vertnd4[0:1],)
            #
            a = mdb.models['Model-1'].rootAssembly
            a.Set(nodes=node, name='Vert-'+str(i+1))
        #
        #-------------------------------------------------------------------------------------
        #
        a = mdb.models['Model-1'].rootAssembly
        a.ReferencePoint(point=(rs[0]*1.05, (rs[2]+rs[3])/2.0, 0.0))
        iD = a.features['RP-1'].id
        a.Set(referencePoints=(a.referencePoints[iD], ), name='Set-RP-1')
        #
        a = mdb.models['Model-1'].rootAssembly
        a.ReferencePoint(point=((rs[0]+rs[1])/2.0, rs[2]*1.05, 0.0))
        iD = a.features['RP-2'].id
        a.Set(referencePoints=(a.referencePoints[iD], ), name='Set-RP-2')
        #
        numset = max(len(EABnds)+len(EABnds1),len(ECDnds)+len(ECDnds1))
        for i in range(numset):
            setname1, setname2 = 'Edge-CD-'+str(i+1), 'Edge-AB-'+str(i+1)
            xconst = 'EdgeEqs-AB-CD-X-'+str(i+1)
            yconst = 'EdgeEqs-AB-CD-Y-'+str(i+1)
            mdb.models['Model-1'].Equation(name=xconst, terms=((1.0, setname1, 1), (-1.0, setname2, 1), 
                (-1.0, 'Set-RP-2', 1)))
            mdb.models['Model-1'].Equation(name=yconst, terms=((1.0, setname1, 2), (-1.0, setname2, 2), 
                (-1.0, 'Set-RP-2', 2))) 
        #
        numset = max(len(EDAnds)+len(EDAnds1), len(EBCnds)+len(EBCnds1))
        for i in range(numset):
            setname1, setname2 = 'Edge-BC-'+str(i+1), 'Edge-DA-'+str(i+1)
            xconst='EdgeEqs-BC-DA-X-'+str(i+1)
            yconst='EdgeEqs-BC-DA-Y-'+str(i+1)
            mdb.models['Model-1'].Equation(name=xconst, terms=((1.0, setname1, 1), (-1.0, setname2, 1), 
                (-1.0, 'Set-RP-1', 1)))
            mdb.models['Model-1'].Equation(name=yconst, terms=((1.0, setname1, 2), (-1.0, setname2, 2), 
                (-1.0, 'Set-RP-1', 2)))
        #
        mdb.models['Model-1'].Equation(name='VertEqs-X-1', terms=((1.0, 'Vert-2', 1), (-1.0, 'Vert-1', 1),
            (-1.0, 'Set-RP-1', 1)))
        mdb.models['Model-1'].Equation(name='VertEqs-Y-1', terms=((1.0, 'Vert-2', 2), (-1.0, 'Vert-1', 2), 
            (-1.0, 'Set-RP-1', 2)))
        #
        mdb.models['Model-1'].Equation(name='VertEqs-X-2', terms=((1.0, 'Vert-3', 1), (-1.0, 'Vert-4', 1), 
            (-1.0, 'Set-RP-1', 1)))
        mdb.models['Model-1'].Equation(name='VertEqs-Y-2', terms=((1.0, 'Vert-3', 2), (-1.0, 'Vert-4', 2),
            (-1.0, 'Set-RP-1', 2)))
        #                        
        mdb.models['Model-1'].Equation(name='VertEqs-X-3', terms=((1.0, 'Vert-4', 1), (-1.0, 'Vert-1', 1), 
            (-1.0, 'Set-RP-2', 1)))
        mdb.models['Model-1'].Equation(name='VertEqs-Y-3', terms=((1.0, 'Vert-4', 2), (-1.0, 'Vert-1', 2), 
            (-1.0, 'Set-RP-2', 2)))
        #
        #-------------------------------------------------------------------------------------
        # 
        a = mdb.models['Model-1'].rootAssembly
        region1, region2, region3 = a.sets['Set-RP-1'], a.sets['Set-RP-2'], a.sets['Vert-1']
        #
        mdb.models['Model-1'].DisplacementBC(name='BC-0', createStepName='Initial', region=region3, u1=0.0, u2=0.0, u3=0.0, ur1=0.0, 
            ur2=0.0, ur3=0.0, amplitude=UNSET, fixed=OFF, distributionType=UNIFORM, fieldName='', localCsys=None)
        #
        mdb.models['Model-1'].DisplacementBC(name='BC-1', createStepName='Step-1', region=region1, u1=disp[0], u2=0.0, ur3=0.0, 
            amplitude=UNSET, fixed=OFF, distributionType=UNIFORM, fieldName='', localCsys=None)   
        mdb.models['Model-1'].DisplacementBC(name='BC-2', createStepName='Step-1', region=region2, u1=0.0, u2=UNSET, ur3=0.0,
            amplitude=UNSET, fixed=OFF, distributionType=UNIFORM, fieldName='', localCsys=None)
        #    
        mdb.models['Model-1'].DisplacementBC(name='BC-3', createStepName='Step-2', region=region1, u1=UNSET, u2=0.0, ur3=0.0, 
            amplitude=UNSET, fixed=OFF, distributionType=UNIFORM, fieldName='', localCsys=None)   
        mdb.models['Model-1'].DisplacementBC(name='BC-4', createStepName='Step-2', region=region2, u1=0.0, u2=disp[1], ur3=0.0,
            amplitude=UNSET, fixed=OFF, distributionType=UNIFORM, fieldName='', localCsys=None)
        #    
        mdb.models['Model-1'].DisplacementBC(name='BC-5', createStepName='Step-3', region=region1, u1=0.0, u2=disp[0]/2.0, ur3=0.0, 
            amplitude=UNSET, fixed=OFF, distributionType=UNIFORM, fieldName='', localCsys=None)   
        mdb.models['Model-1'].DisplacementBC(name='BC-6', createStepName='Step-3', region=region2, u1=disp[1]/2.0, u2=0.0, ur3=0.0,
            amplitude=UNSET, fixed=OFF, distributionType=UNIFORM, fieldName='', localCsys=None)
        #
        #-------------------------------------------------------------------------------------
        #
        mdb.Job(name=jobname, model='Model-1', description='', type=ANALYSIS, atTime=None, waitMinutes=0, waitHours=0, queue=None, 
            memory=90, memoryUnits=PERCENTAGE, getMemoryFromAnalysis=True, explicitPrecision=SINGLE, nodalOutputPrecision=SINGLE, 
            echoPrint=OFF, modelPrint=OFF, contactPrint=OFF, historyPrint=OFF, userSubroutine='', scratch='', resultsFormat=ODB, 
            multiprocessingMode=DEFAULT, numCpus=1, numGPUs=0)
        os.chdir(str(output_dir))
        mdb.jobs[jobname].writeInput(consistencyChecking=OFF)
        _save_preprocessed_cae(preprocessed_cae)
        if failure_report.exists():
            failure_report.unlink()
        print("[%s] wrote %s" % (SCRIPT_TAG, pathname))
        print("[%s] saved %s" % (SCRIPT_TAG, preprocessed_cae))
    except Exception as exc:
        save_error = None
        try:
            _save_preprocessed_cae(failed_cae)
        except Exception as save_exc:
            save_error = save_exc
        _write_failure_report(failure_report, caename, exc)
        if save_error is not None:
            with open(failure_report, "a") as fp:
                fp.write("failed_cae_save_error\t%s\n" % save_error)
        print("[%s] failed %s" % (SCRIPT_TAG, caename))
        print("[%s] wrote %s" % (SCRIPT_TAG, failure_report))
    finally:
        os.chdir(old_cwd)
