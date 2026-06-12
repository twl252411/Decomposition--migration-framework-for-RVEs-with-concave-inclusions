from abaqus import *
from abaqusConstants import *
import numpy as np
#
def _x_axis_intersection_center(points):
    if not points:
        raise ValueError("Curve vertices are empty.")
    if len(points) < 2:
        return float(points[0][0]), 0.0
    if not (np.isclose(points[0][0], points[-1][0]) and np.isclose(points[0][1], points[-1][1])):
        points = points + [points[0]]

    xs = []
    for i in range(len(points) - 1):
        x1, y1 = float(points[i][0]), float(points[i][1])
        x2, y2 = float(points[i + 1][0]), float(points[i + 1][1])
        if np.isclose(y1, 0.0):
            xs.append(x1)
        if np.isclose(y2, 0.0):
            xs.append(x2)
        if (y1 > 0.0 and y2 < 0.0) or (y1 < 0.0 and y2 > 0.0):
            t = -y1 / (y2 - y1)
            xs.append(x1 + t * (x2 - x1))

    if not xs:
        center_x = sum(p[0] for p in points[:-1]) / float(len(points) - 1)
        center_y = sum(p[1] for p in points[:-1]) / float(len(points) - 1)
        return float(center_x), float(center_y)

    return 0.5 * (min(xs) + max(xs)), 0.0


class geometry_shape_2d:
    #
    def __init__(self, user_vars):
        self.user_vars = user_vars
    #
    def lobular2(self, part_name):
        #-----Parameters-----
        radius = self.user_vars['radius']
        #
        s = mdb.models['Model-1'].ConstrainedSketch(name='__profile__', sheetSize=200.0)
        g, v, d, c = s.geometry, s.vertices, s.dimensions, s.constraints
        s.setPrimaryObject(option=STANDALONE)
        s.ArcByCenterEnds(center=(-radius, 0.0), point1=(-radius/2, radius/2*sqrt(3)), point2=(-radius/2, -radius/2*sqrt(3)), direction=COUNTERCLOCKWISE)
        s.ArcByCenterEnds(center=(0.0, -radius*sqrt(3)), point1=(-radius/2, -radius/2*sqrt(3)), point2=(radius/2, -radius/2*sqrt(3)), direction=CLOCKWISE)
        s.ArcByCenterEnds(center=(radius, 0.0), point1=(radius/2, -radius/2*sqrt(3)), point2=(radius/2, radius/2*sqrt(3)), direction=COUNTERCLOCKWISE)      
        s.ArcByCenterEnds(center=(0.0, radius*sqrt(3)), point1=(radius/2, radius/2*sqrt(3)), point2=(-radius/2, radius/2*sqrt(3)), direction=CLOCKWISE)
        #
        p = mdb.models['Model-1'].Part(name=part_name, dimensionality=TWO_D_PLANAR, type=DEFORMABLE_BODY)
        p.BaseShell(sketch=s)
        #
        s.unsetPrimaryObject()
        del mdb.models['Model-1'].sketches['__profile__']
    #
    def pea_shape(self, part_name):
        #-----Parameters-----
        R1, R2, K1, a = self.user_vars['R1'], self.user_vars['R2'], self.user_vars['K1'], self.user_vars['a']
        #
        s = mdb.models['Model-1'].ConstrainedSketch(name='__profile__', sheetSize=200.0)
        g, v, d, c = s.geometry, s.vertices, s.dimensions, s.constraints
        s.setPrimaryObject(option=STANDALONE)
        points = []
        for ith in range(361):
            points = points + [(R1*cos(ith/180.0*pi) + K1*exp(-a*cos(ith/180.0*pi)-a), R2*sin(ith/180.0*pi))]
        center_x, center_y = _x_axis_intersection_center(points)
        centered_points = [(p[0] - center_x, p[1] - center_y) for p in points]
        s.Spline(points=centered_points)
        #
        p = mdb.models['Model-1'].Part(name=part_name, dimensionality=TWO_D_PLANAR, type=DEFORMABLE_BODY)
        p.BaseShell(sketch=s)
        s.unsetPrimaryObject()
        del mdb.models['Model-1'].sketches['__profile__'] 
    #
    def concave_poly(self, part_name):
        #-----Parameters-----
        poly_vertices = self.user_vars['vertices']
        #
        s = mdb.models['Model-1'].ConstrainedSketch(name='__profile__', sheetSize=200.0)
        g, v, d, c = s.geometry, s.vertices, s.dimensions, s.constraints
        s.setPrimaryObject(option=STANDALONE)
        num_vertices = len(poly_vertices)
        for i in range(num_vertices):
            next_i = (i + 1) % num_vertices
            point1, point2 = tuple(poly_vertices[i, :]), tuple(poly_vertices[next_i, :])
            s.Line(point1=point1, point2=point2)
        #
        p = mdb.models['Model-1'].Part(name=part_name, dimensionality=TWO_D_PLANAR, type=DEFORMABLE_BODY)
        p.BaseShell(sketch=s)
        s.unsetPrimaryObject()
        del mdb.models['Model-1'].sketches['__profile__']  
