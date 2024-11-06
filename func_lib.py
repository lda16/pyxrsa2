import numpy as np
import scipy.integrate as integrate
import scipy.optimize as optimize
import random
import matplotlib.pyplot as plt
from scipy.linalg import *
import copy
import time

def Symmetric_Quartic_Polynomial(x,*params):
    #a*(x-x0)^4+b*(x-x0)^2+c, params[a,x0,b,c]
    return params[0]*np.power((x-params[1]),4)+params[2]*(x-params[1])*(x-params[1])+params[3]

def get_type_name(obj):
    return str(type(obj)).split("'")[1]

def rotation_matrix_components(orientation,gamma):
    # with ZYZ convention of Euler angle
    epsilon = 1e-6
    # print(orientation)
    c2 = np.dot(orientation,[0,0,1])
    if np.abs(c2)>1-1e-12:
        s2 = np.linalg.norm(np.cross(orientation,[0,0,1]))
        # print(orientation,np.linalg.norm(orientation))
        if c2>0:
            angle2 = np.arcsin(s2)
        else:
            angle2 = np.pi-np.arcsin(s2)
    else:
        angle2 = np.arccos(c2)
        s2 = np.sin(angle2)
    if(np.abs(s2)<epsilon):
        c1 = 1
        s1 = 0
        c2 = 1
        s2 = 0
    else:
        c1 = orientation[0]/s2
        s1 = orientation[1]/s2
    c3 = np.cos(gamma)
    s3 = np.sin(gamma)
    return s1,s2,s3,c1,c2,c3

def get_mirror_vector(axis,original_vector):
    return 2*np.dot(axis,original_vector)/np.dot(axis,axis)*axis-original_vector

def GetDxDyFromEulerAngle(orientation,gamma):
    # with ZYZ convention of Euler angle
    # print(orientation)
    s1,s2,s3,c1,c2,c3 = rotation_matrix_components(orientation,gamma)
    dx = np.array([c1*c2*c3-s1*s3, c1*s3+c2*c3*s1, -c3*s2])
    dy = np.array([-c3*s1-c1*c2*s3, c1*c3-c2*s1*s3, s2*s3])
    return dx,dy

def GetEulerAngleFromDxDy(dx,dy):
    dx = dx/np.linalg.norm(dx)
    dy = dy/np.linalg.norm(dy)
    dz = np.cross(dx,dy)
    dz = dz/np.linalg.norm(dz)
    dx0,_ = GetDxDyFromEulerAngle(dz,0)
    s1 = Triangular_Det(dx0,dx,dz)
    c1 = np.dot(dx0,dx)
    if np.abs(c1)>1 and np.abs(c1)<1+1e-12:
        c1 = 1.0
    if s1>0:
        return dz,np.arccos(c1)
    else:
        return dz,2*np.pi-np.arccos(c1)

def GetRotationMatrixFromEulerAngle(orientation,gamma):
    # with ZYZ convention of Euler angle
    s1,s2,s3,c1,c2,c3 = rotation_matrix_components(orientation,gamma)
    rotation_matrix = np.array([[c1*c2*c3-s1*s3,-c3*s1-c1*c2*s3,c1*s2],[c1*s3+c2*c3*s1,c1*c3-c2*s1*s3,s1*s2],[-c3*s2,s2*s3,c2]])
    return rotation_matrix

def Get_Phi(dz,dx,p):
    # print('called')
    xy_project = p-dz*np.dot(p,dz)/np.dot(dz,dz)
    xy_pnorm = xy_project/np.linalg.norm(xy_project)
    sin_phi = Triangular_Det(dx,xy_pnorm,dz)
    cos_phi = np.dot(xy_pnorm,dx)
    if sin_phi>0:
        return np.arccos(cos_phi)
    else:
        return -np.arccos(cos_phi)

def Triangular_Det(p1,p2,p3):
    a = np.linalg.det(np.array([p1,p2,p3]))
    return a

def calculate_determinants_cpu(matrices):
    n = len(matrices)
    d = []
    for i in range(n):
        d.append(Triangular_Det(matrices[i][0],matrices[i][1],matrices[i][2]))
    return d

def GetAreaAndNormal(p1,p2,p3):
    v1 = p2-p1
    v2 = p3-p1
    cross = np.cross(v1,v2)
    norm2_cross = np.linalg.norm(cross)
    n = cross/norm2_cross
    a = 0.5*norm2_cross
    return a,n

def zero_func(_):
    return 0

def zero_vector_func(_):
    return np.zeros(3)

def Projection(coordinate,orientation,gamma,point):
    relative_pos = point-coordinate
    dx,dy = GetDxDyFromEulerAngle(orientation,gamma)
    new_coord = np.array([np.dot(relative_pos,dx),\
                          np.dot(relative_pos,dy),\
                          np.dot(relative_pos,orientation)])
    return new_coord

def Gaussian(x,*params):
    return params[0]/np.sqrt(2*np.pi)/params[2]*np.exp(-(x-params[1])*(x-params[1])/2/params[2]/params[2])

def Inside_a_Boundary(bp,p,debug=False,bias_vector=np.zeros(3)):
    # print(bp)
    n = len(bp)
    ref_id = 0
    while(True):
        # print(bp, ref_id)
        if ref_id >=n:
            print(n)
            fig = plt.figure()
            ax = fig.add_subplot(111, projection='3d')
            st = np.array(bp).transpose()
            ax.scatter(st[0], st[1],st[2])
            n= st.shape[1]
            for i in range(st.shape[1]):
                ax.plot([st[0,i], st[0,(i+1)%n]], [st[1,i], st[1,(i+1)%n]], [st[2,i], st[2,(i+1)%n]], color='red')
            ax.scatter(p[0],p[1],p[2])
            plt.show()
            return input('inside the boundary?')
        ref_point = (bp[ref_id]+bp[(ref_id+1)%n])/2
        count = 0
        for i in range(n):
            t = Line_Intersection(p,ref_point,bp[i],bp[(i+1)%n],in_plane_check=False,bias_vector=bias_vector,debug=debug)
            if debug:
                print(t)
                fig = plt.figure()
                ax = fig.add_subplot(111, projection='3d')
                st = np.array(bp).transpose()
                n= st.shape[1]
                for k in range(st.shape[1]):
                    ax.plot([st[0,k], st[0,(k+1)%n]], [st[1,k], st[1,(k+1)%n]], [st[2,k], st[2,(k+1)%n]], color='red')
                ax.plot([bp[i][0],bp[(i+1)%n][0]],[bp[i][1],bp[(i+1)%n][1]],[bp[i][2],bp[(i+1)%n][2]],color='orange')
                ax.plot([p[0],ref_point[0]],[p[1],ref_point[1]],[p[2],ref_point[2]],color='green')
                plt.show()
            if t == 'parallel':
                break
            if t is None:
                return False
            if t[0]>0 and t[1]<=1 and t[1]>=0:
                count +=1
            if np.abs(t[0])<1e-12 and t[1]<=1+1e-12 and t[1]>=0-1e-12:
                if Triangular_Det(bp[i],bp[(i+1)%n],p) <1e-12:
                    return 'on_boundary'
        if t == 'parallel':
            ref_id+=1
            continue
        else:
            break
    if count%2 == 0:
        return False
    else:
        return True

def Line_Intersection(a1,a2,b1,b2,in_plane_check=False,bias_vector=np.zeros(3),debug=False):
    v1 = a2-a1
    n2 = np.cross(b1-bias_vector,b2-bias_vector)
    # if debug:
    #     print(a1,a2,b1,b2)
    #     print(np.dot(n2,v1))
    #     fig = plt.figure()
    #     ax = fig.add_subplot(111, projection='3d')
    #     st = np.array([a1,a2,b1,b2]).transpose()
    #     ax.scatter(st[0], st[1],st[2])
    #     n= st.shape[1]
    #     for i in range(st.shape[1]):
    #         ax.plot([st[0,i], st[0,(i+1)%n]], [st[1,i], st[1,(i+1)%n]], [st[2,i], st[2,(i+1)%n]], color='red')
    #     ax.plot([bias_vector[0],a1[0]],[bias_vector[1],a1[1]],[bias_vector[2],a1[2]])
    #     plt.show()
    if np.abs(np.dot(n2,v1))<1e-12:
        return 'parallel'
    t1 = np.dot(n2,bias_vector-a1)/np.dot(n2,v1)
    v2 = b2-b1
    n1 = np.cross(a1-bias_vector,a2-bias_vector)
    if np.abs(np.dot(n1,v2))<1e-12:
        return 'parallel'
    t2 = np.dot(n1,bias_vector-b1)/np.dot(n1,v2)
    point1 = a1+t1*v1
    point2 = b1+t2*v2
    if (np.linalg.norm(point1-point2)/np.linalg.norm(a2)>0.1 and in_plane_check):
        if debug:
            print(t1,t2)
            print(point1,point2)
        return None # Not in the same/close planar surfaces
    if (np.dot(point1,point2)<0):
        new_t = Line_Plane_CrossPoint(bias_vector,a1,a2,b1,b2)
        if new_t is None and debug:
            fig = plt.figure()
            ax = fig.add_subplot(111, projection='3d')
            st = np.array([a1,a2,b1,b2]).transpose()
            ax.scatter(st[0], st[1],st[2])
            n= st.shape[1]
            for i in range(st.shape[1]):
                ax.plot([st[0,i], st[0,(i+1)%n]], [st[1,i], st[1,(i+1)%n]], [st[2,i], st[2,(i+1)%n]], color='red')
            ax.plot([bias_vector[0],a1[0]],[bias_vector[1],a1[1]],[bias_vector[2],a1[2]])
            plt.show()
            return 'parallel'
        new_p = new_t*a1
        return Line_Intersection(new_p,a2,b1,b2)
    if debug:
        print(t1,t2)
    return t1,t2

def Line_Plane_CrossPoint(origin,direction,p1,p2,p3):
    v1 = p2-p1
    v2 = p3-p1
    n = np.cross(v1,v2)
    if np.dot(n,direction)==0:
        return None
    t = np.dot((p1-origin),n)/np.dot(n,direction)
    return t

def Eigen_CovMat(nodes):
    cov_matrix = np.cov(nodes.T)
    eigenvalues,eigenvectors = eigh(cov_matrix)
    id = [id for id,_ in sorted(enumerate(eigenvalues), key=lambda x: x[1])]
    dx = eigenvectors[:,id[-1]]
    dy = eigenvectors[:,id[-2]]
    dz = np.cross(dx,dy)
    lx = eigenvalues[id[-1]]
    ly = eigenvalues[id[-2]]
    alpha = np.sqrt(lx/ly)
    return dx,dy,dz,lx,ly,alpha

def Generate_normal_function_plane_surface(orientation):
    def function(_):
        return orientation
    return function

def Generate_normal_function_spherical_surface(center):
    def function(node):
        return (center-node)/np.linalg.norm(center-node)
    return function