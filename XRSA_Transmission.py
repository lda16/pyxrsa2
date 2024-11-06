import numpy as np
from func_lib import *
import copy
# from XRSA_Component import *
from XRSA_MeshFunc import *
from XRSA_LinearFunc import *
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

def Transmission(so_jo_dist:XRSA_JointDistribution,in_sp_mesh:XRSA_Mesh,mode='PS',redundancy = 0.4,load_saved_state = False,debug=False):
    if not load_saved_state:
        # original_inspmesh = copy.deepcopy(in_sp_mesh)
        # Visualize_Mesh(original_inspmesh)
        in_sp_mesh = Initialize_in_sp_mesh(so_jo_dist,in_sp_mesh,redundancy=redundancy,mode = mode)
        # Visualize_Mesh(original_inspmesh,in_sp_mesh)
        if mode == 'PS':
            base_angular_mesh,an_ori_gam = Initialize_in_an_mesh_PS(so_jo_dist,in_sp_mesh)
        elif mode == 'PB':
            base_angular_mesh,an_ori_gam = Initialize_in_an_mesh_PB(so_jo_dist,in_sp_mesh,debug=debug)
        # Visualize_Mesh(base_angular_mesh)
        # sp_area = in_sp_mesh.GetTotalArea()
        # an_area = base_angular_mesh.GetTotalArea()
        in_jo_dist = XRSA_JointDistribution(spatial_mesh=in_sp_mesh,base_angular_mesh=base_angular_mesh,an_ori_gam=an_ori_gam)
        in_jo_dist.Save_to_File('in_jo_dist_saved_state.pkl')
    else:
        in_jo_dist = Load_JD_From_File('in_jo_dist_saved_state.pkl')
    
    in_jo_dist = Interact(so_jo_dist,in_jo_dist,debug=False)
    # iter=0
    # while (iter<max_iter):
    #     in_jo_dist = Interact(so_jo_dist,in_jo_dist)
    #     active_sp_area,active_an_area = in_jo_dist.Get_Active_Area()
    #     if (sp_area-active_sp_area)/sp_area < redundancy and (an_area-active_an_area)/an_area < redundancy:
    #         break
    #     else:
    #         in_jo_dist = Update_Joint_Dist(in_jo_dist)
    #         iter=iter+1
    return in_jo_dist

def Transmission_self_adaptive(so_jo_dist:XRSA_JointDistribution,in_sp_mesh:XRSA_Mesh,mode='PS',redundancy = 0.4,load_saved_state = False,max_iter=40,dy_factor=6,debug=False):
    # Visualize_Mesh(in_sp_mesh)
    in_sp_mesh.Get_Edges()
    in_sp_mesh.Get_Boundary()
    insp_bp = in_sp_mesh.Get_Boundary_Points()
    in_sp_mesh = Initialize_in_sp_mesh(so_jo_dist,in_sp_mesh,redundancy=redundancy,mode = mode,dy_factor=dy_factor,debug=False)
    # Visualize_Mesh(in_sp_mesh)
    if not load_saved_state:
        if mode == 'PS':
            base_angular_mesh,an_ori_gam = Initialize_in_an_mesh_PS(so_jo_dist,in_sp_mesh,dy_factor=dy_factor,redundancy=redundancy,debug=False)
        elif mode == 'PB':
            base_angular_mesh,an_ori_gam = Initialize_in_an_mesh_PB(so_jo_dist,in_sp_mesh,dy_factor=dy_factor,redundancy=redundancy,debug=False)
        in_jo_dist = XRSA_JointDistribution(in_sp_mesh,base_angular_mesh,an_ori_gam)
        in_jo_dist.Save_to_File('in_jo_dist_saved_state.pkl')
    else:
        in_jo_dist = Load_JD_From_File('in_jo_dist_saved_state.pkl')
    in_jo_dist = Self_Adaptive_Interact(so_jo_dist,in_jo_dist,insp_bp,max_iter=max_iter,dy_factor=6,redundancy=redundancy,debug=debug)
    return in_jo_dist

def Initialize_in_sp_mesh(so_jo_dist:XRSA_JointDistribution,in_sp_mesh:XRSA_Mesh,dy_factor=6,redundancy=0.4,mode='PS',debug=False):
    print('Initializing spatial mesh')
    so_sp_mesh = so_jo_dist.refmesh
    nodes_involved = [False for i in range(in_sp_mesh.num_nodes)]
    involved_points = []
    cross_centers = []
    for id_so_spatial in range(so_sp_mesh.num_nodes):
        print('Initializing spatial mesh: %u/%u'%(id_so_spatial,so_sp_mesh.num_nodes))
        matrices = []
        so_an_mesh = so_jo_dist.values[id_so_spatial].refmesh
        node_ss = so_sp_mesh.nodes[id_so_spatial]
        # print(in_sp_mesh.nodes)
        cross_points = Get_Ray_Surface_Intersection_NodeSS(node_ss,so_an_mesh,in_sp_mesh)
        projection_mesh = XRSA_Mesh(cross_points,so_an_mesh.facets)
        cross_centers.append(np.average(cross_points,axis=0))
        # Visualize_Mesh(projection_mesh,in_sp_mesh)
        projection_mesh.bias_vector = in_sp_mesh.bias_vector
        for id_in_spatial in range(in_sp_mesh.num_nodes):
            if not nodes_involved[id_in_spatial]:
                facet_id,di = projection_mesh.Get_Target_Facet(in_sp_mesh.nodes[id_in_spatial],padding=False)
                if facet_id is not None:
                    nodes_involved[id_in_spatial] = True
                    involved_points.append(in_sp_mesh.nodes[id_in_spatial])
        if all(nodes_involved):
            return in_sp_mesh
        for id_pm_spatial in range(projection_mesh.num_nodes):
            facet_id,di = in_sp_mesh.Get_Target_Facet(projection_mesh.nodes[id_pm_spatial],padding=False)
            if facet_id is not None:
                involved_points.append(projection_mesh.nodes[id_pm_spatial])
    cross_centers = np.array(cross_centers)
    projection_pbmesh = XRSA_Mesh(cross_centers,so_sp_mesh.facets)
    for id_in_spatial in range(in_sp_mesh.num_nodes):
        if not nodes_involved[id_in_spatial]:
            facet_id,di = projection_pbmesh.Get_Target_Facet(in_sp_mesh.nodes[id_in_spatial],padding=False)
            if facet_id is not None:
                nodes_involved[id_in_spatial] = True
                involved_points.append(in_sp_mesh.nodes[id_in_spatial])
    if in_sp_mesh.handle is not None:
        in_sp_mesh.Delete_from_gpu()
    if projection_pbmesh.handle is not None:
        projection_pbmesh.Delete_from_gpu()
    if all(nodes_involved):
        return in_sp_mesh
    involved_points = np.array(involved_points)
    fake_mesh = XRSA_Mesh(nodes=involved_points,facets=np.zeros((3,3)))
    dx,dy,dz,lx,ly,alpha = Eigen_CovMat(involved_points)
    involved_center = np.average(involved_points,axis=0)
    # relative_involved_points = np.array([involved_points[x]-involved_center for x in range(n_involved)])
    # print(relative_involved_points.shape)
    # projected = np.array([[np.dot(relative_involved_points[x],dx),np.dot(relative_involved_points[x],dy),np.dot(relative_involved_points[x],dz)] for x in range(n_involved)])
    encf_projection = Projection_to_new_Coordinate(involved_center,dx,dy)
    encoder_projection = Mesh_Encoder([encf_projection])
    encoder_projection.Encode(fake_mesh)
    projected = fake_mesh.nodes
    max_dx = np.max(projected[:,0])-np.min(projected[:,0])
    max_dy = np.max(projected[:,1])-np.min(projected[:,1])
    compress_ratio = max_dx/max_dy
    popt,_ = curve_fit(Symmetric_Quartic_Polynomial,projected[:,0],projected[:,1],
                           p0=[0,0,0,0],
                           bounds=([-10,-1,-10,-1],[10,1,10,1]))
    # print(popt)
    quartic_func = lambda x: Symmetric_Quartic_Polynomial(x,*popt)
    encf_fitting_proj = Project_Along_a_Func(dx,dy,quartic_func)
    encf_compress = Extend_along_a_Direction(plane_center = np.zeros(3),direction=np.array([1.,0.,0.]),alpha=1/compress_ratio)
    encoder = Mesh_Encoder([encf_fitting_proj,encf_compress])
    # encoder = Mesh_Encoder([encf_compress])
    encoder.Encode(fake_mesh)
    compressed = fake_mesh.nodes
    max_dx = np.max(compressed[:,0])-np.min(compressed[:,0])
    max_dy = np.max(compressed[:,1])-np.min(compressed[:,1])
    # print(max_dx,max_dy)
    new_center = dx*(np.max(compressed[:,0])+np.min(compressed[:,0]))/2+dy*(np.max(compressed[:,1])+np.min(compressed[:,1]))/2+dz*(np.max(compressed[:,2])+np.min(compressed[:,2]))/2
    _,gamma = GetEulerAngleFromDxDy(dx,dy)
    new_mesh = Generate_Rectangular_Mesh(new_center,np.array([0.,0.,1.]),gamma,shape=[1.2*max_dx,1.2*max_dx],lc=max_dx/dy_factor)
    # Visualize_Mesh(new_mesh,in_sp_mesh)
    encoder.Decode(new_mesh)
    # Visualize_Mesh(new_mesh,in_sp_mesh)
    encoder_projection.Decode(new_mesh)
    # Visualize_Mesh(new_mesh,in_sp_mesh)
    in_sp_mesh.Get_Edges()
    in_sp_mesh.Get_Boundary()
    insp_bp = in_sp_mesh.Get_Boundary_Points()
    new_mesh.Chop(insp_bp)
    new_mesh.norm = copy.deepcopy(in_sp_mesh.norm)
    if debug:
        Visualize_Mesh(new_mesh,in_sp_mesh)
    original_area = in_sp_mesh.GetTotalArea()
    new_area = new_mesh.GetTotalArea()
    if (original_area-new_area)/original_area < redundancy:
        return in_sp_mesh
    else:
        return new_mesh

def Initialize_in_an_mesh_PS(so_jo_dist:XRSA_JointDistribution,in_sp_mesh:XRSA_Mesh,dy_factor=6,redundancy=0.4,debug=False):
    print('Initializing angular mesh: Point Source Mode')
    so_sp_mesh = so_jo_dist.refmesh
    radii = np.zeros(in_sp_mesh.num_nodes)
    ori_gam = []
    for i in range(in_sp_mesh.num_nodes):
        node_is = in_sp_mesh.nodes[i]
        remote_nodes = copy.deepcopy(so_sp_mesh.nodes)
        nodes_normalized = np.array([(remote_nodes[x]-node_is)/np.linalg.norm(remote_nodes[x]-node_is) for x in range(so_sp_mesh.num_nodes)])
        plane_center = np.average(nodes_normalized,axis=0)
        radii[i] = np.max(np.array([np.linalg.norm(nodes_normalized[x]-plane_center) for x in range(so_sp_mesh.num_nodes)]))
        ori_gam.append((plane_center/np.linalg.norm(plane_center),0))
    radius_base = np.max(radii)
    base_mesh = Generate_Circular_Mesh(np.array((0.,0.,1.)),np.array((0.,0.,1.)),radius_base*1.05,lc=radius_base/dy_factor)
    
    base_mesh,ori_gam = Update_angular_mesh(so_jo_dist,in_sp_mesh,base_mesh,ori_gam,max_iter=10,dy_factor=dy_factor,redundancy=redundancy,debug=debug)
    # print('finish')
    return base_mesh,ori_gam
        
def Initialize_in_an_mesh_PB(so_jo_dist:XRSA_JointDistribution,in_sp_mesh:XRSA_Mesh,dy_factor=6,redundancy=0.4,debug=False):
    print('Initializing angular mesh: Parallel Beam Mode')
    so_sp_mesh = so_jo_dist.refmesh
    intersection_nodes = np.zeros((so_sp_mesh.num_nodes,3))
    # print(so_sp_mesh.num_nodes,in_sp_mesh.num_nodes)
    for i in range(so_sp_mesh.num_nodes):
        node_ss = so_sp_mesh.nodes[i]
        # node_sa = np.average(so_jo_dist.values[i].refmesh.nodes,axis=0)
        # node_sa = node_sa/np.linalg.norm(node_sa)
        so_an_mf = so_jo_dist.values[i]
        so_an_mesh = so_an_mf.refmesh
        an_intgs = []
        for j in range(so_an_mesh.num_nodes):
            an_intgs.append(so_an_mf.values[j].Total_Integration())
        if np.sum(np.array(an_intgs)) > 1e-12:
            node_sa = np.sum(np.array([so_an_mesh.nodes[x]*an_intgs[x] for x in range(so_an_mesh.num_nodes)]),axis=0)/np.sum(np.array(an_intgs))
        else:
            node_sa = np.average(so_jo_dist.values[i].refmesh.nodes,axis=0)
        node_sa = node_sa/np.linalg.norm(node_sa)
        # Visualize_Mesh(in_sp_mesh)
        cross_point = Get_Ray_Surface_Intersection(node_ss,node_sa,in_sp_mesh)
        intersection_nodes[i] = cross_point
    # cross_points = Get_Ray_Surface_Intersection_Massive(so_jo_dist,in_sp_mesh)
    # for i in range(so_sp_mesh.num_nodes):
    #     print('Getting angular intersections: %u/%u'%(i,so_sp_mesh.num_nodes))
    #     node_ss = so_sp_mesh.nodes[i]
    #     so_an_mesh = so_jo_dist.values[i].refmesh
    #     cross_points = Get_Ray_Surface_Intersection_NodeSS(node_ss,so_an_mesh,in_sp_mesh)
    #     if debug:
    #         ax = Visualize_Mesh(so_sp_mesh,hold=True)
    #         ax.scatter([node_ss[0]],[node_ss[1]],[node_ss[2]],color='orange')
    #         plt.show()
    #         nodes = cross_points
    #         facets = so_an_mesh.facets
    #         plot_mesh = XRSA_Mesh(nodes,facets)
    #         Visualize_Mesh(plot_mesh,in_sp_mesh)
    #     intersection_nodes[i] = np.average(cross_points,axis=0)
    intersection_facet = copy.deepcopy(so_sp_mesh.facets)
    intersection_mesh = XRSA_Mesh(intersection_nodes,intersection_facet)
    # if debug:
    #     Visualize_Mesh(intersection_mesh,in_sp_mesh)
    orientation_mf = XRSA_MeshFunc_3dvec(intersection_mesh)
    gamma_mf = XRSA_MeshFunc_Scalar(intersection_mesh)
    for i in range(intersection_mesh.num_nodes):
        node_ss = so_sp_mesh.nodes[i]
        reverse_dir = (node_ss-intersection_mesh.nodes[i])/np.linalg.norm(node_ss-intersection_mesh.nodes[i])
        orientation_mf.values[i] = reverse_dir
        # print(reverse_dir)
        so_an_mesh = so_jo_dist.values[i].refmesh
        dx,dy,dz,_,_,_ = Eigen_CovMat(so_an_mesh.nodes)
        if i == 0:
            base_z = dz
            base_x = dx
        else:
            if np.dot(dz,base_z)<0:
                dz=-dz
                dy=-dy
            if np.dot(dx,base_x)<0:
                dx=-dx
                dy=-dy
        ori,gam = GetEulerAngleFromDxDy(dx,dy)
        gamma_mf.values[i] = gam
    
    # orientation_mf.Interpolate()
    # gamma_mf.Interpolate()

    so_base_mesh = copy.deepcopy(so_jo_dist.base_angular_mesh)
    dx,dy,dz,_,_,_ = Eigen_CovMat(so_base_mesh.nodes)
    if np.dot(dz,np.array([0.,0.,1.]))<0:
        dz=-dz
        dy=-dy
    if np.dot(dx,np.array([1.,0.,0.]))<0:
        dx=-dx
        dy=-dy
    # ori,gam = GetEulerAngleFromDxDy(dx,dy)
    max_dx = np.max(np.array([np.dot(dx,(so_base_mesh.nodes[k])) for k in range(so_base_mesh.num_nodes)]))
    min_dx = np.min(np.array([np.dot(dx,(so_base_mesh.nodes[k])) for k in range(so_base_mesh.num_nodes)]))
    max_dy = np.max(np.array([np.dot(dy,(so_base_mesh.nodes[k])) for k in range(so_base_mesh.num_nodes)]))
    min_dy = np.min(np.array([np.dot(dy,(so_base_mesh.nodes[k])) for k in range(so_base_mesh.num_nodes)]))
    shape_x = 1.5*(max_dx-min_dx)
    shape_y = 1.5*(max_dy-min_dy)
    new_base_mesh = Generate_Rectangular_Mesh([0,0,1],[0,0,1],0,[shape_x,shape_x],shape_x/dy_factor)
    encf_extend = Extend_along_a_Direction(np.array([0,0,1]),np.array([0,1,0]),shape_y/shape_x)
    encf_postrans = Position_Transform(np.array([0.,0.,0.]),np.array([0.,0.,1.]),0)
    encoder_extend = Mesh_Encoder([encf_extend,encf_postrans])
    encoder_extend.Encode(new_base_mesh)
    if debug:
        Visualize_Mesh(new_base_mesh,so_base_mesh)
    origam = []
    for i in range(in_sp_mesh.num_nodes):
        node_is = in_sp_mesh.nodes[i]
        ori = orientation_mf(node_is,padding=True)
        ori = ori/np.linalg.norm(ori)
        # if debug:
            # print(ori)
        gam = gamma_mf(node_is,padding=True)
        # print(gam)
        origam.append((ori,gam))
    # print(origam)
    
    # if debug:
    #     for i in range(in_sp_mesh.num_nodes):
    #         ori = 
    
    new_base_mesh,origam = Update_angular_mesh(so_jo_dist,in_sp_mesh,new_base_mesh,origam,max_iter=10,dy_factor=dy_factor,redundancy=redundancy,debug=debug)
    
    return new_base_mesh,origam

def Self_Adaptive_Interact(so_jo_dist,in_jo_dist,insp_bp,max_iter=10,dy_factor=6,redundancy=0.4,debug=False):
    so_sp_mesh = so_jo_dist.refmesh
    in_sp_mesh = in_jo_dist.refmesh
    base_an_mesh = in_jo_dist.base_angular_mesh
    ori_gam = in_jo_dist.an_ori_gam
    
    iter=0
    origin_an_area = base_an_mesh.GetTotalArea()
    origin_sp_area = in_sp_mesh.GetTotalArea()
    while iter<max_iter:
        # involved_nodes = np.zeros(base_mesh.num_nodes,dtype=bool)
        # involved_nodes = []
        # circular = False
        projected_nodes_total = []
        radii = 0
        alpha = 1
        ori_gam_values = []
        base_x = None
        base_z = None
        for i in range(in_sp_mesh.num_nodes):
            print('self adaptive interacting: iteration %u, %u/%u'%(iter,i,in_sp_mesh.num_nodes))
            node_is = in_sp_mesh.nodes[i]
            in_an_mf = in_jo_dist.values[i]
            in_an_mesh = in_an_mf.refmesh
            # if debug:
            #     Visualize_Mesh(in_an_mesh)
            # print(ori_gam[i])
            # postrans_encf = Position_Transform(np.array([0.,0.,0.]),ori_gam[i][0],ori_gam[i][1])
            # mesh_encoder = Mesh_Encoder([postrans_encf])
            # mesh_encoder.Encode(in_an_mesh)
            # if debug:
            #     Visualize_Mesh(in_an_mesh)
            cross_points = Get_Ray_Surface_Intersection_NodeSS(node_is,in_an_mesh,so_sp_mesh)
            # if debug:
            #     nodes = cross_points
            #     facets = in_an_mesh.facets
            #     plot_mesh = XRSA_Mesh(nodes,facets)
            #     Visualize_Mesh(plot_mesh,so_sp_mesh)
            for j in range(cross_points.shape[0]):
                node_ia_rev = -in_an_mesh.nodes[j]
                so_an_mf = so_jo_dist(cross_points[j],debug=False)
                if 'Zero' in get_type_name(so_an_mf):
                    in_an_mf.values[j] = XRSA_Zero_Wavelength_Func()
                    continue
                wfunc = so_an_mf(node_ia_rev)
                # if debug:
                #     ax = Visualize_Mesh(so_an_mf.refmesh,hold=True)
                #     ax.scatter([node_ia_rev[0]],[node_ia_rev[1]],node_ia_rev[2],color='orange')
                #     plt.show()
                # an_facet_id,_ = so_an_mf.refmesh.Get_Target_Facet(node_ia_rev)
                # if an_facet_id is None:
                #     continue
                if 'Zero' in get_type_name(wfunc):
                    in_an_mf.values[j] = XRSA_Zero_Wavelength_Func()
                    continue
                if np.abs(np.dot(node_ia_rev,in_sp_mesh.norm(node_is)))<1e-12:
                    in_an_mf.values[j] = XRSA_Zero_Wavelength_Func()
                else:
                    geometrical_factor = np.abs(np.dot(node_ia_rev,in_sp_mesh.norm(node_is)))/np.abs(np.dot(node_ia_rev,so_sp_mesh.norm(cross_points[j])))
                    in_an_mf.values[j] = so_an_mf(node_ia_rev)*geometrical_factor

                # # Visualize_Mesh(in_sp_mesh)
                # if i == 25:
                #     plot_in_an_mesh = copy.deepcopy(in_an_mesh)
                #     zoom = Zoom_in(np.zeros(3),0.3)
                #     postrans = Position_Transform(node_is,np.array([0.,0.,1.]),0)
                #     encoder = Mesh_Encoder([zoom,postrans])
                #     encoder.Encode(plot_in_an_mesh)
                #     plot_so_an_mesh = copy.deepcopy(so_an_mf.refmesh)
                #     postrans = Position_Transform(cross_points[j],np.array([0.,0.,1.]),0)
                #     encoder = Mesh_Encoder([zoom,postrans])
                #     encoder.Encode(plot_so_an_mesh)
                #     ax = Visualize_Mesh(so_jo_dist.refmesh,in_sp_mesh,plot_in_an_mesh,hold=True)
                #     ax.scatter([node_is[0]],[node_is[1]],[node_is[2]],color='green')
                #     ax.scatter([plot_in_an_mesh.nodes[j,0]],[plot_in_an_mesh.nodes[j,1]],[plot_in_an_mesh.nodes[j,2]],color='orange')
                #     # ax.scatter([cross_points[j,0]],[cross_points[j,1]],[cross_points[j,2]],color='orange')
                #     for k in range(plot_in_an_mesh.num_nodes):
                #         ax.plot([node_is[0],plot_in_an_mesh.nodes[k,0]],[node_is[1],plot_in_an_mesh.nodes[k,1]],[node_is[2],plot_in_an_mesh.nodes[k,2]],color='green',alpha=0.1,linestyle='-')
                #     # for k in range(plot_so_an_mesh.num_nodes):
                #     #     ax.plot([cross_points[j,0],plot_so_an_mesh.nodes[k,0]],[cross_points[j,1],plot_so_an_mesh.nodes[k,1]],[cross_points[j,2],plot_so_an_mesh.nodes[k,2]],color='orange',alpha=0.1,linestyle='-')
                #     # ax.plot([node_is[0],cross_points[j,0]],[node_is[1],cross_points[j,1]],[node_is[2],cross_points[j,2]],color='orange')
                #     plt.show()
                
        for i in range(in_sp_mesh.num_nodes):
            node_is = in_sp_mesh.nodes[i]
            in_an_mf = in_jo_dist.values[i]
            in_an_mesh = in_an_mf.refmesh
            involved_nodes_id_is = np.zeros(in_an_mesh.num_nodes,dtype=bool)
            intensity_an = np.zeros(in_an_mesh.num_nodes)
            for j in range(in_an_mesh.num_nodes):
                intensity_an[j] = in_an_mf.values[j].Total_Integration()
            max_intensity_an = np.max(intensity_an)
            for j in range(in_an_mesh.num_nodes):
                if intensity_an[j]>max_intensity_an*0.05:
                    involved_nodes_id_is[j] = True
            involved_nodes_is = in_an_mesh.nodes[involved_nodes_id_is]
            if involved_nodes_is.shape[0]<3:
                # involved_nodes.append(involved_nodes_is)
                if involved_nodes_is.shape[0] == 0:
                    ori_gam_values.append(ori_gam[i])
                elif involved_nodes_is.shape[0] == 1:
                    ori = involved_nodes_is
                    ori_gam_values.append((ori,ori_gam[i][1]))
                    projected_nodes_total.append(np.zeros(3))
                else:
                    ori = np.average(involved_nodes_is,axis=0)
                    ori_gam_values.append((ori,ori_gam[i][1]))
                    distance = np.linalg.norm(involved_nodes_is[1]-involved_nodes_is[0])
                    projected_nodes_total = projected_nodes_total+[np.array([-distance/2,0.,0.]),np.array([distance/2,0.,0.])]
                continue
            # involved_nodes.append(involved_nodes_is)
            
            plane_center = np.average(involved_nodes_is,axis=0)
            dx,dy,dz,lx,ly,alpha = Eigen_CovMat(involved_nodes_is)
            if base_x is None:
                base_z = plane_center
                base_x = dx
            if np.dot(dz,base_z)<0:
                dy = -dy
                dz = -dz
            if np.dot(dx,base_x)<0:
                dy = -dy
                dx = -dx
            rel = np.array([(involved_nodes_is[x]-plane_center) for x in range(involved_nodes_is.shape[0])])
            radii = np.maximum(radii,np.max(np.array([np.linalg.norm(rel[x]) for x in range(involved_nodes_is.shape[0])])))
            ori = plane_center
            # gam = 0
            projected_nodes = [np.array([np.dot(rel[x],dx),np.dot(rel[x],dy),np.dot(rel[x],dz)]) for x in range(involved_nodes_is.shape[0])]
            projected_nodes_total = projected_nodes_total+projected_nodes
            _,gam = GetEulerAngleFromDxDy(dx,dy)
            # print(dx,dy,ori,gam)
            ori_gam_values.append((ori,gam))
            # if debug:
            #     if 'Zero' not in get_type_name(in_an_mf):
            #         in_an_mf.Display_Pattern(projection=True)
            
            # if debug:
            #     ax = Visualize_Mesh(in_an_mesh,hold=True)
            #     ax.scatter([involved_nodes_is[k,0] for k in range(involved_nodes_is.shape[0])],[involved_nodes_is[k,1] for k in range(involved_nodes_is.shape[0])],[involved_nodes_is[k,2] for k in range(involved_nodes_is.shape[0])],color='green')
            #     plt.show()
        if debug:
            in_jo_dist.Visualize_Spatial_Distribution(display_anmfs=True)
        
        projected_nodes_total = np.array(projected_nodes_total)
        # if debug:
        #     print(projected_nodes_total)
        if projected_nodes_total.shape[0] < 4:
            raise Exception('Too few involved nodes')
        # plane_center = np.average(projected_nodes_total,axis=0)
        # dx,dy,dz,lx,ly,alpha = Eigen_CovMat(projected_nodes_total)
        area = in_jo_dist.values[0].refmesh.GetTotalArea()
        lc_min = 1.5*np.sqrt(area/in_jo_dist.values[0].refmesh.num_facets)
        if alpha<1.5:
            circular = True
        else:
            circular = False
        if circular:
            radii = np.maximum(radii,lc_min)
            ori_gam_values = [(ori_gam_values[x][0],0) for x in range(len(ori_gam_values))]
            new_an_mesh = Generate_Circular_Mesh(np.array((0.,0.,1.)),np.array((0.,0.,1.)),radii*1.05,lc=radii/dy_factor)
        if not circular:
            new_an_mesh = Generate_Fitted_Mesh(projected_nodes_total,dy_factor,lc_min)
            encf_trans = Position_Transform(np.array([0.,0.,-1.]),np.array([0.,0.,1.]),0)
            encf_ptop = Project_To_Plane(np.array([0.,0.,1.]),np.array([0.,0.,0.]))
            encoder = Mesh_Encoder([encf_ptop,encf_trans])
            encoder.Decode(new_an_mesh)
        
        ori_mf = XRSA_MeshFunc_3dvec(in_sp_mesh)    
        gam_mf = XRSA_MeshFunc_Scalar(in_sp_mesh)
        for i in range(in_sp_mesh.num_nodes):
            ori_mf.values[i] = ori_gam_values[i][0]
            gam_mf.values[i] = ori_gam_values[i][1]
                    
        intensity_sp = np.zeros(in_sp_mesh.num_nodes)
        for i in range(in_sp_mesh.num_nodes):
            intensity_sp[i] = in_jo_dist.values[i].Total_Integration()
        max_intensity_sp = np.max(intensity_sp)
        involved_sp_nodes_id = np.zeros(in_sp_mesh.num_nodes,dtype=bool)
        for i in range(in_sp_mesh.num_nodes):
            if intensity_sp[i]>0.05*max_intensity_sp:
                involved_sp_nodes_id[i] = True
        involved_sp_nodes = in_sp_mesh.nodes[involved_sp_nodes_id]
        fake_mesh = XRSA_Mesh(nodes=involved_sp_nodes,facets=np.zeros((3,3)))
        dx,dy,dz,lx,ly,alpha = Eigen_CovMat(involved_sp_nodes)
        involved_center = np.average(involved_sp_nodes,axis=0)
        # relative_involved_points = np.array([involved_points[x]-involved_center for x in range(n_involved)])
        # print(relative_involved_points.shape)
        # projected = np.array([[np.dot(relative_involved_points[x],dx),np.dot(relative_involved_points[x],dy),np.dot(relative_involved_points[x],dz)] for x in range(n_involved)])
        encf_projection = Projection_to_new_Coordinate(involved_center,dx,dy)
        encoder_projection = Mesh_Encoder([encf_projection])
        encoder_projection.Encode(fake_mesh)
        projected = fake_mesh.nodes
        popt,_ = curve_fit(Symmetric_Quartic_Polynomial,projected[:,0],projected[:,1],
                            p0=[0,0,0,0],
                            bounds=([-10,-1,-10,-1],[10,1,10,1]))
        # print(popt)
        quartic_func = lambda x: Symmetric_Quartic_Polynomial(x,*popt)
        encf_fitting_proj = Project_Along_a_Func(np.array([1.,0.,0.]),np.array([0.,1.,0.]),quartic_func)
        encf_compress = Extend_along_a_Direction(plane_center = np.zeros(3),direction=np.array([1.,0.,0.]),alpha=1/alpha)
        encoder = Mesh_Encoder([encf_fitting_proj,encf_compress])
        # encoder = Mesh_Encoder([encf_compress])
        encoder.Encode(fake_mesh)
        compressed = fake_mesh.nodes
        max_dx = np.max(compressed[:,0])-np.min(compressed[:,0])
        max_dy = np.max(compressed[:,1])-np.min(compressed[:,1])
        # print(max_dx,max_dy)
        new_center = dx*(np.max(compressed[:,0])+np.min(compressed[:,0]))/2+dy*(np.max(compressed[:,1])+np.min(compressed[:,1]))/2+dz*(np.max(compressed[:,2])+np.min(compressed[:,2]))/2
        _,gamma = GetEulerAngleFromDxDy(dx,dy)
        new_sp_mesh = Generate_Rectangular_Mesh(new_center,np.array([0.,0.,1.]),gamma,shape=[1.5*max_dx,1.5*max_dx],lc=max_dx/dy_factor)
        encoder.Decode(new_sp_mesh)
        encoder_projection.Decode(new_sp_mesh)
        new_sp_mesh.Chop(insp_bp)
        new_ori_gam = []
        for i in range(new_sp_mesh.num_nodes):
            ori = ori_mf(new_sp_mesh.nodes[i],padding=True)
            gam = gam_mf(new_sp_mesh.nodes[i],padding=True)
            # print((ori,gam))
            new_ori_gam.append((ori,gam))
        if debug:
            Visualize_Mesh(new_sp_mesh,in_sp_mesh)
            Visualize_Mesh(new_an_mesh,base_an_mesh)
        new_sp_area = new_sp_mesh.GetTotalArea()
        new_an_area = new_an_mesh.GetTotalArea()
        if np.abs((origin_sp_area-new_sp_area)/origin_sp_area) > redundancy or np.abs((origin_an_area-new_an_area)/origin_an_area)>redundancy:
            if debug:
                break_or_not = input('Break or not?')
                if break_or_not=='y':
                    return in_jo_dist
                else:
                    base_an_mesh = new_an_mesh
                    origin_an_area = new_an_area
                    ori_gam = new_ori_gam
                    new_sp_mesh.norm = copy.deepcopy(in_sp_mesh.norm)
                    in_sp_mesh = new_sp_mesh
                    in_jo_dist = XRSA_JointDistribution(new_sp_mesh,base_an_mesh,ori_gam)
                    iter=iter+1
            else:            
                base_an_mesh = new_an_mesh
                origin_an_area = new_an_area
                ori_gam = new_ori_gam
                new_sp_mesh.norm = copy.deepcopy(in_sp_mesh.norm)
                in_sp_mesh = new_sp_mesh
                in_jo_dist = XRSA_JointDistribution(new_sp_mesh,base_an_mesh,ori_gam)
                iter=iter+1
        else:
            if debug:
                break_or_not = input('Break or not?')
                if break_or_not=='y':
                    return in_jo_dist
                else:
                    base_an_mesh = new_an_mesh
                    origin_an_area = new_an_area
                    ori_gam = new_ori_gam
                    new_sp_mesh.norm = copy.deepcopy(in_sp_mesh.norm)
                    in_sp_mesh = new_sp_mesh
                    in_jo_dist = XRSA_JointDistribution(new_sp_mesh,base_an_mesh,ori_gam)
                    iter=iter+1
            else:
                return in_jo_dist
    return in_jo_dist

def Update_angular_mesh(so_jo_dist,in_sp_mesh,base_mesh,ori_gam,max_iter=10,dy_factor=10,redundancy=0.4,debug=False):
    so_sp_mesh = so_jo_dist.refmesh
    
    iter=0
    origin_area = base_mesh.GetTotalArea()
    while iter<max_iter:
        # involved_nodes = np.zeros(base_mesh.num_nodes,dtype=bool)
        involved_nodes = []
        # circular = False
        projected_nodes_total = []
        radii = 0
        new_ori_gam = []
        base_x = None
        base_z = None
        for i in range(in_sp_mesh.num_nodes):
            print('Updating angular mesh: iteration %u, %u/%u'%(iter,i,in_sp_mesh.num_nodes))
            node_is = in_sp_mesh.nodes[i]
            in_an_mesh = copy.deepcopy(base_mesh)
            # print(ori_gam[i])
            postrans_encf = Position_Transform(np.array([0.,0.,0.]),ori_gam[i][0],ori_gam[i][1])
            mesh_encoder = Mesh_Encoder([postrans_encf])
            mesh_encoder.Encode(in_an_mesh)
            # cross_points = Get_Ray_Surface_Intersection_NodeSS(node_is,in_an_mesh,so_sp_mesh)
            so_proj = copy.deepcopy(so_sp_mesh)
            for j in range(so_proj.num_nodes):
                so_proj.nodes[j] = (so_proj.nodes[j]-node_is)/np.linalg.norm(so_proj.nodes[j]-node_is)
            if debug:
                if i == 15:
                    Visualize_Mesh(so_proj,in_an_mesh)
            involved_nodes_id_is = np.zeros(in_an_mesh.num_nodes,dtype=bool)
            for j in range(in_an_mesh.num_nodes):
                # node_ia_rev = -in_an_mesh.nodes[j]
                facet_id,di = so_proj.Get_Target_Facet(in_an_mesh.nodes[j])
                if facet_id is None:
                    continue
                involved_nodes_id_is[j] = True
            involved_nodes_is = in_an_mesh.nodes[involved_nodes_id_is]
            if debug:
                if i == 15:
                    ax = Visualize_Mesh(in_an_mesh,hold=True)
                    ax.scatter([involved_nodes_is[k,0] for k in range(involved_nodes_is.shape[0])],[involved_nodes_is[k,1] for k in range(involved_nodes_is.shape[0])],[involved_nodes_is[k,2] for k in range(involved_nodes_is.shape[0])],color='orange')
                    plt.show()
            if involved_nodes_is.shape[0]<3:
                if involved_nodes_is.shape[0] == 0:
                    new_ori_gam.append(ori_gam[i])
                elif involved_nodes_is.shape[0] == 1:
                    ori = involved_nodes_is
                    new_ori_gam.append((ori,ori_gam[i][1]))
                    projected_nodes_total.append(np.zeros(3))
                else:
                    ori = np.average(involved_nodes_is,axis=0)
                    new_ori_gam.append((ori,ori_gam[i][1]))
                    distance = np.linalg.norm(involved_nodes_is[1]-involved_nodes_is[0])
                    projected_nodes_total = projected_nodes_total+[np.array([-distance/2,0.,0.]),np.array([distance/2,0.,0.])]
                continue
            involved_nodes.append(involved_nodes_is)
            
            plane_center = np.average(involved_nodes_is,axis=0)
            dx,dy,dz,lx,ly,alpha = Eigen_CovMat(involved_nodes_is)
            if base_x is None:
                base_z = plane_center
                base_x = dx
            if np.dot(dz,base_z)<0:
                dy = -dy
                dz = -dz
            if np.dot(dx,base_x)<0:
                dy = -dy
                dx = -dx
            rel = np.array([(involved_nodes_is[x]-plane_center) for x in range(involved_nodes_is.shape[0])])
            radii = np.maximum(radii,np.max(np.array([np.linalg.norm(rel[x]) for x in range(involved_nodes_is.shape[0])])))
            ori = plane_center
            # gam = 0
            projected_nodes = [np.array([np.dot(rel[x],dx),np.dot(rel[x],dy),np.dot(rel[x],dz)]) for x in range(involved_nodes_is.shape[0])]
            projected_nodes_total = projected_nodes_total+projected_nodes
            _,gam = GetEulerAngleFromDxDy(dx,dy)
            # print(dx,dy,ori,gam)
            new_ori_gam.append((ori,gam))
        projected_nodes_total = np.array(projected_nodes_total)
        if debug:
            print(projected_nodes_total)
        if projected_nodes_total.shape[0] < 4:
            raise Exception('Too few involved_nodes')
        # plane_center = np.average(projected_nodes_total,axis=0)
        # dx,dy,dz,lx,ly,alpha = Eigen_CovMat(projected_nodes_total)
        if alpha<1.5:
            circular = True
        else:
            circular = False
        if circular:
            new_ori_gam = [(new_ori_gam[x][0],0) for x in range(len(new_ori_gam))]
            new_mesh = Generate_Circular_Mesh(np.array((0.,0.,1.)),np.array((0.,0.,1.)),radii*1.05,lc=radii/3)
        if not circular:
            popt,_ = curve_fit(Symmetric_Quartic_Polynomial,projected_nodes_total[:,0],projected_nodes_total[:,1],
                           p0=[0,0,0,0],
                           bounds=([-10,-0.1,-10,-1.],[10,0.1,10,1.]))
            quartic_func = lambda x: Symmetric_Quartic_Polynomial(x,*popt)
            fake_mesh = XRSA_Mesh(projected_nodes_total,np.array([]))
            max_dx = np.max(fake_mesh.nodes[:,0])-np.min(fake_mesh.nodes[:,0])
            max_dy = np.max(fake_mesh.nodes[:,1])-np.min(fake_mesh.nodes[:,1])
            alpha = max_dx/max_dy
            encf_fitting_proj = Project_Along_a_Func(np.array([1.,0.,0.]),np.array([0.,1.,0.]),quartic_func)
            encf_compress = Extend_along_a_Direction(plane_center=np.zeros(3),direction=np.array([1.,0.,0.]),alpha=1/alpha) 
            encoder = Mesh_Encoder([encf_fitting_proj,encf_compress])
            encoder.Encode(fake_mesh)
            max_dx = np.max(fake_mesh.nodes[:,0])-np.min(fake_mesh.nodes[:,0])
            max_dy = np.max(fake_mesh.nodes[:,1])-np.min(fake_mesh.nodes[:,1])
            new_center = np.array([(np.max(fake_mesh.nodes[:,0])+np.min(fake_mesh.nodes[:,0]))/2,
                                (np.max(fake_mesh.nodes[:,1])+np.min(fake_mesh.nodes[:,1]))/2,
                                (np.max(fake_mesh.nodes[:,2])+np.min(fake_mesh.nodes[:,2]))/2])
            new_mesh = Generate_Rectangular_Mesh(new_center,np.array([0.,0.,1.]),0,[1.2*max_dx,1.2*max_dy],max_dy/dy_factor)
            encoder.Decode(new_mesh)
            encf_trans = Position_Transform(np.array([0.,0.,-1.]),np.array([0.,0.,1.]),0)
            encf_ptop = Project_To_Plane(np.array([0.,0.,1.]),np.array([0.,0.,0.]))
            encoder = Mesh_Encoder([encf_ptop,encf_trans])
            encoder.Decode(new_mesh)
        if debug:
            Visualize_Mesh(new_mesh,base_mesh)
        new_area = new_mesh.GetTotalArea()
        if (origin_area-new_area)/origin_area > redundancy:
            if debug:
                Visualize_Mesh(base_mesh,new_mesh)
            base_mesh = new_mesh
            origin_area = new_area
            ori_gam = new_ori_gam
            iter=iter+1
        else:
            return base_mesh,ori_gam

def Interact(so_jo_dist:XRSA_JointDistribution,in_jo_dist:XRSA_JointDistribution,debug=False):
    so_sp_mesh = so_jo_dist.refmesh
    in_sp_mesh = in_jo_dist.refmesh
    # Visualize_Mesh(so_sp_mesh)
    # cross_points = Get_Ray_Surface_Intersection_Massive(in_jo_dist,so_sp_mesh,debug=False)
    # print(cross_points)
    for i in range(in_sp_mesh.num_nodes):
        print('Interact: %u/%u'%(i,in_sp_mesh.num_nodes))
        in_an_mesh = in_jo_dist.values[i].refmesh
        node_is = in_sp_mesh.nodes[i]
        cross_points = Get_Ray_Surface_Intersection_NodeSS(node_is,in_an_mesh,so_sp_mesh)
        if debug:
            # Visualize_Mesh(in_an_mesh)
            # Visualize_Mesh(so_sp_mesh,in_an_mesh)
            # print(so_sp_mesh.nodes,cross_points[i])
            ax = Visualize_Mesh(so_sp_mesh,hold=True)
            for j in range(in_an_mesh.num_facets):
                nodes = np.array([cross_points[in_an_mesh.facets[j,x]] for x in range(3)])
                nodes = nodes.transpose()
                for k in range(3):
                    ax.plot([nodes[0,k], nodes[0,(k+1)%3]], [nodes[1,k], nodes[1,(k+1)%3]], [nodes[2,k], nodes[2,(k+1)%3]], color='orange')
            plt.show()
            involved_annodes = []
        for j in range(in_an_mesh.num_nodes):
            cpoint = cross_points[j]
            so_an_mf = so_jo_dist(cpoint,debug=False)
            if 'Zero' in get_type_name(so_an_mf):
                continue
            # if debug:
            #     typename = get_type_name(so_an_mf)
            #     if 'Zero' not in typename:
            #         ax = Visualize_Mesh(so_sp_mesh,hold=True)
            #         ax.scatter(cpoint[0],cpoint[1],cpoint[2],color='orange')
            #         plt.show()
            #         Visualize_Mesh(so_an_mf.refmesh)
            ori_reverse = (node_is-cpoint)/np.linalg.norm(node_is-cpoint)
            if debug:
                # so_an_mf(ori_reverse).plot_func()
                involved_annodes.append(cpoint)
            # print(so_an_mf)
            if np.abs(np.dot(ori_reverse,in_sp_mesh.norm(node_is)))<1e-12:
                in_jo_dist.values[i].values[j] = XRSA_Zero_Wavelength_Func()
            else:
                geometrical_factor = np.abs(np.dot(ori_reverse,in_sp_mesh.norm(node_is)))/np.abs(np.dot(ori_reverse,so_sp_mesh.norm(cpoint)))
                # print(geometrical_factor)
                # so_an_mf(ori_reverse).plot_func()
                # print(so_an_mf(ori_reverse))
                # aa = so_an_mf(ori_reverse)*bien_factor
                # aa.plot_func()
                in_jo_dist.values[i].values[j] = so_an_mf(ori_reverse,debug=False)*geometrical_factor
                if debug:
                    in_jo_dist.values[i].values[j].plot_func()
                    print(in_jo_dist.values[i].values[j].Total_Integration())
        if debug:
            
            ax = Visualize_Mesh(so_sp_mesh,hold=True)
            for j in range(in_an_mesh.num_facets):
                nodes = np.array([cross_points[in_an_mesh.facets[j,x]] for x in range(3)])
                nodes = nodes.transpose()
                for k in range(3):
                    ax.plot([nodes[0,k], nodes[0,(k+1)%3]], [nodes[1,k], nodes[1,(k+1)%3]], [nodes[2,k], nodes[2,(k+1)%3]], color='orange')
            for j in range(len(involved_annodes)):
                node = involved_annodes[j]
                ax.scatter([node[0]], [node[1]], [node[2]], color='green')
            plt.show()
            in_jo_dist.values[i].Display_Pattern(projection=True)
            # print(in_jo_dist.values[i].values[j].Total_Integration())
                # in_jo_dist.values[i].values[j].plot_func()
    return in_jo_dist

def Get_Ray_Surface_Intersection(node_ss,node_sa,in_sp_mesh,debug=False):
    matrices = []
    for id_in_spatial in range(in_sp_mesh.num_facets):
        rel_position = np.array([in_sp_mesh.nodes[in_sp_mesh.facets[id_in_spatial,x]]-node_ss for x in range(3)],dtype=np.float32)
        remote_projections = np.array([rel_position[x]/np.linalg.norm(rel_position[x]) for x in range(3)])
        # print(remote_projections,node_sa)
        matrix0 = remote_projections
        matrix1 = np.array([node_sa]+[remote_projections[(x+1)%3] for x in range(2)],dtype=np.float32)
        matrix2 = np.array([node_sa]+[remote_projections[(x+2)%3] for x in range(2)],dtype=np.float32)
        matrix3 = np.array([node_sa]+[remote_projections[(x+3)%3] for x in range(2)],dtype=np.float32)
        matrices = matrices+[matrix0,matrix1,matrix2,matrix3]
    d_array = calculate_determinants_cpu(matrices)
    d_array = np.array(d_array).reshape(in_sp_mesh.num_facets,-1)
    di_array = np.zeros((in_sp_mesh.num_facets,3))
    for k in range(in_sp_mesh.num_facets):
        for l in range(3):
            di_array[k,l] = d_array[k,l+1]/d_array[k,0]
    one_negative_profile=[]
    target_facet_id = None
    flag=True
    for id_in_spatial in range(in_sp_mesh.num_facets):
        di = di_array[id_in_spatial]
        if all(di>0):
            target_facet_id = id_in_spatial
            flag=False
            break
        elif np.sum(di<0)==1:
            one_negative_profile.append([id_in_spatial,np.min(di)])
    if flag:
        # print(one_negative_profile)
        one_negative_profile = np.array(one_negative_profile)
        arg_onenegative = np.argmax(one_negative_profile[:,1])
        target_facet_id = one_negative_profile[arg_onenegative,0]
    target_facet_id = int(target_facet_id)
    p_nodes = in_sp_mesh.GetNodesFromFacet(target_facet_id)
    t = Line_Plane_CrossPoint(node_ss,node_sa,p_nodes[0],p_nodes[1],p_nodes[2])
    cross_point = node_ss+node_sa*t
    cross_point = np.array(cross_point)
    return cross_point

def Get_Ray_Surface_Intersection_NodeSS(node_ss,so_an_mesh,in_sp_mesh,debug=False,use_cuda = False):
    matrices = []
    for id_so_angular in range(so_an_mesh.num_nodes):
        node_sa = so_an_mesh.nodes[id_so_angular]
        for id_in_spatial in range(in_sp_mesh.num_facets):
            rel_position = np.array([in_sp_mesh.nodes[in_sp_mesh.facets[id_in_spatial,x]]-node_ss for x in range(3)],dtype=np.float32)
            remote_projections = np.array([rel_position[x]/np.linalg.norm(rel_position[x]) for x in range(3)])
            # print(remote_projections,node_sa)
            matrix0 = remote_projections
            matrix1 = np.array([node_sa]+[remote_projections[(x+1)%3] for x in range(2)],dtype=np.float32)
            matrix2 = np.array([node_sa]+[remote_projections[(x+2)%3] for x in range(2)],dtype=np.float32)
            matrix3 = np.array([node_sa]+[remote_projections[(x+3)%3] for x in range(2)],dtype=np.float32)
            matrices = matrices+[matrix0,matrix1,matrix2,matrix3]
    # print(len(matrices))
    d_array = calculate_determinants(matrices)
    d_array = np.array(d_array).reshape(so_an_mesh.num_nodes,in_sp_mesh.num_facets,-1)
    di_array = np.zeros((so_an_mesh.num_nodes,in_sp_mesh.num_facets,3))
    for j in range(so_an_mesh.num_nodes):
        for k in range(in_sp_mesh.num_facets):
            # if debug:
            #     print(d_array[j,k])
            for l in range(3):
                di_array[j,k,l] = d_array[j,k,l+1]/d_array[j,k,0]
    target_facet_ids = np.zeros(so_an_mesh.num_nodes,dtype=int)
    for id_so_angular in range(so_an_mesh.num_nodes):
        one_negative_profile=[]
        flag=True
        for id_in_spatial in range(in_sp_mesh.num_facets):
            di = di_array[id_so_angular,id_in_spatial]
            if all(di>0):
                target_facet_ids[id_so_angular] = id_in_spatial
                flag=False
                break
            elif np.sum(di<0)==1:
                one_negative_profile.append([id_in_spatial,np.min(di)])
        if flag:
            # print(one_negative_profile)
            one_negative_profile = np.array(one_negative_profile)
            arg_onenegative = np.argmax(one_negative_profile[:,1])
            target_facet_ids[id_so_angular] = one_negative_profile[arg_onenegative,0]
    if use_cuda:
        origins = []
        directions = []
        p1s = []
        p2s = []
        p3s = []
        for id_so_angular in range(so_an_mesh.num_nodes):
            node_sa = so_an_mesh.nodes[id_so_angular]
            p_nodes = in_sp_mesh.GetNodesFromFacet(target_facet_ids[id_so_angular])
            # if debug:
            #     print(node_ss,node_sa,p_nodes)
            origins.append(node_ss)
            directions.append(node_sa)
            p1s.append(p_nodes[0])
            p2s.append(p_nodes[1])
            p3s.append(p_nodes[2])
        cross_points = line_plane_crosspoints(origins,directions,p1s,p2s,p3s)
    else:
        cross_points = []
        for id_so_angular in range(so_an_mesh.num_nodes):
            node_sa = so_an_mesh.nodes[id_so_angular]
            p_nodes = in_sp_mesh.GetNodesFromFacet(target_facet_ids[id_so_angular])
            t = Line_Plane_CrossPoint(node_ss,node_sa,p_nodes[0],p_nodes[1],p_nodes[2])
            cross_point = node_ss+node_sa*t
            cross_points.append(cross_point)
        cross_points = np.array(cross_points)
    # if debug:
    #     print(cross_points)
    cross_points = cross_points.reshape(so_an_mesh.num_nodes,3)
    return cross_points

def Get_Ray_Surface_Intersection_Massive(so_jo_dist,in_sp_mesh,debug=False):
    so_sp_mesh = so_jo_dist.refmesh
    cross_points = []
    for id_so_spatial in range(so_sp_mesh.num_nodes):
        print('Getting Intersection Points:%u/%u'%(id_so_spatial,so_sp_mesh.num_nodes))
        cross_points_nodess = Get_Ray_Surface_Intersection_NodeSS(so_sp_mesh.nodes[id_so_spatial],so_jo_dist.values[id_so_spatial].refmesh,in_sp_mesh,debug=debug)
        # if debug:
        #     print(cross_points_nodess)
        cross_points.append(cross_points_nodess)
    return np.array(cross_points)
