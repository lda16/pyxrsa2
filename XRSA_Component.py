import numpy as np
from func_lib import *
from XRSA_MeshFunc import *
from XRSA_LinearFunc import *
from XRSA_Transmission import *

class XRSA_Component():
    def __init__(self, spatial_mesh, name) -> None:
        self.name = name
        self.spatial_mesh = spatial_mesh
        self.incident = {}
        self.response = {}

    def Interact(self,source_component:'XRSA_Component',response_name,mode='PS',load_saved_state = False,max_iter=40,dy_factor=6,debug=False):
        in_key_name = response_name
        self.incident[in_key_name] = Transmission_self_adaptive(source_component.response[in_key_name],self.spatial_mesh,mode=mode,load_saved_state=load_saved_state,max_iter=max_iter,dy_factor=dy_factor,debug=debug)
        # self.incident[in_key_name] = Transmission(source_component.response[in_key_name],self.spatial_mesh,mode=mode, load_saved_state = load_saved_state,debug=debug)

    def Response(self,incident_name):
        response_name = incident_name+"_"+self.name
        self.response[response_name] = None

class XRSA_Source(XRSA_Component):
    def __init__(self, spatial_mesh, joint_dist, name='default_source') -> None:
        super().__init__(spatial_mesh, name)
        response_name = name
        self.response[response_name] = joint_dist

def Generate_Pinhole_Isotropic_Source(coordinate,orientation,wavelength_func,name = 'Default'):
    spatial_mesh = Generate_Circular_Mesh(coordinate,orientation,1e-4,1e-4)
    angular_mesh = Generate_Dome_Mesh(np.array([0,0,0]),orientation,1,0.1,min_theta=np.pi/3)
    joint_dist = XRSA_JointDistribution(spatial_mesh=spatial_mesh,base_angular_mesh=angular_mesh)
    for id_spatial in range(joint_dist.refmesh.num_nodes):
        angular_meshfunc = joint_dist.values[id_spatial]
        for id_angular in range(angular_meshfunc.refmesh.num_nodes):
            angular_meshfunc.values[id_angular]=wavelength_func

    source = XRSA_Source(spatial_mesh,joint_dist,name)
    return source

def Generate_PB_Source(coordinate,orientation,gamma,wavelength_func,shape,lc,name='Default'):
    spatial_mesh = Generate_Rectangular_Mesh(coordinate,orientation,gamma,shape,lc)
    angular_mesh = Generate_Circular_Mesh(orientation,orientation,np.radians(0.01),np.radians(0.01))

class XRSA_Detector(XRSA_Component):
    def __init__(self, spatial_mesh, name='default_detector') -> None:
        super().__init__(spatial_mesh, name)

    def Display_Pattern(self,in_name):
        in_jo_dist = self.incident[in_name]
        in_jo_dist.Visualize_Spatial_Distribution()


class XRSA_Crystal(XRSA_Component):
    def __init__(self,spatial_mesh,spacing,rocking_curve=None,name='Default'):
        super().__init__(spatial_mesh,name=name)
        self.spacing = spacing
        if rocking_curve is not None:
            self.rc_func = rocking_curve
        else:
            self.rc_func = XRSA_LinearFunc(func=lambda x:Gaussian(x,[1.0,0.,5e-5]),lower_limit=-2e-4,upper_limit=2e-4)

    def Bragg_wavelength(self,theta):
        return 2*self.spacing*np.sin(np.pi/2-theta)

    # All the theta to be converted to the angle between the direction and the z-axis, in accord with r-theta-phi coordinate. NOT THE THETA IN THE 2d*sin(theta)
    def Bragg_theta(self,wavelength):
        # print(wavelength,self.spacing)
        return np.pi/2-np.arcsin(wavelength/2/self.spacing)

    def Dispersion(self,in_an_mf,norm_vec,node_ra,w_range,debug=False):
        node_bp = get_mirror_vector(norm_vec,node_ra)
        base_theta = np.arccos(np.dot(norm_vec,node_ra)/np.linalg.norm(norm_vec)/np.linalg.norm(node_ra))
        dw_dtheta = -2*self.spacing*np.cos(np.pi/2-base_theta)
        base_w = self.Bragg_wavelength(base_theta)
        core_w = lambda w: self.rc_func(self.Bragg_theta(w)-base_theta) #theta is the angle between the norm and the node_ra. NOT THE THETA IN THE 2d*sin(theta)
        core_wfunc = XRSA_Wavelength_Func(core_w,base_w+dw_dtheta*self.rc_func.domain[1],base_w+dw_dtheta*self.rc_func.domain[0])
        if 'Zero' in get_type_name(in_an_mf):
            if debug:
                print('zero angular_mf')
            return XRSA_Zero_Wavelength_Func()
        wfunc_in = in_an_mf(node_bp,debug=False)
        # print(wfunc_in.domain,core_wfunc.domain)
        if 'Zero' in get_type_name(wfunc_in):
            return XRSA_Zero_Wavelength_Func()
        if np.maximum(wfunc_in.domain[0],core_wfunc.domain[0]) > np.minimum(wfunc_in.domain[1],core_wfunc.domain[1]):
            return XRSA_Zero_Wavelength_Func()
        wfunc_re = XRSA_Wavelength_Func(lambda w: wfunc_in(w)*core_wfunc(w),np.maximum(wfunc_in.domain[0],core_wfunc.domain[0]),np.minimum(wfunc_in.domain[1],core_wfunc.domain[1]))
        if debug:
            print(np.maximum(wfunc_in.domain[0],core_wfunc.domain[0]),np.minimum(wfunc_in.domain[1],core_wfunc.domain[1]))
            wfunc_re.plot_func()
        return wfunc_re

    def Get_Window(self):
        return [self.rc_func.domain[0],self.rc_func.domain[1]]

    def Response_self_adaptive(self,incident_name,max_iter=10,dy_factor=6,dtheta_factor=6,redundancy=0.4,debug=False):
        super().Response(incident_name)
        in_jo_dist = self.incident[incident_name]
        re_sp_mesh = copy.deepcopy(in_jo_dist.refmesh)
        re_sp_mesh = self.Initialize_re_sp_mesh(in_jo_dist,re_sp_mesh,dy_factor=dy_factor,max_iter=max_iter)
        re_an_basemesh, an_ori_gam, w_range_list = self.Initialize_re_an_mesh(in_jo_dist,re_sp_mesh,dtheta_factor=dtheta_factor)
        # print(an_ori_gam)
        re_jo_dist = XRSA_JointDistribution(re_sp_mesh,re_an_basemesh,an_ori_gam)
        re_sp_mesh = re_jo_dist.refmesh
        
        w_range_max = XRSA_MeshFunc_Scalar(re_sp_mesh)
        w_range_min = XRSA_MeshFunc_Scalar(re_sp_mesh)
        for i in range(re_sp_mesh.num_nodes):
            w_range_max.values[i] = w_range_list[i][1]
            w_range_min.values[i] = w_range_list[i][0]
        
        in_sp_mesh = in_jo_dist.refmesh
        in_sp_mesh.Get_Edges()
        in_sp_mesh.Get_Boundary()
        insp_bp = in_sp_mesh.Get_Boundary_Points()
        
        iter=0
        origin_re_sp_area = re_sp_mesh.GetTotalArea()
        origin_re_an_area = re_an_basemesh.GetTotalArea()
        ori_gam = re_jo_dist.an_ori_gam
        base_an_mesh = re_jo_dist.base_angular_mesh
        while iter<max_iter:
            ori_gam_values = []
            projected_nodes_total = []
            radii = 0
            alpha = 1
            base_x = None
            base_z = None
            for id_spatial in range(re_sp_mesh.num_nodes):
                print('Response adaptive: Iteration %u, %u/%u'%(iter,id_spatial,re_sp_mesh.num_nodes))
                node_rs = re_sp_mesh.nodes[id_spatial]
                re_an_mf = re_jo_dist.values[id_spatial]
                re_an_mesh = re_an_mf.refmesh
                in_an_mf = in_jo_dist(node_rs)
                # Visualize_Mesh(re_an_mesh,in_an_mf.refmesh)
                w_range = [w_range_min(node_rs,padding=True),w_range_max(node_rs,padding=True)]
                norm_vec = re_sp_mesh.norm(node_rs)
                for id_angular in range(re_an_mesh.num_nodes):
                    print('response_angular: %u/%u'%(id_angular,re_an_mesh.num_nodes))
                    node_ra = re_an_mesh.nodes[id_angular]
                    new_wavelength_func = self.Dispersion(in_an_mf,norm_vec,node_ra,w_range,debug=False)
                    re_an_mf.values[id_angular] = new_wavelength_func
            
            intensity_sp = np.zeros(re_sp_mesh.num_nodes)
            # new_wrange_list = []
            for id_spatial in range(re_sp_mesh.num_nodes):
                # re_jo_dist.values[id_spatial].Display_Pattern()
                node_rs = re_sp_mesh.nodes[id_spatial]
                re_an_mf = re_jo_dist.values[id_spatial]
                re_an_mesh = re_an_mf.refmesh
                involved_nodes_id_rs = np.zeros(re_an_mesh.num_nodes,dtype=bool)
                intensity_an = np.zeros(re_an_mesh.num_nodes)
                for id_angular in range(re_an_mesh.num_nodes):
                    intensity_an[id_angular] = re_an_mf.values[id_angular].Total_Integration()
                    # print('angular',intensity_an[id_angular])
                values_an = XRSA_MeshFunc_Scalar(re_an_mesh)
                values_an.values = intensity_an
                intensity_sp[id_spatial] = values_an.Total_Integration()
                # print('spatial',intensity_sp[id_spatial])
                max_intensity_an = np.max(intensity_an)
                for id_angular in range(re_an_mesh.num_nodes):
                    if intensity_an[id_angular]>max_intensity_an*0.05:
                        involved_nodes_id_rs[id_angular] = True
                
                involved_nodes_rs = re_an_mesh.nodes[involved_nodes_id_rs]
                if involved_nodes_rs.shape[0] != 0:
                    plane_center = np.average(involved_nodes_rs,axis=0)
                    encf_ptop = Project_To_Plane(plane_center/np.linalg.norm(plane_center),np.zeros(3))
                    encoder_ptop = Mesh_Encoder([encf_ptop])
                    encoder_ptop.Encode(re_an_mesh)
                involved_nodes_rs = re_an_mesh.nodes[involved_nodes_id_rs]
                if involved_nodes_rs.shape[0]<3:
                    if involved_nodes_rs.shape[0] == 0:
                        ori_gam_values.append((ori_gam[id_spatial][0],None))
                    elif involved_nodes_rs.shape[0] == 1:
                        ori = involved_nodes_rs
                        ori_gam_values.append((ori,None))
                        projected_nodes_total.append(np.zeros(3))
                        encoder_ptop.Decode(re_an_mesh)
                    else:
                        ori = np.average(involved_nodes_rs,axis=0)
                        ori_gam_values.append((ori,None))
                        distance = np.linalg.norm(involved_nodes_rs[1]-involved_nodes_rs[0])
                        projected_nodes_total = projected_nodes_total+[np.array([-distance/2,0.,0.]),np.array([distance/2,0.,0.])]
                        encoder_ptop.Decode(re_an_mesh)
                    continue
                encoder_ptop.Decode(re_an_mesh)
                # ax = Visualize_Mesh(re_an_mesh,hold=True)
                # ax.scatter([involved_nodes_rs[x,0] for x in range(involved_nodes_rs.shape[0])],[involved_nodes_rs[x,1] for x in range(involved_nodes_rs.shape[0])],[involved_nodes_rs[x,2] for x in range(involved_nodes_rs.shape[0])],color='green')
                # plt.show()
                # print(involved_nodes_rs)
                dx,dy,dz,lx,ly,alpha = Eigen_CovMat(involved_nodes_rs)
                # print(dx,dy,dz,alpha)
                if base_x is None:
                    base_z = plane_center
                    base_x = dx
                if np.dot(dz,base_z)<0:
                    dy = -dy
                    dz = -dz
                if np.dot(dx,base_x)<0:
                    dy = -dy
                    dx = -dx
                rel = np.array([(involved_nodes_rs[x]-plane_center) for x in range(involved_nodes_rs.shape[0])])
                radii = np.maximum(radii,np.max(np.array([np.linalg.norm(rel[x]) for x in range(involved_nodes_rs.shape[0])])))
                ori = plane_center
                projected_nodes = [np.array([np.dot(rel[x],dx),np.dot(rel[x],dy),np.dot(rel[x],dz)]) for x in range(involved_nodes_rs.shape[0])]
                projected_nodes_total = projected_nodes_total+projected_nodes
                _,gam = GetEulerAngleFromDxDy(dx,dy)
                ori_gam_values.append((ori,gam))

                # new_wrange_list.append([np.min(np.array([re_an_mf.values[x].domain[0] for x in range(re_an_mesh.num_nodes)])),np.max(np.array([re_an_mf.values[x].domain[1] for x in range(re_an_mesh.num_nodes)]))])
            valid_gammas = []
            for id_spatial in range(re_sp_mesh.num_nodes):
                if ori_gam_values[id_spatial][1] is not None:
                    valid_gammas.append(ori_gam_values[id_spatial][1])
            if len(valid_gammas) > 0:
                avg_gammas = np.average(np.array(valid_gammas))
            else:
                avg_gammas = 0
            for id_spatial in range(re_sp_mesh.num_nodes):
                if ori_gam_values[id_spatial][1] is None:
                    ori_gam_values[id_spatial] = (ori_gam_values[id_spatial][0],avg_gammas)
                
            # print(ori_gam_values)
            projected_nodes_total = np.array(projected_nodes_total)
            if projected_nodes_total.shape[0]<4:
                raise Exception('Too few involved nodes')
            
            lc_min = 1.5*np.sqrt(origin_re_an_area/re_jo_dist.base_angular_mesh.num_facets)
            if alpha<1.5:
                circular = True
            else:
                circular = False
            
            if circular:
                radii = np.maximum(radii,lc_min)
                ori_gam_values = [(ori_gam_values[x][0],0) for x in range(len(ori_gam_values))]
                new_an_mesh = Generate_Circular_Mesh(np.array((0.,0.,1.)),np.array((0.,0.,1.)),radii*1.05,lc=radii/dy_factor)
            if not circular:
                new_an_mesh = Generate_Fitted_Mesh(projected_nodes_total,dy_factor,lc_min,redundancy=(np.sqrt(1+redundancy)-1))     
                encf_trans = Position_Transform(np.array([0.,0.,-1.]),np.array([0.,0.,1.]),0)
                encf_ptop = Project_To_Plane(np.array([0.,0.,1.]),np.array([0.,0.,0.]))
                encoder = Mesh_Encoder([encf_ptop,encf_trans])
                encoder.Decode(new_an_mesh)
            
            ori_mf = XRSA_MeshFunc_3dvec(re_sp_mesh)
            gam_mf = XRSA_MeshFunc_Scalar(re_sp_mesh)
            for id_spatial in range(re_sp_mesh.num_nodes):
                ori_mf.values[id_spatial] = ori_gam_values[id_spatial][0]
                gam_mf.values[id_spatial] = ori_gam_values[id_spatial][1]
            max_intensity_sp = np.max(intensity_sp)
            involved_sp_nodes_id = np.zeros(re_sp_mesh.num_nodes,dtype=bool)
            for i in range(re_sp_mesh.num_nodes):
                if intensity_sp[i]>0.05*max_intensity_sp:
                    involved_sp_nodes_id[i] = True
            # print(max_intensity_sp)
            # print(intensity_sp)
            # print(involved_sp_nodes_id)
            involved_sp_nodes = re_sp_mesh.nodes[involved_sp_nodes_id]
            fake_mesh = XRSA_Mesh(nodes=involved_sp_nodes,facets=np.zeros((3,3)))
            dx,dy,dz,lx,ly,alpha = Eigen_CovMat(involved_sp_nodes)
            involved_center = np.average(involved_sp_nodes,axis=0)
            encf_projection = Projection_to_new_Coordinate(involved_center,dx,dy)
            encoder_projection = Mesh_Encoder([encf_projection])
            encoder_projection.Encode(fake_mesh)
            projected = fake_mesh.nodes
            # print(projected)
            new_sp_mesh = Generate_Fitted_Mesh(projected,dy_factor)
            # print(new_sp_mesh.nodes)
            encoder_projection.Decode(new_sp_mesh)
            # print(insp_bp)
            # if debug:
            #     re_jo_dist.Visualize_Spatial_Distribution(display_anmfs=True)
            # Visualize_Mesh(new_sp_mesh)
            new_sp_mesh.Chop(insp_bp)
            new_ori_gam = []
            for i in range(new_sp_mesh.num_nodes):
                ori = ori_mf(new_sp_mesh.nodes[i],padding=True)
                gam = gam_mf(new_sp_mesh.nodes[i],padding=True)
                print(gam)
                new_ori_gam.append((ori,gam))
            # if debug:
            #     for id_spatial in range(new_sp_mesh.num_nodes):
            #         origin_an_mf = re_jo_dist(new_sp_mesh.nodes[id_spatial])
            #         if 'Zero' in get_type_name(origin_an_mf):
            #             continue
            #         origin_an_mesh = origin_an_mf.refmesh
            #         an_mesh = copy.deepcopy(new_an_mesh)
            #         postrans_encf = Position_Transform(np.array([0.,0.,0.]),new_ori_gam[id_spatial][0],new_ori_gam[id_spatial][1])
            #         mesh_encoder = Mesh_Encoder([postrans_encf])
            #         mesh_encoder.Encode(an_mesh)
            #         Visualize_Mesh(an_mesh,origin_an_mesh)

            if debug:
                Visualize_Mesh(new_sp_mesh,re_sp_mesh)
                Visualize_Mesh(new_an_mesh,base_an_mesh)
            new_sp_area = new_sp_mesh.GetTotalArea()
            new_an_area = new_an_mesh.GetTotalArea()
            if debug:
                break_or_not = input('Break or not?')
                if break_or_not=='y':
                    break
                else:
                    base_an_mesh = new_an_mesh
                    origin_re_an_area = new_an_area
                    ori_gam = new_ori_gam
                    new_sp_mesh.norm = copy.deepcopy(re_sp_mesh.norm)
                    re_sp_mesh = new_sp_mesh
                    re_jo_dist = XRSA_JointDistribution(re_sp_mesh,base_an_mesh,ori_gam)
                    # w_range_list = new_wrange_list
                    iter = iter+1
            else:
                if np.abs((origin_re_sp_area-new_sp_area)/origin_re_sp_area)>redundancy or np.abs((origin_re_an_area-new_an_area)/origin_re_an_area)>redundancy:
                    base_an_mesh = new_an_mesh
                    origin_re_an_area = new_an_area
                    ori_gam = new_ori_gam
                    new_sp_mesh.norm = copy.deepcopy(re_sp_mesh.norm)
                    re_sp_mesh = new_sp_mesh
                    re_jo_dist = XRSA_JointDistribution(re_sp_mesh,base_an_mesh,ori_gam)
                    # w_range_list = new_wrange_list
                    iter = iter+1
                else:
                    break
        response_name = incident_name+"_"+self.name
        self.response[response_name] = re_jo_dist
            

    def Initialize_re_sp_mesh(self,in_jo_dist,re_sp_mesh:XRSA_Mesh,dy_factor=6,max_iter=10,criterion=0.1,debug=False):
        print('Initializing response spatial mesh')
        # for i in range(re_sp_mesh.num_facets):
        #     print(i)
        #     print(re_sp_mesh.facets[i])
        #     ax = Visualize_Mesh(re_sp_mesh,hold=True)
        #     for j in range(3):
        #         ax.plot([re_sp_mesh.nodes[re_sp_mesh.facets[i,j]][0],re_sp_mesh.nodes[re_sp_mesh.facets[i,(j+1)%3]][0]],
        #                 [re_sp_mesh.nodes[re_sp_mesh.facets[i,j]][1],re_sp_mesh.nodes[re_sp_mesh.facets[i,(j+1)%3]][1]],
        #                 [re_sp_mesh.nodes[re_sp_mesh.facets[i,j]][2],re_sp_mesh.nodes[re_sp_mesh.facets[i,(j+1)%3]][2]],color='orange')
        #     plt.show()
        re_sp_mesh.Get_Edges()
        re_sp_mesh.Get_Boundary()

        iter=0
        last_area = re_sp_mesh.GetTotalArea()
        res_mesh = copy.deepcopy(re_sp_mesh)
        if res_mesh.ptop_encoder is not None:
            res_mesh.ptop_encoder.Encode(res_mesh)
        resp_bp = re_sp_mesh.Get_Boundary_Points()
        if res_mesh.ptop_encoder is not None:
            res_mesh.ptop_encoder.Decode(res_mesh)
        window = self.Get_Window()
        while(iter<max_iter):
            involved_facets = np.zeros(res_mesh.num_facets,dtype=bool)
            res_an_mf_list = []
            norm_list = []
            for i in range(res_mesh.num_nodes):
                angular_mf = in_jo_dist(res_mesh.nodes[i])
                # print(angular_mf)
                # ax = Visualize_Mesh(in_jo_dist.refmesh,hold=True)
                # ax.scatter(res_mesh.nodes[i,0],res_mesh.nodes[i,1],res_mesh.nodes[i,2])
                # plt.show()
                res_an_mf_list.append(angular_mf)
                # print(in_jo_dist.refmesh.norm(res_mesh.nodes[i]))
                norm_list.append(res_mesh.norm(res_mesh.nodes[i]))
            for id_facet in range(res_mesh.num_facets):
                min_relative=np.inf
                max_relative=-np.inf
                for i in range(3):
                    angular_mf = res_an_mf_list[res_mesh.facets[id_facet,i]]
                    # print(angular_mf)
                    if 'Zero' in get_type_name(angular_mf):
                        continue
                    angular_mesh = angular_mf.refmesh
                    norm_vector = norm_list[res_mesh.facets[id_facet,i]]
                    for j in range(angular_mesh.num_nodes):
                        if 'Zero' in get_type_name(angular_mf.values[j]):
                            continue
                        # print(angular_mesh.nodes[j],norm_vector,np.dot(angular_mesh.nodes[j],norm_vector))
                        ic_angle = np.arccos(np.dot(angular_mesh.nodes[j],norm_vector)/np.linalg.norm(angular_mesh.nodes[j])/np.linalg.norm(norm_vector))
                        max_wavelength = angular_mf.values[j].domain[1]
                        min_bragg = self.Bragg_theta(max_wavelength)-window[1]
                        min_wavelength = angular_mf.values[j].domain[0]
                        max_bragg = self.Bragg_theta(min_wavelength)-window[0]
                        # print(self.Bragg_theta(max_wavelength),self.Bragg_theta(min_wavelength),window)
                        # print(ic_angle,min_bragg,max_bragg)
                        if ic_angle>min_bragg and ic_angle<max_bragg:
                            involved_facets[id_facet]=True
                            break
                        elif ic_angle<min_bragg:
                            min_relative = np.minimum(min_relative,ic_angle-min_bragg)
                        elif ic_angle>max_bragg:
                            max_relative = np.maximum(max_relative,ic_angle-max_bragg)
                        if max_relative>0 and min_relative<0:
                            involved_facets[id_facet]=True
                            break
                    if involved_facets[id_facet]:
                        break
            if res_mesh.ptop_encoder is not None:
                res_mesh.ptop_encoder.Encode(res_mesh)
            # print(involved_facets)
            new_mesh = res_mesh.Update_Mesh(involved_facets,dy_factor=dy_factor,debug=False)
            new_mesh.Chop(resp_bp)
            new_mesh.norm = copy.deepcopy(re_sp_mesh.norm)
            if res_mesh.ptop_encoder is not None:
                new_mesh.ptop_encoder = copy.deepcopy(res_mesh.ptop_encoder)
                new_mesh.ptop_encoder.Decode(new_mesh)
            if debug:
                Visualize_Mesh(new_mesh,re_sp_mesh)
            new_area = new_mesh.GetTotalArea()
            if (last_area-new_area)/last_area<criterion:
                break
            else:
                # Visualize_Mesh(re_sp_mesh,new_mesh)
                last_area = new_area
                res_mesh = new_mesh
            iter = iter+1
        return res_mesh

    def Initialize_re_an_mesh(self,in_jo_dist,re_sp_mesh:XRSA_Mesh,dtheta_factor=4):
        print('Initializing response angular mesh')
        max_d_theta_an_mesh = 0
        max_d_phi_an_mesh = 0
        window=self.Get_Window()
        # The convolutional core only works with the angular meshfunc.
        wavelength_range_list = []
        theta_phi_list = []
        for id_spatial in range(re_sp_mesh.num_nodes):
            node_rs = re_sp_mesh.nodes[id_spatial]
            angular_mf = in_jo_dist(node_rs)
            # print(angular_mf)
            if 'Zero' in get_type_name(angular_mf):
                # print('zero wfunc')
                wavelength_range_list.append(None)
                theta_phi_list.append(None)
                continue
            angular_mesh = angular_mf.refmesh
            facets_involved = np.zeros(angular_mesh.num_facets,dtype=bool)
            # Only the involved angular facets are taken into consideration when calculating the max conv_core width
            norm_vector = re_sp_mesh.norm(node_rs)
            max_w_list = []
            min_w_list = []
            for id_facet in range(angular_mesh.num_facets):
                min_relative = np.inf
                max_relative = -np.inf
                for i in range(3):
                    # print(angular_mesh.facets[id_facet,i])
                    wavelength_func = angular_mf.values[angular_mesh.facets[id_facet,i]]
                    typename = get_type_name(wavelength_func)
                    if 'Zero' in typename:
                        continue
                    node_ia = angular_mesh.nodes[angular_mesh.facets[id_facet,i]]
                    ic_angle = np.arccos(np.dot(node_ia,norm_vector)/np.linalg.norm(node_ia)/np.linalg.norm(norm_vector))
                    max_w = wavelength_func.domain[1]
                    min_bragg = self.Bragg_theta(max_w)-window[1]
                    min_w = wavelength_func.domain[0]
                    max_bragg = self.Bragg_theta(min_w)-window[0]
                    # print(ic_angle,max_bragg,min_bragg,ic_angle-max_bragg,ic_angle-min_bragg)
                    if ic_angle<max_bragg and ic_angle>min_bragg:
                        facets_involved[id_facet] = True
                        break
                    elif ic_angle>max_bragg:
                        max_relative = np.maximum(max_relative,ic_angle-max_bragg)
                    elif ic_angle<min_bragg:
                        min_relative = np.minimum(min_relative,ic_angle-min_bragg)
                    if max_relative>0 and min_relative<0:
                        facets_involved[id_facet] = True
                        break
            # print(facets_involved)
            if not any(facets_involved):
                # print('not any')
                wavelength_range_list.append(None)
                theta_phi_list.append(None)
                continue
            involved_facets = angular_mesh.facets[facets_involved]
            involved_nodes = []
            for id_facet in range(involved_facets.shape[0]):
                for i in range(3):
                    # print(angular_mesh.facets[id_facet,i],involved_nodes)
                    if angular_mesh.facets[id_facet,i] not in involved_nodes:
                        involved_nodes.append(angular_mesh.facets[id_facet,i])
            nodes_involved = []
            # print(node_rs)
            # print(involved_nodes)
            for nodetag in involved_nodes:
                nodes_involved.append(angular_mesh.nodes[nodetag])
                
            if len(nodes_involved) != 0:
                plane_center = np.average(np.array(nodes_involved),axis=0)
                encf_ptop = Project_To_Plane(plane_center/np.linalg.norm(plane_center),np.zeros(3))
                encoder_ptop = Mesh_Encoder([encf_ptop])
                encoder_ptop.Encode(angular_mesh)
            
            nodes_involved = []
            for nodetag in involved_nodes:
                nodes_involved.append(angular_mesh.nodes[nodetag])
            encoder_ptop.Decode(angular_mesh)
            for nodetag in involved_nodes:
                wavelength_func = angular_mf.values[nodetag]
                typename = get_type_name(wavelength_func)
                if 'Zero' in typename:
                    continue
                max_w = wavelength_func.domain[1]
                min_w = wavelength_func.domain[0]
                max_w_list.append(max_w)
                min_w_list.append(min_w)
            if len(max_w_list) == 0:
                wavelength_range_list.append(None)
                theta_phi_list.append(None)
                continue
            nodes_involved = np.array(nodes_involved)
            dz_getphi = norm_vector
            plane_center = np.average(nodes_involved,axis=0)
            dy_getphi = plane_center-norm_vector*np.dot(plane_center,norm_vector)/np.dot(norm_vector,norm_vector)
            dy_getphi = dy_getphi/np.linalg.norm(dy_getphi)
            dx_getphi = np.cross(dy_getphi,dz_getphi)
            # print(dz_getphi,dx_getphi)
            phi_list = []
            theta_list = []
            for node_id in range(nodes_involved.shape[0]):
                theta = np.arccos(np.dot(nodes_involved[node_id],dz_getphi)/np.linalg.norm(nodes_involved[node_id])/np.linalg.norm(dz_getphi))
                theta_list.append(theta)
                phi = Get_Phi(dz_getphi,dx_getphi,nodes_involved[node_id])
                phi_list.append(phi)
            max_wavelength_node_rs = max(max_w_list)
            min_wavelength_node_rs = min(min_w_list)
            wavelength_range_list.append([min_wavelength_node_rs,max_wavelength_node_rs])
            # print(min(theta_list),max(theta_list),min_wavelength_node_rs,max_wavelength_node_rs)
            theta_range = self.Get_Theta_Range(min(theta_list),max(theta_list),min_wavelength_node_rs,max_wavelength_node_rs)
            # print(theta_range)
            phi_range = self.Get_Phi_Range(min(phi_list),max(phi_list),(theta_range[0]+theta_range[1])/2)
            # print(min(phi_list),max(phi_list),phi_range)
            # print(nodes_involved,dz_getphi,dx_getphi)
            max_d_phi_an_mesh = np.maximum(max_d_phi_an_mesh,phi_range[1]-phi_range[0])
            max_d_theta_an_mesh = np.maximum(max_d_theta_an_mesh,theta_range[1]-theta_range[0])
            theta_phi_list.append([dz_getphi,dx_getphi,(theta_range[0]+theta_range[1])/2,(phi_range[0]+phi_range[1])/2])
            # print(theta_range[0],theta_range[1],phi_range[0],phi_range[1])
        # print(theta_phi_list)
        alpha = max_d_phi_an_mesh/max_d_theta_an_mesh
        avg_theta = np.average(np.array([theta_phi_list[x][2] for x in range(re_sp_mesh.num_nodes) if theta_phi_list[x] is not None]))
        avg_phi = np.average(np.array([theta_phi_list[x][2] for x in range(re_sp_mesh.num_nodes) if theta_phi_list[x] is not None]))
        # print(theta_phi_list)
        # print(avg_theta,avg_phi,alpha)
        base_angular_mesh = Generate_Rectangular_Mesh(np.array([1.,avg_theta,avg_phi]),np.array([1.,0.,0.]),0.,[1.2*max_d_theta_an_mesh,1.2*max_d_theta_an_mesh],max_d_theta_an_mesh/dtheta_factor)
        # print(base_angular_mesh.nodes)
        encf_extend = Extend_along_a_Direction(np.array([1.,avg_theta,avg_phi]),np.array([0.,0.,1.]),alpha)
        encf_projection = Project_from_R_Theta_Phi()
        encoder = Mesh_Encoder([encf_extend,encf_projection])
        encoder.Encode(base_angular_mesh)
        # print(base_angular_mesh.nodes,avg_theta,avg_phi)
        encf_reset_position = Position_Transform(np.array([0.,0.,0.]),np.array([np.sin(avg_theta)*np.cos(avg_phi),np.sin(avg_theta)*np.sin(avg_phi),np.cos(avg_theta)]),0)
        encoder_reset = Mesh_Encoder([encf_reset_position])
        encoder_reset.Decode(base_angular_mesh)
        # Visualize_Mesh(base_angular_mesh)

        involved_spnodes = np.zeros(re_sp_mesh.num_nodes,dtype=bool)
        an_ori_gam = []
        nodetags = []
        for id_spatial in range(re_sp_mesh.num_nodes):
            if theta_phi_list[id_spatial] is not None:
                theta_phi = theta_phi_list[id_spatial]
                # print(theta_phi)
                # ori2,gam2 = GetEulerAngleFromDxDy(theta_phi[1],np.cross(theta_phi[0],theta_phi[1]))
                dx_ref,dy_ref = GetDxDyFromEulerAngle(np.array([np.sin(theta_phi[2])*np.cos(theta_phi[3]),np.sin(theta_phi[2])*np.sin(theta_phi[3]),np.cos(theta_phi[2])]),0)
                dx = dx_ref[0]*theta_phi[1]+dx_ref[1]*np.cross(theta_phi[0],theta_phi[1])+dx_ref[2]*theta_phi[0]
                dy = dy_ref[0]*theta_phi[1]+dy_ref[1]*np.cross(theta_phi[0],theta_phi[1])+dy_ref[2]*theta_phi[0]
                ori,gam = GetEulerAngleFromDxDy(dx,dy)
                if gam>np.pi:
                    gam = gam-2*np.pi
                # print(dx_ref,dy_ref,ori,gam)
                an_ori_gam.append((ori,gam))
                involved_spnodes[id_spatial] = True
                nodetags.append(id_spatial)
            else:
                an_ori_gam.append(None)
        node_from_tags = dict(zip(nodetags,list(range(len(nodetags)))))
        facets = []
        for i in range(re_sp_mesh.num_facets):
            if all([re_sp_mesh.facets[i,x] in nodetags for x in range(3)]):
                facets.append([node_from_tags[re_sp_mesh.facets[i,x]] for x in range(3)])
        facets = np.array(facets)
        nodes = np.array([re_sp_mesh.nodes[nodetags[x]] for x in range(len(nodetags))])
        
        extrap_mesh = XRSA_Mesh(nodes,facets)
        extrap_mesh.bias_vector = copy.deepcopy(re_sp_mesh.bias_vector)
        extrap_mesh.norm = copy.deepcopy(re_sp_mesh.norm)
        # ax = Visualize_Mesh(extrap_mesh,re_sp_mesh,hold=True)
        # for i in range(len(nodetags)):
        #     node = re_sp_mesh.nodes[nodetags[i]]
        #     ax.scatter([node[0]],[node[1]],[node[2]],color='orange')
        # plt.show()
        
        ori_meshfunc = XRSA_MeshFunc_Scalar(extrap_mesh)
        ori_meshfunc.values = [an_ori_gam[nodetags[x]][0] for x in range(len(nodetags))]
        gam_meshfunc = XRSA_MeshFunc_Scalar(extrap_mesh)
        gam_meshfunc.values = [an_ori_gam[nodetags[x]][1] for x in range(len(nodetags))]
        w_func_min_mf = XRSA_MeshFunc_Scalar(extrap_mesh)
        w_func_min_mf.values = [wavelength_range_list[nodetags[x]][0] for x in range(len(nodetags))]
        w_func_max_mf = XRSA_MeshFunc_Scalar(extrap_mesh)
        w_func_max_mf.values = [wavelength_range_list[nodetags[x]][1] for x in range(len(nodetags))]
        not_involved_nodes = np.array(list(range(re_sp_mesh.num_nodes)))[[not involved_spnodes[i] for i in range(involved_spnodes.shape[0])]].tolist()
        # print(ori_meshfunc.refmesh.num_facets,ori_meshfunc.refmesh.num_nodes)
        for id_spatial in not_involved_nodes:
            # print(re_sp_mesh.nodes[id_spatial])
            an_ori_gam[id_spatial]=(ori_meshfunc(re_sp_mesh.nodes[id_spatial],padding=True,debug=False),gam_meshfunc(re_sp_mesh.nodes[id_spatial],padding=True))
            wavelength_range_list[id_spatial] = (w_func_min_mf(re_sp_mesh.nodes[id_spatial],padding=True),w_func_max_mf(re_sp_mesh.nodes[id_spatial],padding=True))
        # test_origam = []
        # for i in range(re_sp_mesh.num_nodes):
        #     print(an_ori_gam[i])
        #     test_origam.append((an_ori_gam[i][0],an_ori_gam[i][1]+np.pi))
        
        # test = XRSA_JointDistribution(re_sp_mesh,base_angular_mesh,an_ori_gam)
        # for i in range(test.refmesh.num_nodes):
        #     an_mesh = test.values[i].refmesh
        #     norm_vec = test.refmesh.norm(test.refmesh.nodes[i])
        #     for j in range(an_mesh.num_nodes):
        #         an_mesh.nodes[j] = get_mirror_vector(norm_vec,an_mesh.nodes[j])
        #     Visualize_Mesh(an_mesh,in_jo_dist.values[i].refmesh)
        
        #     Visualize_Mesh(test.values[i].refmesh,in_jo_dist.values[i].refmesh)
        # base_angular_mesh,an_ori_gam,wavelength_range_list = self.Update_re_an_mesh(incident_name,re_sp_mesh,base_angular_mesh,an_ori_gam,wavelength_range_list)
        return base_angular_mesh,an_ori_gam,wavelength_range_list

    def Get_Theta_Range(self,min_theta_ia,max_theta_ia,min_w,max_w):
        window = self.Get_Window()
        min_theta = max(min_theta_ia,self.Bragg_theta(max_w)-window[1])
        max_theta = min(max_theta_ia,self.Bragg_theta(min_w)-window[0])
        return [min_theta,max_theta]

    def Get_Phi_Range(self,min_phi_ia,max_phi_ia,_):
        return [min_phi_ia+np.pi,max_phi_ia+np.pi]

class XRSA_Mosaic_Crystal(XRSA_Crystal):
    def __init__(self, spatial_mesh, spacing, rocking_curve=None, mosaic_spread=np.radians(1.), max_reflectivity=0.25, name='Default'):
        super().__init__(spatial_mesh, spacing, rocking_curve, name)
        self.mosaic_spread = mosaic_spread
        self.mosaic_dist = XRSA_LinearFunc(lambda x: Gaussian(x,max_reflectivity*np.sqrt(2*np.pi)*(mosaic_spread/2.355),0.,mosaic_spread/2.355),-2*mosaic_spread,2*mosaic_spread)

    def Get_Window(self):
        return [self.mosaic_dist.domain[0]+self.rc_func.domain[0],self.mosaic_dist.domain[1]+self.rc_func.domain[1]]

    def Response(self, incident_name):
        super().Response(incident_name)
        in_jo_dist = self.incident[incident_name]
        re_sp_mesh = copy.deepcopy(in_jo_dist.refmesh)
        # Visualize_Mesh(re_sp_mesh)
        re_sp_mesh = self.Initialize_re_sp_mesh(in_jo_dist,re_sp_mesh)
        # Visualize_Mesh(re_sp_mesh,in_jo_dist.refmesh)
        re_an_basemesh, an_ori_gam, w_range_list = self.Initialize_re_an_mesh(in_jo_dist,re_sp_mesh,dtheta_factor=8)
        # print(an_ori_gam)
        re_jo_dist = XRSA_JointDistribution(re_sp_mesh,re_an_basemesh,an_ori_gam)
        re_sp_mesh = re_jo_dist.refmesh

        for id_spatial in range(re_sp_mesh.num_nodes):
            print('response: %u/%u'%(id_spatial,re_sp_mesh.num_nodes))
            node_rs = re_sp_mesh.nodes[id_spatial]
            re_an_mf = re_jo_dist.values[id_spatial]
            re_an_mesh = re_an_mf.refmesh
            in_an_mf = in_jo_dist(node_rs)
            # Visualize_Mesh(re_an_mesh,in_an_mf.refmesh)
            w_range = w_range_list[id_spatial]
            norm_vec = re_sp_mesh.norm(node_rs)
            
            # # print(in_an_mf.refmesh.nodes)
            # plot_in_an_mesh = copy.deepcopy(in_an_mf.refmesh)
            # plot_re_an_mesh = copy.deepcopy(re_an_mesh)
            # postrans = Position_Transform(node_rs,np.array([0.,0.,1.]),0)
            # # zoom = Zoom_in(np.zeros(3),10)
            # encoder = Mesh_Encoder([postrans])
            # encoder.Encode(plot_in_an_mesh)
            # encoder.Encode(plot_re_an_mesh)
            
            # Visualize_Mesh(plot_in_an_mesh, re_sp_mesh, plot_re_an_mesh)
            # plot_in_an_mesh = copy.deepcopy(in_an_mf.refmesh)
            # reflect = Reflect_Against_Norm(norm_vec)
            # encoder = Mesh_Encoder([reflect])
            # encoder.Encode(plot_in_an_mesh)
            # plot_in_an_domains_max = XRSA_MeshFunc_Scalar(plot_in_an_mesh)
            # plot_in_an_domains_min = XRSA_MeshFunc_Scalar(plot_in_an_mesh)
            # intensity_in_an = XRSA_MeshFunc_Scalar(plot_in_an_mesh)
            # for i in range(plot_in_an_mesh.num_nodes):
            #     plot_in_an_domains_max.values[i] = in_an_mf.values[i].domain[1]
            #     plot_in_an_domains_min.values[i] = in_an_mf.values[i].domain[0]
            #     intensity_in_an.values[i] = in_an_mf.values[i].Total_Integration()
            # plot_in_an_domains_max.Display_Pattern()
            # plot_in_an_domains_min.Display_Pattern()
            # intensity_in_an.Display_Pattern()
            
            # plot_re_an_domains_max = XRSA_MeshFunc_Scalar(re_an_mesh)
            # plot_re_an_domains_min = XRSA_MeshFunc_Scalar(re_an_mesh)
            # for i in range(re_an_mesh.num_nodes):
            #     node_ra = re_an_mesh.nodes[i]
            #     base_theta = np.arccos(np.dot(norm_vec,node_ra)/np.linalg.norm(norm_vec)/np.linalg.norm(node_ra))
            #     dw_dtheta = -2*self.spacing*np.cos(np.pi/2-base_theta)
            #     base_w = self.Bragg_wavelength(base_theta)
            #     plot_re_an_domains_max.values[i] = base_w+dw_dtheta*self.rc_func.domain[0]
            #     plot_re_an_domains_min.values[i] = base_w+dw_dtheta*self.rc_func.domain[1]
            # plot_re_an_domains_max.Display_Pattern()
            # plot_re_an_domains_min.Display_Pattern()
            
            # ax = Visualize_Mesh(re_sp_mesh,hold=True)
            # ax.scatter([node_rs[0]],[node_rs[1]],[node_rs[2]],color='green')
            # plt.show()
            
            for id_angular in range(re_an_mesh.num_nodes):
                print('response_angular: %u/%u'%(id_angular,re_an_mesh.num_nodes))
                node_ra = re_an_mesh.nodes[id_angular]
                new_wavelength_func = self.Dispersion(in_an_mf,norm_vec,node_ra,w_range,debug=False)
                # if new_wavelength_func.Total_Integration() > 1e-8:
                #     # plot_in_an_mesh = copy.deepcopy(in_an_mf.refmesh)
                #     # reflect = Reflect_Against_Norm(norm_vec)
                #     # encoder = Mesh_Encoder([reflect])
                #     # encoder.Encode(plot_in_an_mesh)
                #     ax = Visualize_Mesh(plot_in_an_mesh,re_an_mesh,hold=True)
                #     ax.scatter([node_ra[0]],[node_ra[1]],[node_ra[2]],color='orange')
                #     plt.show()
                #     wfunc = self.Dispersion(in_an_mf,norm_vec,node_ra,w_range,debug=True)
                re_an_mf.values[id_angular] = new_wavelength_func
            # re_an_mf.Display_Pattern(projection=True)
        response_name = incident_name+"_"+self.name
        self.response[response_name] = re_jo_dist

    def Dispersion(self,in_an_mf,norm_vec,node_ra,w_range,theta_factor=9,phi_factor=7,debug=False):
        # st_time = time.time()
        window = self.Get_Window()
        # print(window[1]-window[0],np.max(self.mosaic_dist.domains[:,1]),np.min(self.mosaic_dist.domains[:,0]))
        # print(norm_vec,node_ra)
        ic_angle = np.arccos(np.dot(norm_vec,node_ra)/np.linalg.norm(norm_vec)/np.linalg.norm(node_ra))
        # print(ic_angle)
        max_theta_dispersional = self.Bragg_theta(w_range[0])
        min_theta_dispersional = self.Bragg_theta(w_range[1])
        inner_products = np.array([np.dot(in_an_mf.refmesh.nodes[i],norm_vec)/np.linalg.norm(in_an_mf.refmesh.nodes[i])/np.linalg.norm(norm_vec) for i in range(in_an_mf.refmesh.num_nodes)])
        min_theta_spatial = np.arccos(np.max(inner_products))
        max_theta_spatial = np.arccos(np.min(inner_products))
        # print(ic_angle,self.Bragg_wavelength(ic_angle+window[0]),min_theta_spatial)
        # print(2*min_theta_dispersional,self.Bragg_theta(self.Bragg_wavelength(ic_angle+window[0]))*2,min_theta_spatial+ic_angle)
        min_theta = np.max(np.array([2*min_theta_dispersional,self.Bragg_theta(self.Bragg_wavelength(ic_angle+window[0]))*2,min_theta_spatial+ic_angle]))
        # flag_mintheta = np.argmax(np.array([2*min_theta_dispersional,self.Bragg_theta(self.Bragg_wavelength(ic_angle+window[0]))*2,min_theta_spatial+ic_angle]))
        # print(2*max_theta_dispersional,self.Bragg_theta(self.Bragg_wavelength(ic_angle+window[1]))*2,max_theta_spatial+ic_angle)
        max_theta = np.min(np.array([2*max_theta_dispersional,self.Bragg_theta(self.Bragg_wavelength(ic_angle+window[1]))*2,max_theta_spatial+ic_angle]))
        if max_theta<min_theta:
            return XRSA_Zero_Wavelength_Func()
        # flag_maxtheta = np.argmin(np.array([2*max_theta_dispersional,self.Bragg_theta(self.Bragg_wavelength(ic_angle+window[1]))*2,max_theta_spatial+ic_angle]))
        # ratio = (max_theta-min_theta)/(max(self.rc_func.domains[:][1])-min(self.rc_func.domains[:][0]))
        phi_base_theta =  min_theta/2
        phi_range_mosaic = np.arcsin(np.sin(np.radians(4*self.mosaic_spread))/np.sin(phi_base_theta))
        dz_decode = node_ra/np.linalg.norm(node_ra)
        dx_decode = norm_vec-node_ra*np.dot(node_ra,norm_vec)/np.dot(node_ra,node_ra)
        dx_decode = dx_decode/np.linalg.norm(dx_decode)
        dy_decode = np.cross(dz_decode,dx_decode)
        dy_decode = dy_decode/np.linalg.norm(dy_decode)
        for i in range(in_an_mf.refmesh.num_nodes):
            node_ia = in_an_mf.refmesh.nodes[i]
            node_ia = node_ia/np.linalg.norm(node_ia)
        phi_list = [Get_Phi(dz_decode,dx_decode,in_an_mf.refmesh.nodes[i]) for i in range(in_an_mf.refmesh.num_nodes)]
        # for i in range(in_an_mf.refmesh.num_nodes):
        #     print(in_an_mf.refmesh.nodes[i],node_ra)
        #     print(dz_decode,dx_decode)
        #     print(phi_list[i])
        # print(phi_list)
        max_phi_spatial = max(phi_list)
        min_phi_spatial = min(phi_list)
        
        theta_grid = np.linspace(min_theta/2,max_theta/2,theta_factor)
        phi_grid = np.linspace(min_phi_spatial,max_phi_spatial,phi_factor)
        rc_func_conv = XRSA_LinearFunc(lambda y: self.rc_func(-y/2),-2*self.rc_func.domain[1],-2*self.rc_func.domain[0])
        ratio = (max_theta-min_theta)/(self.rc_func.domain[1]-self.rc_func.domain[0])/2
        integ_rc = rc_func_conv.Total_Integration()
        # phi_norm_factor = np.zeros((theta_factor,phi_factor))
        A_factor = self.mosaic_dist(ic_angle-(min_theta+max_theta)/4)/(np.sin((min_theta+max_theta)/2)*integ_rc*integrate.quad(lambda phi:self.mosaic_dist(np.sqrt(np.power(ic_angle-(min_theta+max_theta)/4,2)+np.power(phi,2))), -phi_range_mosaic/2,phi_range_mosaic/2)[0])
        # for i in range(theta_factor):
        #     for j in range(phi_factor):
        #     # print(integrate.quad(lambda phi:self.mosaic_dist(np.arcsin(np.sin(theta_grid[i])*np.sin(phi))),-phi_range_mosaic/2,phi_range_mosaic/2))
        #         phi_norm_factor[i,j] = self.mosaic_dist(np.sqrt(np.power(ic_angle-theta_grid[i],2)+np.power(phi_grid[j],2)))/(integ_rc*np.sin(2*theta_grid[i]))/integrate.quad(lambda phi:self.mosaic_dist(np.arcsin(np.sin(theta_grid[i])*np.sin(phi))),-phi_range_mosaic/2,phi_range_mosaic/2)[0]
        # print(phi_norm_factor)
        
        base_w_list = np.array([self.Bragg_wavelength(theta_grid[i]) for i in range(theta_factor)])
        w_vals = np.zeros(theta_factor)
        in_wfuncs = []
        # print(theta_grid,phi_grid)
        # ed_time = time.time()
        # print(ed_time-st_time)
        # st_time = time.time()
        # print(dx_decode,dy_decode,dz_decode)
        # ax = Visualize_Mesh(in_an_mf.refmesh,hold=True)
        
        # cpu
        for i in range(theta_factor):
            wfuncs_theta = []
            for j in range(phi_factor):
                node_ref_ra = np.array([np.sin(2*theta_grid[i])*np.cos(phi_grid[j]),np.sin(2*theta_grid[i])*np.sin(phi_grid[j]),np.cos(2*theta_grid[i])])
                # print(theta_grid[i],phi_grid[j])
                node = node_ref_ra[0]*dx_decode+node_ref_ra[1]*dy_decode+node_ref_ra[2]*dz_decode
                # ax = Visualize_Mesh(in_an_mf.refmesh,hold=True)
                # ax.scatter([node[0]],[node[1]],[node[2]])
                # plt.show()
                # print(in_an_mf(node).domains)
                # node_list.append(node)
                # print(i,j)
                wfuncs_theta.append(in_an_mf(node,debug=False,use_cuda=True))
                # print(j,wfuncs_theta[j].Total_Integration())
                # print(in_an_mf(node))
                # ax.scatter([node[0]],[node[1]],node[2],color = 'orange')
                # plt.show()
            in_wfuncs.append(wfuncs_theta)
        # gpu
        # node_list = []
        # for i in range(theta_factor):
        #     # wfuncs_theta = []
        #     for j in range(phi_factor):
        #         node_ref_ra = np.array([np.sin(2*theta_grid[i])*np.cos(phi_grid[j]),np.sin(2*theta_grid[i])*np.sin(phi_grid[j]),np.cos(2*theta_grid[i])])
        #         # print(theta_grid[i],phi_grid[j])
        #         node = node_ref_ra[0]*dx_decode+node_ref_ra[1]*dy_decode+node_ref_ra[2]*dz_decode
        #         # ax = Visualize_Mesh(in_an_mf.refmesh,hold=True)
        #         # ax.scatter([node[0]],[node[1]],[node[2]])
        #         # plt.show()
        #         # print(in_an_mf(node).domains)
        #         node_list.append(node)
        #         # wfuncs_theta.append(in_an_mf(node,debug=debug,use_cuda=True))
        #         # print(in_an_mf(node))
        #         # ax.scatter([node[0]],[node[1]],node[2],color = 'orange')
        #         # plt.show()
        #     # in_wfuncs.append(wfuncs_theta)
        # in_an_mf.refmesh.Send_to_gpu()
        # target_facet_ids = in_an_mf.refmesh.Get_Target_Facets_Cuda(node_list)
        # # target_facet_ids = target_facet_ids.reshape((theta_factor,phi_factor))
        # in_an_mf.refmesh.Delete_from_gpu()
        # for i in range(theta_factor):
        #     wfunc_theta = []
        #     for j in range(phi_factor):
        #         facet_id, di_array = target_facet_ids[i*phi_factor+j]
        #         print(i,j)
        #         print(facet_id,di_array)
        #         if facet_id is None:
        #             wfunc_theta.append(XRSA_Zero_Wavelength_Func())
        #         else:
        #             total_di = sum(di_array)
        #             di_array = di_array/total_di
        #             wfunc_theta.append(in_an_mf.Triple_Wavedist_Addition(facet_id,di_array))
        #             # new_wf.Merge_Values()
        #     in_wfuncs.append(wfunc_theta)
        
        
        # plt.show()
        # ed_time = time.time()
        # print(ed_time-st_time)
        # st_time = time.time()
        for i in range(theta_factor):
            #wavelength
            base_w = base_w_list[i]
            total_integ_tp = 0
            for j in range(phi_factor):
                #phi
                fw_val = np.zeros(theta_factor)
                for k in range(theta_factor):
                    #theta
                    fw_val[k] = in_wfuncs[k][j](base_w)
                    # print(i,j,k)
                    # print(in_wfuncs[k][j].domains,base_w,fw_val[k])
                x_array = np.array([theta_grid[l]*2 for l in range(theta_factor)])
                F1 = interp1d(x_array,fw_val,kind='linear',fill_value=0, bounds_error=False)
                F_in_theta = XRSA_LinearFunc(F1,min_theta,max_theta)
                node_ref_ra = np.array([np.sin(2*theta_grid[i])*np.cos(phi_grid[j]),np.sin(2*theta_grid[i])*np.sin(phi_grid[j]),np.cos(2*theta_grid[i])])
                node = node_ref_ra[0]*dx_decode+node_ref_ra[1]*dy_decode+node_ref_ra[2]*dz_decode
                norm_crystallite = (node+node_ra)/np.linalg.norm(node+node_ra)
                angle_bias = np.arccos(np.dot(norm_crystallite,norm_vec)/np.linalg.norm(norm_crystallite)/np.linalg.norm(norm_vec))
                # print(angle_bias)
                # F_in_theta.plot_func()
                if ratio>20:
                    convolved_phit = F_in_theta(theta_grid[i]*2)*integ_rc*A_factor*self.mosaic_dist(angle_bias)*np.sin(2*theta_grid[i])
                elif ratio<0.05:
                    convolved_phit = integrate.quad(lambda x: F_in_theta(x),min_theta,max_theta)[0]*rc_func_conv((max_theta+min_theta)/2-2*theta_grid[i])*A_factor*self.mosaic_dist(angle_bias)*np.sin(2*theta_grid[i])
                else:
                    convolved_phit = integrate.quad(lambda x: F_in_theta(x)*rc_func_conv(x-2*theta_grid[i]),min_theta,max_theta)[0]*A_factor*self.mosaic_dist(angle_bias)*np.sin(2*theta_grid[i])
                total_integ_tp= total_integ_tp+convolved_phit*(max_phi_spatial-min_phi_spatial)/phi_factor
                # print(convolved_phit,integ_rc,phi_norm_factor[i],total_integ_tp)
            w_vals[i] = total_integ_tp
        # print(base_w_list,w_vals,base_w_list[0],base_w_list[-1])
        F_wfunc = interp1d(base_w_list[::-1],w_vals[::-1],fill_value=0, bounds_error=False)
        # print(F_wfunc(base_w_list[2]))
        epsilon = (base_w_list[0]-base_w_list[-1])*1e-6
        # print(base_w_list[::-1],w_vals[::-1],base_w_list[-1]+epsilon,base_w_list[0]-epsilon)
        new_wfunc = XRSA_Wavelength_Func(F_wfunc,base_w_list[-1]+epsilon,base_w_list[0]-epsilon)
        if new_wfunc.Total_Integration() is None:
            print(base_w_list)
            print(w_vals)
            print(base_w_list[-1]+epsilon,base_w_list[0]-epsilon)
            new_wfunc = XRSA_Zero_Wavelength_Func()
        if debug:
            print(new_wfunc.Get_moments(),new_wfunc.Total_Integration())
            new_wfunc.plot_func()
        # print(base_w_list[::-1],new_wfunc.domains)
        # new_wfunc.plot_func()
        # ed_time = time.time()
        # print(ed_time-st_time)
        return new_wfunc

    def Get_Theta_Range(self, min_theta_ia, max_theta_ia, min_w, max_w):
        window = self.Get_Window()
        min_theta = np.maximum(2*self.Bragg_theta(max_w)-max_theta_ia,self.Bragg_theta(max_w)-window[1])
        max_theta = np.minimum(2*self.Bragg_theta(min_w)-min_theta_ia,self.Bragg_theta(min_w)-window[0])
        return [min_theta,max_theta]

    def Get_Phi_Range(self, min_phi_ia, max_phi_ia, _):
        return [min_phi_ia-self.mosaic_spread*1.5+np.pi,max_phi_ia+self.mosaic_spread*1.5+np.pi]

class XRSA_Spherical_Crystal(XRSA_Crystal):
    def __init__(self, spatial_mesh, spacing, curvature_radius, rocking_curve=None, name='Default'):
        super().__init__(spatial_mesh, spacing, rocking_curve, name)
        self.curvature_radius = curvature_radius
        
def Generate_Spherical_Crystal(coordinate,orientation,gamma,shape,lc,spacing,curvature_radius,rc_curve,name='Default'):
    mesh = Generate_Rectangular_Mesh(coordinate,orientation,gamma,shape,lc)
    curvature_center = coordinate+orientation*curvature_radius
    encf_ptop = Project_To_Plane(plane_center=coordinate,base_point=curvature_center)
    encoder_ptop = Mesh_Encoder([encf_ptop])
    encoder_ptop.Decode(mesh)
    mesh.ptop_encoder = encoder_ptop
    mesh.norm = Generate_normal_function_spherical_surface(curvature_center)
    return XRSA_Spherical_Crystal(mesh,spacing,curvature_radius,rc_curve,name)

if __name__ == '__main__':
    omega = 36.063
    source_coordinate = np.array([-1.0*np.cos(np.radians(omega)),0,1.0*np.sin(np.radians(omega))])
    source_orientation = np.array([np.cos(np.radians(omega)),0,-np.sin(np.radians(omega))])
    wavelength_func = XRSA_Wavelength_Func(lambda x: Gaussian(x, 1., 3.95, 0.002),3.945,3.955)
    source = Generate_Pinhole_Isotropic_Source(source_coordinate,source_orientation,1.0,wavelength_func)
    # for i in range(source.response['original0'].refmesh.num_nodes):
    #     print('check')
    #     ax = Visualize_Mesh(source.response['original0'].refmesh,hold=True)
    #     nodes = source.response['original0'].refmesh.nodes[i]
    #     ax.scatter(nodes[0],nodes[1],nodes[2],color='orange')
    #     plt.show()
    #     integ = source.response['original0'].values[i].Total_Integration()
    #     print(integ)
    detector_coordinate = np.zeros(3)
    detector_orientation = np.array([0.,0.,1.])
    detector_shape = np.array([0.1,0.1])
    # source.response['original0'].Visualize_Spatial_Distribution()
    detector_mesh = Generate_Rectangular_Mesh(detector_coordinate,detector_orientation,0,detector_shape,0.02)
    # print(detector_mesh.nodes)

    in_jo_dist = Transmission(source.response['original0'],detector_mesh)
    in_jo_dist.Visualize_Spatial_Distribution()