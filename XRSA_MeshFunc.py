import numpy as np
import gmsh
import sys
from func_lib import *
from cuda_lib import *
import copy
from scipy.interpolate import *
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from scipy.linalg import *
from scipy.optimize import curve_fit
from XRSA_LinearFunc import *
import time
import pickle


class XRSA_Mesh():
    mshtag = 0
    def __init__(self,nodes,facets):
        self.num_nodes = nodes.shape[0]
        self.nodes = nodes
        self.num_facets = facets.shape[0]
        self.facets = facets
        self.norm = None
        self.facet_det = None
        self.facet_area = None
        XRSA_Mesh.mshtag += 1
        self.model_name = 'msh_%u'%(XRSA_Mesh.mshtag)
        # Don't operate on the values below unless necessary.
        self.bias_vector = np.zeros(3)
        self.ptop_encoder = None
        self.handle = None
    
    def Get_Facet_Det_cuda(self):
        matrices = []
        for i in range(self.num_facets):
            matrix = np.array([self.nodes[self.facets[i,x]]-self.bias_vector for x in range(3)],dtype=np.float32)
            matrices.append(matrix)
        self.facet_det = calculate_determinants(matrices)

        # result check
        for i in range(np.size(self.facet_det)):
            if np.abs(self.facet_det[i])<1e-14:
                print('warning: determinant approx. to 0. det = %.04e. Check the bias vector: [%.04f, %.04f, %.04f]'%(self.facet_det[i],self.bias_vector[0],self.bias_vector[1],self.bias_vector[2]))
    
    def Get_Area_And_Normal_of_Facets(self):
        self.facet_area = np.zeros(self.num_facets)
        self.facet_normal = np.zeros((self.num_facets,3))
        for i in range(self.num_facets):
            p1 = self.nodes[self.facets[i,0]]
            p2 = self.nodes[self.facets[i,1]]
            p3 = self.nodes[self.facets[i,2]]
            self.facet_area[i], self.facet_normal[i,:] = GetAreaAndNormal(p1,p2,p3)

    def GetTotalArea(self):
        if self.facet_area is None:
            self.Get_Area_And_Normal_of_Facets()
        return np.sum(self.facet_area)
    
    def Get_Facet_Det(self):
        self.facet_det = np.zeros(self.num_facets)
        for i in range(self.num_facets):
            p1 = self.nodes[self.facets[i,0]]-self.bias_vector
            p2 = self.nodes[self.facets[i,1]]-self.bias_vector
            p3 = self.nodes[self.facets[i,2]]-self.bias_vector
            deta = Triangular_Det(p1,p2,p3) # Notice this sequence. Cannot change.
            if np.abs(deta)>1e-14:
                self.facet_det[i] = deta
            else:
                print(deta)
                print("Check the bias_vector: [%.04f, %.04f, %.04f]"%(self.bias_vector[0],self.bias_vector[1],self.bias_vector[2]))
        
    def GetNodesFromFacet(self,id_facet):
        return np.array([self.nodes[self.facets[id_facet,x]] for x in range(3)])

    def Get_Target_Facet(self,p,padding=False,debug=False,use_cuda = False):
        return self.Get_Target_Facet_cpu(p,padding=padding,debug=debug)
        # if use_cuda:
        #     if self.handle is None:
        #         # print('sending to gpu')
        #         self.Send_to_gpu()
        #     return self.Get_Target_Facet_gpu(p,padding=padding,debug=debug)
        # else:
        #     return self.Get_Target_Facet_cpu(p,padding=padding,debug=debug)
    
    def Get_Target_Facet_cpu(self,p,padding=False,debug=False):
        matrices = []
        for i in range(self.num_facets):
            matrices=matrices+[np.array([self.nodes[self.facets[i,x]]-self.bias_vector for x in range(3)],dtype=np.float32),
                               np.array([p-self.bias_vector]+[(self.nodes[self.facets[i,(x+1)%3]]-self.bias_vector) for x in range(2)],dtype=np.float32),
                               np.array([p-self.bias_vector]+[(self.nodes[self.facets[i,(x+2)%3]]-self.bias_vector) for x in range(2)],dtype=np.float32),
                               np.array([p-self.bias_vector]+[(self.nodes[self.facets[i,(x+3)%3]]-self.bias_vector) for x in range(2)],dtype=np.float32)]
        d_array = calculate_determinants_cpu(matrices)
        if debug:
            print(p,self.bias_vector,self.nodes)
            print(matrices)
            print(d_array)
        d_array = np.array(d_array).reshape(self.num_facets,4)
        di_array = np.zeros((self.num_facets,3))
        for i in range(self.num_facets):
            # if debug and np.abs(d_array[i,0])<1e-11:
            #     print(d_array[i])
            di_array[i] = np.array([d_array[i,x+1]/d_array[i,0] for x in range(3)])
            if debug:
                if all(di_array[i]>-1e-7):
                    matrices_debug = [np.array([self.nodes[self.facets[i,x]]-self.bias_vector for x in range(3)],dtype=np.float32),
                               np.array([p-self.bias_vector]+[(self.nodes[self.facets[i,(x+1)%3]]-self.bias_vector) for x in range(2)],dtype=np.float32),
                               np.array([p-self.bias_vector]+[(self.nodes[self.facets[i,(x+2)%3]]-self.bias_vector) for x in range(2)],dtype=np.float32),
                               np.array([p-self.bias_vector]+[(self.nodes[self.facets[i,(x+3)%3]]-self.bias_vector) for x in range(2)],dtype=np.float32)]
                    print(matrices_debug)
                    d = [Triangular_Det(matrices_debug[x][0],matrices_debug[x][1],matrices_debug[x][2]) for x in range(4)]
                    ax = Visualize_Mesh(self,hold=True)
                    print(d)
                    print(d_array[i,0])
                    for k in range(3):
                        print(d_array[i,k+1])
                        ax.plot([self.nodes[self.facets[i,k]][0],self.nodes[self.facets[i,(k+1)%3]][0]],[self.nodes[self.facets[i,k]][1],self.nodes[self.facets[i,(k+1)%3]][1]],[self.nodes[self.facets[i,k]][2],self.nodes[self.facets[i,(k+1)%3]][2]],color='orange')
                    ax.scatter([p[0]],[p[1]],[p[2]],color='orange')
                    print(di_array[i])
                    plt.show()        
        one_negative_profile=[]
        target_facet_id = None
        for i in range(self.num_facets):
            di = di_array[i]
            # print(di_array[i])
            if all(di>-1e-7):
                target_facet_id = i
                break
            elif np.sum(di<-1e-7)==1:
                one_negative_profile.append([i,np.min(di)])
        if target_facet_id is None:
            if padding:
                one_negative_profile = np.array(one_negative_profile)
                arg_onenegative = np.argmax(one_negative_profile[:,1])
                target_facet_id = one_negative_profile[arg_onenegative,0].astype(int)
        if target_facet_id is not None:
            return target_facet_id,di_array[target_facet_id]
        else:
            return None,None
        
    # def Get_Target_Facet_cuda(self,p,padding=False,debug=False):
        
        
    def Update_Mesh(self,involved,dy_factor=10,debug=False):
        involved_facets = self.facets[involved]
        involved_nodes = []
        for i in range(involved_facets.shape[0]):
            for j in range(3):
                if involved_facets[i,j] not in involved_nodes:
                    involved_nodes.append(involved_facets[i,j])
        nodes = self.nodes[involved_nodes]
        nodes_from_tags = dict(zip(involved_nodes,list(range(len(involved_nodes)))))
        facets = np.zeros(involved_facets.shape,dtype=np.int8)
        for i in range(involved_facets.shape[0]):
            for j in range(3):
                facets[i,j] = nodes_from_tags[involved_facets[i,j]]
        # print(nodes,facets)
        extracted_mesh = XRSA_Mesh(nodes,facets)
        # Visualize_Mesh(extracted_mesh)
        plane_center = np.array([np.average(nodes[:,0]),np.average(nodes[:,1]),np.average(nodes[:,2])])
        dx,dy,dz,lx,ly,alpha = Eigen_CovMat(nodes)
        projected = []
        for i in range(nodes.shape[0]):
            rel_node = nodes[i]-plane_center
            projected.append([np.dot(rel_node,dx),np.dot(rel_node,dy),np.dot(rel_node,dz)])
        projected=np.array(projected)
        max_dx = np.max(projected[:,0])-np.min(projected[:,0])
        max_dy = np.max(projected[:,1])-np.min(projected[:,1])
        popt,_ = curve_fit(Symmetric_Quartic_Polynomial,projected[:,0],projected[:,1],
                           p0=[0,0,0,0],
                           bounds=([-10,-0.1,-10,-1.],[10,0.1,10,1.]))
        quartic_func = lambda x: Symmetric_Quartic_Polynomial(x,*popt)
        encf_projection = Projection_to_new_Coordinate(plane_center,dx,dy)
        encf_fitting_proj = Project_Along_a_Func(np.array([1.,0.,0.]),np.array([0.,1.,0.]),quartic_func)
        encf_compress = Extend_along_a_Direction(plane_center=np.zeros(3),direction=np.array([1.,0.,0.]),alpha=1/alpha) 
        encoder = Mesh_Encoder([encf_projection,encf_fitting_proj,encf_compress])
        encoder.Encode(extracted_mesh)
        
        max_dx = np.max(extracted_mesh.nodes[:,0])-np.min(extracted_mesh.nodes[:,0])
        max_dy = np.max(extracted_mesh.nodes[:,1])-np.min(extracted_mesh.nodes[:,1])
        new_center = np.array([(np.max(extracted_mesh.nodes[:,0])+np.min(extracted_mesh.nodes[:,0]))/2,
                               (np.max(extracted_mesh.nodes[:,1])+np.min(extracted_mesh.nodes[:,1]))/2,
                               (np.max(extracted_mesh.nodes[:,2])+np.min(extracted_mesh.nodes[:,2]))/2])
        new_mesh = Generate_Rectangular_Mesh(new_center,np.array([0.,0.,1.]),0,[1.2*max_dx,1.2*max_dy],max_dy/dy_factor)
        
        encoder.Decode(new_mesh)
        return new_mesh
    
    def Send_to_gpu(self):
        # Transfer mesh data to the GPU and store references
        d_nodes = cuda.to_device(self.nodes)
        d_facets = cuda.to_device(self.facets)
        d_bias_vector = cuda.to_device(self.bias_vector)
        self.handle = (d_nodes, d_facets, d_bias_vector)
        
    def Delete_from_gpu(self):
    # Check if GPU resources have been allocated
        if self.handle is not None:
            # Unpack device arrays from handle
            d_nodes, d_facets, d_bias_vector = self.handle
            
            # Free GPU memory by deleting each device array
            del d_nodes
            del d_facets
            del d_bias_vector
            
            # Synchronize to ensure memory is freed
            cuda.synchronize()
            
            # Clear the handle to indicate resources have been released
            self.handle = None
        else:
            print("No GPU data to delete. Call Send_to_gpu first if you intend to allocate.")


    ## High Computation Consumption. Use when necessary only.
    def Get_Edges(self):
        edges = []
        for i in range(self.num_facets):
            for j in range(3):
                e = [self.facets[i,j],self.facets[i,(j+1)%3]]
                if (e in edges) or (e[::-1] in edges):
                    continue
                else:
                    edges.append(e)
        self.edges = np.array(edges)
        self.num_edges = np.size(edges,axis=0)
    
    ## High Computation Consumption. Use when necessary only. 
    def Get_Boundary(self):
        boundary = []
        for i in range(self.num_edges):
            edge = self.edges[i]
            count = 0
            for j in range(self.num_facets):
                if (edge[0] in self.facets[j] and edge[1] in self.facets[j]):
                    count +=1
            # print(edge,count)
            # ax = Visualize_Mesh(self,hold=True)
            # ax.plot([self.nodes[self.edges[i][0]][0],self.nodes[self.edges[i][1]][0]],
            #         [self.nodes[self.edges[i][0]][1],self.nodes[self.edges[i][1]][1]],
            #         [self.nodes[self.edges[i][0]][2],self.nodes[self.edges[i][1]][2]],color='orange')
            # plt.show()
            if count == 2:
                continue
            elif count == 1:
                boundary.append(edge)
            else:
                print(edge,count)
                raise Exception("An edge should be used by either 1 or 2 facets.")
        prev = None
        # print(boundary)
        
        sorted = []
        while(len(boundary)!=0):
            if prev is not None:
                for j in range(len(boundary)):
                    if prev[1] not in boundary[j]:
                        NotFound = True
                        continue
                    elif boundary[j][0] == prev[1]:
                        sorted.append(boundary[j])
                        prev=boundary[j]
                        boundary.pop(j)
                        NotFound = False
                        break
                    else:
                        boundary[j] = boundary[j][::-1]
                        sorted.append(boundary[j])
                        prev=boundary[j]
                        boundary.pop(j)
                        NotFound = False
                        break
                if NotFound:
                    prev = None
                    # raise Exception('Multiple Loops not supported.')
            else:
                sorted.append(boundary[0])
                prev = boundary[0]
                boundary.pop(0)
                continue
        self.boundary = sorted

    def Get_Boundary_Points(self):
        boundary_points = []
        for id_node in self.boundary:
            boundary_points.append(self.nodes[id_node[0]])
        return boundary_points

    def Chop(self,bp,debug=False):
        print('chopping')
        n=len(bp)
        nodetags=np.arange(self.num_nodes)
        chopped_nodes = []
        chopped_facets = []
        for i in range(self.num_nodes):
            is_inside = Inside_a_Boundary(bp,self.nodes[i],bias_vector=self.bias_vector,debug=False)
            if not (is_inside and is_inside != 'on_boundary'):
                chopped_nodes.append(i)
                # ax = Visualize_Mesh(self,hold=True)
                # ax.scatter([self.nodes[i,0]],[self.nodes[i,1]],[self.nodes[i,2]],color='green')
                # for k in range(n):
                #     ax.plot([bp[k][0],bp[(k+1)%n][0]],[bp[k][1],bp[(k+1)%n][1]],[bp[k][2],bp[(k+1)%n][2]],color='orange')
                # plt.show()
                # stop_and_debug = input('stop and debug ?')
                # if stop_and_debug == 'y':
                #     is_inside = Inside_a_Boundary(bp,self.nodes[i],bias_vector=self.bias_vector,debug=True)
                #     print(is_inside)
                for j in range(self.num_facets):
                    facet_nodetags = self.facets[j].tolist()
                    if (i in facet_nodetags) and (j not in chopped_facets):
                        chopped_facets.append(j)
        q = len(chopped_facets)
        # print(chopped_facets)
        chopped_edges = []
        for idchoppedfacet in range(q):
            facet_nodetags = self.facets[chopped_facets[idchoppedfacet]]
            idx_chopped = np.where([node in chopped_nodes for node in facet_nodetags.tolist()])[0]
            num_chopped = idx_chopped.size
            chopped_nodetag = facet_nodetags[idx_chopped[0]]
            if num_chopped == 1:
                appended_nodes_current = []
                for l in range(n):
                    t1 = Line_Intersection(bp[l],bp[(l+1)%n],self.nodes[chopped_nodetag],self.nodes[facet_nodetags[(idx_chopped[0]+1)%3]],in_plane_check=False,bias_vector=self.bias_vector)
                    t2 = Line_Intersection(bp[l],bp[(l+1)%n],self.nodes[chopped_nodetag],self.nodes[facet_nodetags[(idx_chopped[0]+2)%3]],in_plane_check=False,bias_vector=self.bias_vector)
                    # print(t1,t2)
                    if t1 == 'parallel' or t2 == 'parallel':
                        continue
                    if t1[0]<1+1e-12 and t1[0] > -1e-12 and t1[1]<1+1e-12 and t1[1]>-1e-12:
                        is_chopped_edge = False
                        for m in range(len(chopped_edges)):
                            if chopped_nodetag in chopped_edges[m] and facet_nodetags[(idx_chopped[0]+1)%3] in chopped_edges[m]:
                                is_chopped_edge = True
                                break
                        if not is_chopped_edge:
                            nodepoint = self.nodes[chopped_nodetag]+t1[1]*(self.nodes[facet_nodetags[(idx_chopped[0]+1)%3]]-self.nodes[chopped_nodetag])
                            new_tag = int(self.num_nodes)
                            self.nodes = np.vstack((self.nodes,nodepoint))
                            appended_nodes_current.insert(0,new_tag)
                            self.num_nodes = self.num_nodes+1
                            chopped_edges.append([facet_nodetags[(idx_chopped[0]+1)%3],chopped_nodetag,new_tag])
                        else:
                            new_tag = chopped_edges[m][2]
                            appended_nodes_current.insert(0,new_tag)
                    if t2[0]<1+1e-12 and t2[0] > -1e-12 and t2[1]<1+1e-12 and t2[1]>-1e-12:
                        is_chopped_edge = False
                        for m in range(len(chopped_edges)):
                            if chopped_nodetag in chopped_edges[m] and facet_nodetags[(idx_chopped[0]+2)%3] in chopped_edges[m]:
                                is_chopped_edge = True
                                break
                        if not is_chopped_edge:
                            nodepoint = self.nodes[chopped_nodetag]+t2[1]*(self.nodes[facet_nodetags[(idx_chopped[0]+2)%3]]-self.nodes[chopped_nodetag])
                            new_tag = int(self.num_nodes)
                            self.nodes = np.vstack((self.nodes,nodepoint))
                            appended_nodes_current.append(new_tag)
                            self.num_nodes = self.num_nodes+1
                            chopped_edges.append([facet_nodetags[(idx_chopped[0]+2)%3],chopped_nodetag,new_tag])
                        else:
                            new_tag = chopped_edges[m][2]
                            appended_nodes_current.append(new_tag)
                    if len(appended_nodes_current)==2:
                        break
                # print(appended_nodes_current,facet_nodetags,idx_chopped)
                # ax = Visualize_Mesh(self,hold=True)
                # for i in range(3):
                #     ax.plot([self.nodes[facet_nodetags[i]][0],self.nodes[facet_nodetags[(i+1)%3]][0]],[self.nodes[facet_nodetags[i]][1],self.nodes[facet_nodetags[(i+1)%3]][1]],[self.nodes[facet_nodetags[i]][2],self.nodes[facet_nodetags[(i+1)%3]][2]],color='green')
                # for i in range(n):
                #     ax.plot([bp[i][0],bp[(i+1)%n][0]],[bp[i][1],bp[(i+1)%n][1]],[bp[i][2],bp[(i+1)%n][2]],color='orange')
                # plt.show()
                new_facet1 = np.array([appended_nodes_current[0],appended_nodes_current[1],facet_nodetags[(idx_chopped[0]+1)%3]])
                new_facet2 = np.array([facet_nodetags[(idx_chopped[0]+1)%3],facet_nodetags[(idx_chopped[0]+2)%3],appended_nodes_current[1]])
                self.facets = np.vstack((self.facets,new_facet1)).astype(int)
                self.facets = np.vstack((self.facets,new_facet2)).astype(int)
                self.num_facets=self.num_facets+2
            if num_chopped ==2:
                if (idx_chopped[0]+1)%3 not in idx_chopped.tolist():
                    unchopped = (idx_chopped[0]+1)%3
                else:
                    unchopped = (idx_chopped[0]-1)%3
                appended_nodes_current = []
                for l in range(n):
                    t1 = Line_Intersection(bp[l],bp[(l+1)%n],self.nodes[facet_nodetags[idx_chopped[0]]],self.nodes[facet_nodetags[unchopped]],in_plane_check=False,bias_vector=self.bias_vector)
                    t2 = Line_Intersection(bp[l],bp[(l+1)%n],self.nodes[facet_nodetags[idx_chopped[1]]],self.nodes[facet_nodetags[unchopped]],in_plane_check=False,bias_vector=self.bias_vector)
                    if t1 == 'parallel' or t2 == 'parallel':
                        continue
                    if t1[0]<1+1e-12 and t1[0] > -1e-12 and t1[1]<1+1e-12 and t1[1]>-1e-12:
                        is_chopped_edge = False
                        for m in range(len(chopped_edges)):
                            if facet_nodetags[idx_chopped[0]] in chopped_edges[m] and facet_nodetags[unchopped] in chopped_edges[m]:
                                is_chopped_edge = True
                                break
                        if not is_chopped_edge:
                            nodepoint = self.nodes[facet_nodetags[idx_chopped[0]]]+t1[1]*(self.nodes[facet_nodetags[unchopped]]-self.nodes[facet_nodetags[idx_chopped[0]]])
                            new_tag = int(self.num_nodes)
                            self.nodes = np.vstack((self.nodes,nodepoint))
                            appended_nodes_current.insert(0,new_tag)
                            self.num_nodes = self.num_nodes+1
                            chopped_edges.append([facet_nodetags[unchopped],facet_nodetags[idx_chopped[0]],new_tag])
                        else:
                            new_tag = chopped_edges[m][2]
                            appended_nodes_current.insert(0,new_tag)
                    if t2[0]<1+1e-12 and t2[0] > -1e-12 and t2[1]<1+1e-12 and t2[1]>-1e-12:
                        is_chopped_edge = False
                        for m in range(len(chopped_edges)):
                            if facet_nodetags[idx_chopped[1]] in chopped_edges[m] and facet_nodetags[unchopped] in chopped_edges[m]:
                                is_chopped_edge = True
                                break
                        if not is_chopped_edge:
                            nodepoint = self.nodes[facet_nodetags[idx_chopped[1]]]+t2[1]*(self.nodes[facet_nodetags[unchopped]]-self.nodes[facet_nodetags[idx_chopped[1]]])
                            new_tag = int(self.num_nodes)
                            self.nodes = np.vstack((self.nodes,nodepoint))
                            appended_nodes_current.append(new_tag)
                            self.num_nodes = self.num_nodes+1
                            chopped_edges.append([facet_nodetags[unchopped],facet_nodetags[idx_chopped[1]],new_tag])
                        else:
                            new_tag = chopped_edges[m][2]
                            appended_nodes_current.append(new_tag)
                # print(appended_edges,facet_nodetags[unchopped],appended_nodes_current)
                    if len(appended_nodes_current)==2:
                        break
                new_facet1 = np.array([appended_nodes_current[0],appended_nodes_current[1],facet_nodetags[unchopped]])
                self.facets = np.vstack((self.facets,new_facet1)).astype(int)
                self.num_facets=self.num_facets+1
        k = 0
        num_nodes_chopped = 0
        num_facets_chopped = 0
        while (k<self.num_nodes):
            if k+num_nodes_chopped in chopped_nodes:
                self.nodes = np.delete(self.nodes,k,axis=0)
                self.num_nodes = self.num_nodes-1
                num_nodes_chopped = num_nodes_chopped+1
                l = 0
                while (l<self.num_facets):
                    if k in self.facets[l].tolist():
                        self.facets = np.delete(self.facets,l,axis=0)
                        self.num_facets = self.num_facets-1
                        num_facets_chopped = num_facets_chopped+1
                        l=l-1
                    else:
                        for m in range(3):
                            if self.facets[l,m]>k:
                                self.facets[l,m] = self.facets[l,m]-1
                    l = l+1
                k=k-1
            k = k+1
            # print(k,self.num_nodes)
        assert num_nodes_chopped==len(chopped_nodes) and num_facets_chopped==len(chopped_facets)

class XRSA_MeshFunc():
    def __init__(self,refmesh:XRSA_Mesh):
        self.refmesh = refmesh
        self.values = np.zeros(self.refmesh.num_nodes)
        
    def __call__(self,p,padding=False,debug=False):
        target_facet_id,di = self.refmesh.Get_Target_Facet(p,padding=padding,debug=debug)
        if target_facet_id is None:
            return 0
        else:
            f1 = self.values[self.refmesh.facets[target_facet_id,0]]
            f2 = self.values[self.refmesh.facets[target_facet_id,1]]
            f3 = self.values[self.refmesh.facets[target_facet_id,2]]
            return f1*di[0]+f2*di[1]+f3*di[2]
    
    def Send_refmesh_to_gpu(self):
        handle = self.refmesh.Send_to_gpu()
        return handle

class XRSA_MeshFunc_Scalar(XRSA_MeshFunc):
    def __init__(self, refmesh: XRSA_Mesh):
        super().__init__(refmesh)
        self.Set_Values_with_Func(zero_func)
    
    def Set_Values_with_Func(self,func):
        for i in range(self.refmesh.num_nodes):
            p = self.refmesh.nodes[i]
            self.values[i]=func(p)
            
    def Total_Integration(self):
        total_intg = 0
        for i in range(self.refmesh.num_facets):
            area,_ = GetAreaAndNormal(self.refmesh.nodes[self.refmesh.facets[i,0]],self.refmesh.nodes[self.refmesh.facets[i,1]],self.refmesh.nodes[self.refmesh.facets[i,2]])
            total_intg = total_intg + (self.values[self.refmesh.facets[i,0]]+self.values[self.refmesh.facets[i,1]]+self.values[self.refmesh.facets[i,2]])*(area/3)
        return total_intg
    
    def Display_Pattern(self,ax=None,projection=False,**kwargs):
        mesh = self.refmesh
        points = np.zeros((mesh.num_nodes,3))
        if projection:
            try:
                coordinate = kwargs['coordinate']
                orientation = kwargs['orientation']
                gamma = kwargs['gamma']
            except:
                coordinate = np.average(mesh.nodes,axis=0)
                # print(mesh.nodes)
                dx,dy,dz,_,_,_ = Eigen_CovMat(mesh.nodes)
                orientation = dz
                _,gamma = GetEulerAngleFromDxDy(dx,dy)
        for i in range(mesh.num_nodes):
            if projection:
                new_coord = Projection(coordinate,orientation,gamma,mesh.nodes[i])
            else:
                new_coord = mesh.nodes[i]
            points[i,:2] = new_coord[:2]
            points[i,2] = self.values[i]
            # points[i,2] = np.sum(new_coord[:2])
        triangles = np.zeros((mesh.num_facets,3))
        for i in range(mesh.num_facets):
            for j in range(3):
                triangles[i,j]=mesh.facets[i,j]
        if (ax is None):
            fig = plt.figure()
            ax = fig.add_subplot(111, projection='3d')
        ax.plot_trisurf(points[:, 0], points[:, 1], triangles, points[:, 2], cmap='viridis')
        # ax.set_xlabel('x')
        # ax.set_ylabel('y')
        # ax.set_zlabel('z')
        ax.set_xlabel('x(m)')
        ax.set_ylabel('y(m)')
        ax.set_zlabel('Power per area(W/${\\rm m}^2$)')
        # ax.set_box_aspect([0.5,1,1])
        # print(points)
        plt.show()
        
class XRSA_MeshFunc_3dvec(XRSA_MeshFunc):
    def __init__(self, refmesh: XRSA_Mesh):
        super().__init__(refmesh)
        self.values = np.zeros((refmesh.num_nodes,3))

class XRSA_Angular_MeshFunc(XRSA_MeshFunc):
    def __init__(self, refmesh: XRSA_Mesh):
        super().__init__(refmesh)
        self.values = [XRSA_Zero_Wavelength_Func() for i in range(self.refmesh.num_nodes)]
        
    def __call__(self,p,padding=False,debug=False,parametric_moment=1,use_cuda=False):
        target_facet_id,di = self.refmesh.Get_Target_Facet(p,padding=padding,debug=False,use_cuda=use_cuda)
        if target_facet_id is None:
            return XRSA_Zero_Wavelength_Func()
        else:
            total_di = sum(di)
            di = di/total_di
            if debug:
                print(target_facet_id,di)
            new_wf = self.Triple_Wavedist_Addition(target_facet_id,di,parametric_moment=parametric_moment,debug=debug)
            # new_wf.Merge_Values()
            return new_wf
        
    def Triple_Wavedist_Addition(self,facet_id,di,parametric_moment=1,debug=False):
        wfs = [self.values[self.refmesh.facets[facet_id,x]] for x in range(3)]
        if debug:
            print(di)
            for i in range(3):
                if 'Zero' in get_type_name(wfs[i]):
                    print('wfs',wfs[i])
                else:
                    print('wfs%u'%(i))
                    wfs[i].plot_func()
                # print(wfs[i],wfs[i].domain)
        if parametric_moment == 0:
            return wfs[0]*di[0]+wfs[1]*di[1]+wfs[2]*di[2]
        elif parametric_moment == 1:
            Nonezero = np.zeros(3,dtype=bool)
            mus = np.zeros(3)
            for i in range(3):
                Nonezero[i] = 'Zero' not in get_type_name(wfs[i])
                if Nonezero[i]:
                    # if debug:
                    #     wfs[i].plot_func()
                    mus[i] = (wfs[i].domain[1]+wfs[i].domain[0])/2
                    if debug:
                        print('mus%u:%f'%(i,mus[i]))
                    if mus[i] is None:
                        mus[i] = 0
                        Nonezero[i] = False
            if not np.any(Nonezero):
                return XRSA_Zero_Linear_Func()
            valid_di = np.zeros(3)
            for i in range(3):
                if Nonezero[i]:
                    valid_di[i] = di[i]
            if np.sum(valid_di)<1e-6:
                return XRSA_Zero_Linear_Func()
            rescaled_di = valid_di / np.sum(valid_di)
            if debug:
                print(Nonezero,valid_di,rescaled_di)
            new_mu = np.sum([rescaled_di[i] * mus[i] for i in range(3)])
            new_range = np.max(np.array([(wfs[x].domain[1] - wfs[x].domain[0]) for x in range(3) if Nonezero[x]]))
            new_func = XRSA_LinearFunc(
                lambda x: (
                    (wfs[0](x - new_mu + mus[0]) * di[0] if Nonezero[0] else 0) +
                    (wfs[1](x - new_mu + mus[1]) * di[1] if Nonezero[1] else 0) +
                    (wfs[2](x - new_mu + mus[2]) * di[2] if Nonezero[2] else 0)
                ),
                new_mu - new_range / 2,
                new_mu + new_range / 2
            )
            if debug:
                print('new_mu:',new_mu)
                print(new_mu - new_range / 2,new_mu + new_range / 2)
            if debug:
                if 'Zero' in get_type_name(new_func):
                    print('new')
                    print(new_func)
                else:
                    print('new')
                    mus_new = new_func.Get_moments(order=1,debug=False)
                    print(mus_new)
                    new_func.plot_func()
            return new_func
        
    def Display_Pattern(self,ax=None,projection=False):
        mesh = self.refmesh
        points = np.zeros((mesh.num_nodes,3))
        if projection:
            coordinate = np.average(mesh.nodes,axis=0)
            # print(mesh.nodes)
            dx,dy,dz,_,_,_ = Eigen_CovMat(mesh.nodes)
            orientation = dz
            _,gamma = GetEulerAngleFromDxDy(dx,dy)
        for i in range(mesh.num_nodes):
            if projection:
                new_coord = Projection(coordinate,orientation,gamma,mesh.nodes[i])
            else:
                new_coord = mesh.nodes[i]
            points[i,:2] = new_coord[:2]
            points[i,2] = self.values[i].Total_Integration()
            # points[i,2] = np.sum(new_coord[:2])
        triangles = np.zeros((mesh.num_facets,3))
        for i in range(mesh.num_facets):
            for j in range(3):
                triangles[i,j]=mesh.facets[i,j]
        if (ax is None):
            fig = plt.figure()
            ax = fig.add_subplot(111, projection='3d')
        ax.plot_trisurf(points[:, 0], points[:, 1], triangles, points[:, 2], cmap='viridis')
        # ax.set_box_aspect([0.5,1,1])
        # print(points)
        plt.show()
        
    def Total_Integration(self):
        value_meshfunc = XRSA_MeshFunc_Scalar(self.refmesh)
        for i in range(self.refmesh.num_nodes):
            value_meshfunc.values[i] = self.values[i].Total_Integration()
        return value_meshfunc.Total_Integration()
    
    def Delete_from_gpu(self):
        self.refmesh.Delete_from_gpu()

class XRSA_JointDistribution(XRSA_MeshFunc):
    def __init__(self,spatial_mesh,base_angular_mesh,an_ori_gam=None):
        super().__init__(spatial_mesh)
        self.base_angular_mesh = base_angular_mesh
        # print(an_ori_gam)
        if an_ori_gam is None:
            self.an_ori_gam = [(np.array([0,0,1]),0) for i in range(self.refmesh.num_nodes)]
        else:
            typename = get_type_name(an_ori_gam)
            if 'tuple' in typename:
                self.an_ori_gam = [an_ori_gam for i in range(self.refmesh.num_nodes)]
            elif 'list' in typename:
                try:
                    assert len(an_ori_gam) == self.refmesh.num_nodes
                except:
                    raise Exception('Ambiguous input: list length %u'%(len(an_ori_gam)))
                self.an_ori_gam = an_ori_gam
            else:
                raise Exception('Unaccepted type: %s'%(typename))
        
        self.values = []
        for id_spatial in range(self.refmesh.num_nodes):
            # print('Generating joint_dist:%u/%u'%(id_spatial,self.refmesh.num_nodes))
            angular_mesh = copy.deepcopy(self.base_angular_mesh)
            postrans_encf = Position_Transform(np.array([0.,0.,0.]),self.an_ori_gam[id_spatial][0],self.an_ori_gam[id_spatial][1])
            mesh_encoder = Mesh_Encoder([postrans_encf])
            mesh_encoder.Encode(angular_mesh)
            angular_meshfunc = XRSA_Angular_MeshFunc(angular_mesh)
            self.values.append(angular_meshfunc)
    
    def __call__(self,p,padding=False,debug=False):
        target_facet_id,di = self.refmesh.Get_Target_Facet(p,padding=padding,debug=False)
        if target_facet_id is None:
            return XRSA_Zero_Angular_MeshFunc()
        else:
            total_di = sum(di)
            di = di/total_di
            return self.Triple_AngularMF_Addition(target_facet_id,di,debug=debug)
    
    def Triple_AngularMF_Addition(self,facet_id,di,debug=False):
        node_ids = self.refmesh.facets[facet_id]
        mfs = [self.values[node_ids[x]] for x in range(3)]
        # zero_args = [('Zero' in get_type_name(mfs[x])) for x in range(3)]
        ori_facet = [self.an_ori_gam[node_ids[i]][0] for i in range(3)]
        gam_facet = [self.an_ori_gam[node_ids[i]][1] for i in range(3)]
        #gamma check
        if max(gam_facet)-min(gam_facet)>np.pi:
            for i in range(3):
                if gam_facet[i]>(max(gam_facet)+min(gam_facet))/2:
                    gam_facet[i] = gam_facet[i]-2*np.pi
                    
        new_ori = ori_facet[0]*di[0]+ori_facet[1]*di[1]+ori_facet[2]*di[2]
        new_gam = gam_facet[0]*di[0]+gam_facet[1]*di[1]+gam_facet[2]*di[2]
        if debug:
            print('di array:',di)
            for i in range(3):
                print('meshfunc%u'%(i))
                print(ori_facet[i])
                print(gam_facet[i])
                print(mfs[i].Total_Integration())
                mfs[i].Display_Pattern()
        # print(new_ori,new_gam)
        # Visualize_Mesh(mfs[0].refmesh)
        # Visualize_Mesh(mfs[1].refmesh)
        # Visualize_Mesh(mfs[2].refmesh)
        
        new_angularmesh = copy.deepcopy(self.base_angular_mesh)
        postrans_encf = Position_Transform(np.array([0.,0.,0.]),new_ori,new_gam)
        mesh_encoder = Mesh_Encoder([postrans_encf])
        mesh_encoder.Encode(new_angularmesh)
        new_angularmf = XRSA_Angular_MeshFunc(new_angularmesh)
        
        for i in range(new_angularmesh.num_nodes):
            new_angularmf.values[i] = mfs[0].values[i]*di[0]+mfs[1].values[i]*di[1]+mfs[2].values[i]*di[2]
        if debug:
            print('new meshfunc')
            print(new_ori)
            print(new_gam)
            print(new_angularmf.Total_Integration())
            new_angularmf.Display_Pattern()
        # integ = new_angularmf.Total_Integration()
        # print(integ)
        return new_angularmf
    
    def Visualize_Spatial_Distribution(self,ax=None,projection=False,display_anmfs = False,**kwargs):
        spatial_mesh = self.refmesh
        visualize_meshfunc = XRSA_MeshFunc_Scalar(spatial_mesh)
        if projection:
            try:
                coordinate = kwargs['coordinate']
                orientation = kwargs['orientation']
                gamma = kwargs['gamma']
            except:
                raise Exception('Please input the coordinate, the orientation, and the gamma for the Projection.')
        for id_spatial in range(spatial_mesh.num_nodes):
            print("Integrating: %u/%u"%(id_spatial,spatial_mesh.num_nodes))
            # for id_angular in range(self.values[id_spatial].refmesh.num_nodes):
            #     print(self.values[id_spatial].values[id_angular])
            # if id_spatial==5:
            #     for k in range(self.values[id_spatial].refmesh.num_nodes):
            #         self.values[id_spatial].values[k].plot_func()
            intg_angular = self.values[id_spatial].Total_Integration()
            if display_anmfs:
                self.values[id_spatial].Display_Pattern(projection=True)
            visualize_meshfunc.values[id_spatial] = intg_angular
        if projection:
            visualize_meshfunc.Display_Pattern(ax,projection=projection,coordinate=coordinate,orientation=orientation,gamma=gamma)
        else:
            visualize_meshfunc.Display_Pattern(ax,projection=projection)
        
    def Send_refmesh_to_gpu(self):
        sp_handle = super().Send_refmesh_to_gpu()
        an_handles = []
        for i in range(self.refmesh.num_nodes):
            an_handle = self.values[i].Send_refmesh_to_gpu()
            an_handles.append(an_handle)
        return (sp_handle,an_handles)
    
    def Save_to_File(self,filename='default.pkl'):
        sp_mesh = copy.deepcopy(self.refmesh)
        sp_mesh.norm = None
        an_basemesh = copy.deepcopy(self.base_angular_mesh)
        an_basemesh.norm = None
        jdsave = XRSA_JDSave(sp_mesh,an_basemesh,self.an_ori_gam)
        spnorm_mf = XRSA_MeshFunc_3dvec(sp_mesh)
        for i in range(sp_mesh.num_nodes):
            values_sp = []
            norm = self.refmesh.norm(self.refmesh.nodes[i])
            spnorm_mf.values[i] = norm
            for j in range(an_basemesh.num_nodes):
                wfunc = self.values[i].values[j]
                if 'Zero' in get_type_name(wfunc):
                    values_sp.append([np.array([]),np.array([])])
                    continue
                # print(wfunc.domain[1],wfunc.domain[0])
                epsilon = (wfunc.domain[1]-wfunc.domain[0])*1e-6
                x = np.linspace(wfunc.domain[0]+epsilon,wfunc.domain[1]-epsilon,101)
                vals = np.array([wfunc(x[k]) for k in range(101)])
                values_sp.append([x,vals])
            jdsave.values.append(values_sp)
        jdsave.sp_mesh_norm = spnorm_mf
        jdsave.Save(filename)
    
def Load_JD_From_File(filename):
    with open(filename,'rb') as file:
        jdsave = pickle.load(file)
    obj = XRSA_JointDistribution(jdsave.refmesh,jdsave.base_angular_mesh,jdsave.an_ori_gam)
    obj.refmesh.norm = jdsave.sp_mesh_norm
    for i in range(obj.refmesh.num_nodes):
        an_mf = obj.values[i]
        for j in range(an_mf.refmesh.num_nodes):
            wfunc_vals = jdsave.values[i][j]
            # print(wfunc_vals)
            if wfunc_vals[0].shape[0] < 1:
                obj.values[i].values[j] = XRSA_Zero_Wavelength_Func()
                continue
            F = interp1d(wfunc_vals[0],wfunc_vals[1],fill_value=0, bounds_error=False)
            wfunc = XRSA_LinearFunc(F,wfunc_vals[0][0]+1e-10,wfunc_vals[0][-1]-1e-10)
            obj.values[i].values[j] = wfunc
    print(f"Object loaded from {filename}")
    return obj
        
class XRSA_JDSave(XRSA_JointDistribution):
    def __init__(self, spatial_mesh, base_angular_mesh, an_ori_gam=None):
        super().__init__(spatial_mesh, base_angular_mesh, an_ori_gam)
        self.values = []
        self.sp_mesh_norm = None
    
    def Save(self,filename='default.pkl'):
        with open(filename,'wb') as file:
            pickle.dump(self,file)
            print(f"Object saved to {filename}")


class XRSA_Zero_Angular_MeshFunc(XRSA_MeshFunc):
    def __init__(self):
        self.refmesh = None
        self.values = None
    
    def __call__(self,_):
        return XRSA_Zero_Wavelength_Func()

class Mesh_Encoder():
    def __init__(self,encoding_list=[]) -> None:
        self.encoding_list = encoding_list
    
    def Encode(self,mesh,debug=False):
        for id_node in range(mesh.num_nodes):
            for f in self.encoding_list:
                mesh.nodes[id_node] = f.forward(mesh.nodes[id_node])
    
    def Decode(self,mesh):
        for id_node in range(mesh.num_nodes):
            for f in self.encoding_list[::-1]:
                mesh.nodes[id_node] = f.backward(mesh.nodes[id_node])

class Encoding_Func():
    def __init__(self) -> None:
        self.trans_matrix = np.eye(3)
        self.disp_vector = np.zeros(3)
    
    def forward(self,p):
        p = np.dot(self.trans_matrix,p)
        p = p+self.disp_vector
        return p
    
    def backward(self,p):
        p = p-self.disp_vector
        p = np.linalg.solve(self.trans_matrix,p)
        return p

class Position_Transform(Encoding_Func):
    def __init__(self,coordinate,orientation,gamma) -> None:
        super().__init__()
        self.trans_matrix = GetRotationMatrixFromEulerAngle(orientation,gamma)
        self.disp_vector = coordinate

class Zoom_in(Encoding_Func):
    def __init__(self,base_point,alpha) -> None:
        super().__init__()
        self.base_point = base_point
        self.alpha=alpha
        
    def forward(self,p):
        return self.base_point+self.alpha*(p-self.base_point)
    
    def backward(self, p):
        return self.base_point+1/self.alpha*(p-self.base_point)

class Projection_to_new_Coordinate(Encoding_Func):
    def __init__(self,plane_center,dx,dy) -> None:
        super().__init__()
        self.center = plane_center
        self.dx = dx/np.linalg.norm(dx)
        self.dy = dy/np.linalg.norm(dy)
        self.dz = np.cross(self.dx,self.dy)
    
    def forward(self,p):
        rel = p-self.center
        projected = np.array([np.dot(rel,self.dx),np.dot(rel,self.dy),np.dot(rel,self.dz)])
        return projected

    def backward(self, p):
        rel = p[0]*self.dx+p[1]*self.dy+p[2]*self.dz
        return rel+self.center

class Reflect_Against_Norm(Encoding_Func):
    def __init__(self,norm) -> None:
        self.norm = norm
        
    def forward(self,p):
        return get_mirror_vector(self.norm,p)
    
    def backward(self, p):
        return get_mirror_vector(self.norm,p)

class Project_Along_a_Func(Encoding_Func):
    def __init__(self,x_dir,y_dir,func) -> None:
        self.func = func
        self.dx = x_dir/np.linalg.norm(x_dir)
        self.dy = y_dir/np.linalg.norm(y_dir)
        self.dz = np.cross(self.dx,self.dy)
        
    def forward(self,p):
        px = np.dot(p,self.dx)
        py = np.dot(p,self.dy)
        pz = np.dot(p,self.dz)
        py = py-self.func(px)
        return np.array(px*self.dx+py*self.dy+pz*self.dz)
    
    def backward(self, p):
        px = np.dot(p,self.dx)
        py = np.dot(p,self.dy)
        pz = np.dot(p,self.dz)
        py = py+self.func(px)
        return np.array(px*self.dx+py*self.dy+pz*self.dz)

class Extend_along_a_Direction(Encoding_Func):
    def __init__(self,plane_center,direction,alpha) -> None:
        super().__init__()
        self.plane_center = plane_center
        self.direction = direction
        self.alpha = alpha
    
    def forward(self,p):
        p=p+(self.alpha-1)*np.dot((p-self.plane_center),self.direction)/np.dot(self.direction,self.direction)*self.direction
        return p
    
    def backward(self, p):
        p = p-(self.alpha-1)/self.alpha*np.dot((p-self.plane_center),self.direction)/np.dot(self.direction,self.direction)*self.direction
        return p

class Project_from_R_Theta_Phi(Encoding_Func):
    def __init__(self) -> None:
        pass
    
    def forward(self, p):
        return np.array([p[0]*np.sin(p[1])*np.cos(p[2]), p[0]*np.sin(p[1])*np.sin(p[2]), p[0]*np.cos(p[1])])
    
    def backward(self, p):
        return np.array([np.linalg.norm(p), np.arccos(np.dot(p/np.linalg.norm(p),np.array([0.,0.,1.]))), np.arccos(np.dot(p-np.array([0.,0.,1])*np.dot(p/np.linalg.norm(p),np.array([0.,0.,1.])),np.array([1.,0.,0.])))])

class Project_To_Plane(Encoding_Func):
    def __init__(self,plane_center,base_point) -> None:
        super().__init__()
        self.plane_center = plane_center
        self.base_point = base_point
        
    def forward(self, p):
        normal = self.plane_center-self.base_point
        v1 = p-self.base_point
        t = np.dot(normal,normal)/np.dot(normal,v1)
        projected=self.base_point+t*v1
        return projected
    
    def backward(self, p):
        p = self.base_point+(p-self.base_point)/np.linalg.norm(p-self.base_point)*np.linalg.norm(self.plane_center-self.base_point)
        return p

class Project_To_UnitSphere(Project_To_Plane):
    def __init__(self, plane_center, base_point) -> None:
        super().__init__(plane_center, base_point)
        
    def forward(self, p):
        return self.base_point+(p-self.base_point)/np.linalg.norm(p-self.base_point)
    
    def backward(self, p):
        return super().forward(p)

def Generate_Rectangular_Mesh(coordinate,orientation,gamma,shape,lc):
    print('Generating rectangular mesh.')
    coordinate = np.array(coordinate)
    orientation = np.array(orientation)
    gmsh.initialize()
    gmsh.model.add('default')
    dx,dy = GetDxDyFromEulerAngle(orientation,gamma)
    # print(orientation)
    point1 = coordinate-dx*shape[0]/2-dy*shape[1]/2
    point2 = coordinate+dx*shape[0]/2-dy*shape[1]/2
    point3 = coordinate+dx*shape[0]/2+dy*shape[1]/2
    point4 = coordinate-dx*shape[0]/2+dy*shape[1]/2
    gmsh.model.geo.addPoint(point1[0],point1[1],point1[2],lc,1)
    gmsh.model.geo.addPoint(point2[0],point2[1],point2[2],lc,2)
    gmsh.model.geo.addPoint(point3[0],point3[1],point3[2],lc,3)
    gmsh.model.geo.addPoint(point4[0],point4[1],point4[2],lc,4)
    gmsh.model.geo.addLine(1, 2, 1)
    gmsh.model.geo.addLine(2, 3, 2)
    gmsh.model.geo.addLine(3, 4, 3)
    gmsh.model.geo.addLine(4, 1, 4)
    gmsh.model.geo.addCurveLoop([1,2,3,4],1)
    gmsh.model.geo.addPlaneSurface([1], 1)
    gmsh.model.geo.synchronize()
    gmsh.model.mesh.generate(2)
    nodes_got = gmsh.model.mesh.getNodes()
    elements = gmsh.model.mesh.getElements(dim=2)
    facets_got = elements[2]
    mesh = Load_mesh(nodes_got,facets_got)
    gmsh.finalize()
    mesh.norm = Generate_normal_function_plane_surface(orientation)
    if np.abs(np.dot(coordinate,orientation))<1e-6:
        mesh.bias_vector=coordinate+orientation
    return mesh

def Generate_Circular_Mesh(coordinate,orientation,radius,lc):
    print('Generating circular mesh.')
    coordinate = np.array(coordinate)
    orientation = np.array(orientation)
    gmsh.initialize()
    gmsh.model.add('default')
    gmsh.model.occ.addCircle(coordinate[0],coordinate[1],coordinate[2],radius,zAxis=orientation)
    gmsh.model.occ.addCurveLoop([1],1)
    gmsh.model.occ.addPlaneSurface([1],1)
    gmsh.model.occ.synchronize()
    gmsh.model.mesh.setSize(gmsh.model.getEntities(0),lc)
    gmsh.model.mesh.generate(2)
    nodes_got = gmsh.model.mesh.getNodes()
    elements = gmsh.model.mesh.getElements(dim=2)
    facets_got = elements[2]
    mesh = Load_mesh(nodes_got,facets_got)
    gmsh.finalize()
    mesh.norm = Generate_normal_function_plane_surface(orientation)
    if np.abs(np.dot(coordinate,orientation))<1e-6:
        mesh.bias_vector=coordinate+orientation
    return mesh

def Generate_Dome_Mesh(coordinate,orientation,radius,lc,min_theta=0):
    print('Generating dome mesh.')
    coordinate = np.array(coordinate)
    orientation = np.array(orientation)
    gmsh.initialize()
    gmsh.model.add('default')
    gmsh.model.occ.addSphere(coordinate[0],coordinate[1],coordinate[2],radius,angle1=min_theta)
    gmsh.model.occ.synchronize()
    gmsh.model.mesh.setSize(gmsh.model.getEntities(0),lc)
    gmsh.model.mesh.generate(2)
    
    #Remove unused nodes
    nodes = gmsh.model.mesh.getNodes()
    num_nodes = len(nodes[0])
    nodes = dict(zip(nodes[0],np.array(nodes[1]).reshape((-1,3))))
    nodetags = list(nodes.keys())
    nodes_from_tags = dict(zip(nodes.keys(),np.arange(num_nodes)))
    # print(elements)
    elements = gmsh.model.mesh.getElements(dim=2)
    # print(elements)
    facets = np.array(elements[2],dtype=int).reshape((-1,3))
    num_facets = np.size(facets,axis = 0)
    
    # starttime = time.time()
    i=0
    while(i<num_nodes):
        cor = nodes[nodetags[i]]
        if (np.dot(cor,cor)<0.999):
            j=0
            while(j<num_facets):
                if nodetags[i] in facets[j]:
                    facets = np.delete(facets,j,axis=0)
                    num_facets = num_facets-1
                    continue
                j=j+1
            nodes.pop(nodetags[i])
            num_nodes = num_nodes-1
            nodetags = list(nodes.keys())
            nodes_from_tags = dict(zip(nodes.keys(),np.arange(num_nodes)))
            continue
        i = i+1
    
    nodes_new = []
    facets_new = []
    for i in range(num_nodes):
        nodes_new.append(nodes[nodetags[i]])
    for i in range(num_facets):
        facets_new.append(np.array([nodes_from_tags[facets[i,x]] for x in range(3)]))
    nodes = np.array(nodes_new)
    facets = np.array(facets_new)
    mesh = XRSA_Mesh(nodes,facets)
    
    # endtime = time.time()
    # print(endtime-starttime)
    
    encf_to_origin = Position_Transform(-coordinate,np.array([0,0,1]),0)
    encf_trans = Position_Transform([0,0,0],orientation,0)
    encf_coord_back = Position_Transform(coordinate,np.array([0,0,1]),0)
    encoder = Mesh_Encoder([encf_to_origin,encf_trans,encf_coord_back])
    encoder.Encode(mesh)
    mesh.norm = Generate_normal_function_spherical_surface(coordinate)
    gmsh.finalize()
    return mesh

def Generate_Fitted_Mesh(nodes,dy_factor=6,lc_min=1e-6,redundancy=0.5):
    plane_center = np.average(nodes,axis=0)
    dx,dy,dz,lx,ly,alpha = Eigen_CovMat(nodes)
    fake_mesh = XRSA_Mesh(nodes=nodes,facets=np.array([]))
    encf_projection = Projection_to_new_Coordinate(plane_center,dx,dy)
    encoder_projection = Mesh_Encoder([encf_projection])
    encoder_projection.Encode(fake_mesh)
    projected_nodes = fake_mesh.nodes
    popt,_ = curve_fit(Symmetric_Quartic_Polynomial,projected_nodes[:,0],projected_nodes[:,1],
                    p0=[0,0,0,0],
                    bounds=([-10,-0.1,-10,-1.],[10,0.1,10,1.]))
    quartic_func = lambda x: Symmetric_Quartic_Polynomial(x,*popt)
    max_dx = np.maximum(np.max(fake_mesh.nodes[:,0])-np.min(fake_mesh.nodes[:,0]),lc_min)
    max_dy = np.maximum(np.max(fake_mesh.nodes[:,1])-np.min(fake_mesh.nodes[:,1]),lc_min)
    alpha = max_dx/max_dy
    # encf_projection = Projection_to_new_Coordinate(plane_center,dx,dy)
    encf_fitting_proj = Project_Along_a_Func(np.array([1.,0.,0.]),np.array([0.,1.,0.]),quartic_func)
    encf_compress = Extend_along_a_Direction(plane_center=np.zeros(3),direction=np.array([1.,0.,0.]),alpha=1/alpha) 
    encoder = Mesh_Encoder([encf_fitting_proj,encf_compress])
    encoder.Encode(fake_mesh)
    max_dx = np.maximum(np.max(fake_mesh.nodes[:,0])-np.min(fake_mesh.nodes[:,0]),lc_min)
    max_dy = np.maximum(np.max(fake_mesh.nodes[:,1])-np.min(fake_mesh.nodes[:,1]),lc_min)
    new_center = np.array([(np.max(fake_mesh.nodes[:,0])+np.min(fake_mesh.nodes[:,0]))/2,
                        (np.max(fake_mesh.nodes[:,1])+np.min(fake_mesh.nodes[:,1]))/2,
                        (np.max(fake_mesh.nodes[:,2])+np.min(fake_mesh.nodes[:,2]))/2])
    new_mesh = Generate_Rectangular_Mesh(new_center,np.array([0.,0.,1.]),0,[(1+redundancy)*max_dx,(1+redundancy)*max_dy],max_dy/dy_factor)
    encoder.Decode(new_mesh)
    encoder_projection.Decode(new_mesh)
    return new_mesh

def Load_Mesh_From_File(filename,coordinate,orientation,gamma):
    gmsh.initialize()
    gmsh.open(filename)
    nodes_got = gmsh.model.mesh.getNodes()
    elements = gmsh.model.mesh.getElements(dim=2)
    facets_got = elements[2]
    mesh = Load_mesh(nodes_got,facets_got)
    gmsh.finalize()
    encf_postrans = Position_Transform(coordinate,orientation,gamma)
    encoder = Mesh_Encoder([encf_postrans])
    encoder.Encode(mesh)
    mesh.norm = Generate_normal_function_plane_surface(orientation)
    if np.abs(np.dot(coordinate,orientation))<1e-6:
        mesh.bias_vector=coordinate+orientation
    return mesh

def Load_mesh(nodes_got,facets_got):
    num_nodes = len(nodes_got[0])
    nodes_raw = dict(zip(nodes_got[0],np.array(nodes_got[1]).reshape((-1,3))))
    nodetags = list(nodes_raw.keys())
    nodes_from_tags = dict(zip(nodes_raw.keys(),np.arange(num_nodes)))
    elements = gmsh.model.mesh.getElements(dim=2)
    facets_raw = np.array(facets_got,dtype=int).reshape((-1,3))
    num_facets = np.size(facets_raw,axis = 0)
    nodes = []
    facets = []
    for i in range(num_nodes):
        nodes.append(nodes_raw[nodetags[i]])
    for i in range(num_facets):
        facets.append(np.array([nodes_from_tags[facets_raw[i,x]] for x in range(3)]))
    nodes = np.array(nodes)
    facets = np.array(facets)
    mesh = XRSA_Mesh(nodes,facets)
    return mesh

def Synchronize_Coordinate(mesh, base_z = [0,0,1], base_x = [1,0,0], base_coord = None):
    nodes_array = np.array([mesh.nodes[x] for x in range(mesh.num_nodes)])
    mesh_center = np.average(np.array([mesh.nodes[i] for i in range(mesh.num_nodes)]),axis=0)
    dx,dy,dz,_,_,_ = Eigen_CovMat(nodes_array)
    if np.dot(dz,base_z)<0:
        dz=-dz
        dy=-dy
    if np.dot(dx,base_x)<0:
        dx=-dx
        dy=-dy
    ori,gam = GetEulerAngleFromDxDy(dx,dy)
    if base_coord is None:
        coordinate = mesh_center
    else:
        coordinate = base_coord
    orientation = ori
    gamma = gam
    return coordinate,orientation,gamma

def Visualize_Mesh(*args,**kwargs):
    visualize_colors = ['red','green','orange','green','orange']
    projection = kwargs.get('projection',False)
    if projection:
        try:
            coordinate = kwargs['coordinate']
            orientation = kwargs['orientation']
            gamma = kwargs['gamma']
        except:
            raise Exception('Please input the coordinate, the orientation, and the gamma for the Projection.')
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    for k in range(len(args)):
        Mesh = args[k]
        for j in range(Mesh.num_facets):
            nodes = np.array([Mesh.nodes[Mesh.facets[j,x]] for x in range(3)])
            if projection:
                for i in range(3):
                    nodes[i] = Projection(coordinate,orientation,gamma,nodes[i])
            nodes = nodes.transpose()
            for i in range(3):
                ax.plot([nodes[0,i], nodes[0,(i+1)%3]], [nodes[1,i], nodes[1,(i+1)%3]], [nodes[2,i], nodes[2,(i+1)%3]], color=visualize_colors[k])
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_zlabel('z')
    if 'hold' in kwargs:
        return ax
    else:
        plt.show()

if __name__ == '__main__':
    mesh = Generate_Rectangular_Mesh(np.array([0,0,0]),np.array([1,0,0]),0,np.array([1.0,1.0]),0.03)
    # mesh = Generate_Circular_Mesh([0,0,0],[0,0,1],1e-6,1e-6)
    # mesh = Generate_Dome_Mesh([0,0,0],[0,0.707,0.707],1,0.03)
    mesh.bias_vector = np.array([0,0,1])
    print(mesh.num_facets)
    starttime = time.time()
    mesh.Get_Facet_Det_cuda()
    endtime = time.time()
    print(endtime-starttime)
    
    starttime = time.time()
    mesh.Get_Facet_Det()
    endtime = time.time()
    print(endtime-starttime)
    
    starttime = time.time()
    mesh.Get_Facet_Det_cuda()
    endtime = time.time()
    print(endtime-starttime)