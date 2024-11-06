from XRSA_Component import *
from XRSA_Transmission import *
from XRSA_LinearFunc import *
from XRSA_MeshFunc import *

omega = 53.5
curvature_radius = 1.0
fm = curvature_radius*np.sin(np.radians(omega))
source_coordinate = np.array([-2.*fm*np.cos(np.radians(omega)),0,2.*fm*np.sin(np.radians(omega))])
source_orientation = np.array([np.cos(np.radians(omega)),0,-np.sin(np.radians(omega))])
wavelength_func = XRSA_Wavelength_Func(lambda x: Gaussian(x, 1., 3.95, 0.0001),3.949,3.951)
spatial_mesh = Generate_Rectangular_Mesh(source_coordinate,source_orientation,0,np.array([0.1,0.1]),0.02)
angular_mesh = Generate_Dome_Mesh(np.zeros(3),source_orientation,1.,0.1,np.radians(85))
joint_dist = XRSA_JointDistribution(spatial_mesh,angular_mesh)
for i in range(spatial_mesh.num_nodes):
    angular_mf = joint_dist.values[i]
    for j in range(angular_mesh.num_nodes):
        node_an = angular_mf.refmesh.nodes[j]
        angular_mf.values[j] = wavelength_func*np.dot(node_an,source_orientation)
source = XRSA_Source(spatial_mesh,joint_dist,'Source')

spherical_crystal_coordinate = np.zeros(3)
spherical_crystal_orientation = np.array([0.,0.,1.])
spherical_crystal_gamma = 0
spherical_crystal_shape = np.array([0.1,0.1])
spacing = 2.457
lc_sph = 0.01
rc_width_sph = 1e-4
rc_func_sph = XRSA_LinearFunc(lambda x: Gaussian(x,1.0*np.sqrt(2*np.pi)*(rc_width_sph/2.355),0.,rc_width_sph/2.355),-2*rc_width_sph,2*rc_width_sph)
spherical_crystal = Generate_Spherical_Crystal(spherical_crystal_coordinate,spherical_crystal_orientation,spherical_crystal_gamma,spherical_crystal_shape,lc_sph,spacing,curvature_radius,rc_func_sph,'AlphaSiO2')

detector_orientation = np.array([fm*np.cos(np.radians(omega)),0,fm*np.sin(np.radians(omega))])
detector_coordinate = np.array([-np.cos(np.radians(omega)),0,-np.sin(np.radians(omega))])
detector_shape = np.array([0.1,0.1])
# source.response['Source'].Visualize_Spatial_Distribution()
detector_mesh = Generate_Rectangular_Mesh(detector_coordinate,detector_orientation,0,detector_shape,0.01)
detector = XRSA_Detector(detector_mesh,name='CMOS')

spherical_crystal.Interact(source,'Source')
spherical_crystal.incident['Source'].Save_to_File('Johann_sph_in.pkl')
JDsave = Load_JD_From_File('Johann_sph_in.pkl')
# JDsave.Visualize_Spatial_Distribution(display_anmfs=True)
spherical_crystal.incident['Source'] = JDsave
spherical_crystal.Response_self_adaptive('Source',debug=True)
spherical_crystal.response['Source_AlphaSiO2'].Save_to_File('Johann_sph_re.pkl')
JDsave = Load_JD_From_File('Johann_sph_re.pkl')
# JDsave.Visualize_Spatial_Distribution(display_anmfs=True)
spherical_crystal.incident['Source_HOPG'] = JDsave
detector.Interact(spherical_crystal,'Source_AlphaSiO2')
detector.incident['Source_AlphaSiO2'].Save_to_File('Johann_det_in.pkl')
detector.Display_Pattern('Source_AlphaSiO2')