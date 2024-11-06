from XRSA_Component import *
from XRSA_Transmission import *
from XRSA_LinearFunc import *
from XRSA_MeshFunc import *

omega = 36.063
source_coordinate = np.array([-1.0*np.cos(np.radians(omega)),0,1.0*np.sin(np.radians(omega))])
source_orientation = np.array([np.cos(np.radians(omega)),0,-np.sin(np.radians(omega))])
wavelength_func = XRSA_Wavelength_Func(lambda x: Gaussian(x, 1., 3.95, 0.002),3.945,3.955)
source = Generate_Pinhole_Isotropic_Source(source_coordinate,source_orientation,wavelength_func,name='Source')
source.response['Source'].Visualize_Spatial_Distribution()

mosaic_crystal_coordinate = np.zeros(3)
mosaic_crystal_orientation = np.array([0.,0.,1.])
mosaic_crystal_shape = np.array([0.1,0.1])
mosaic_crystal_mesh = Generate_Rectangular_Mesh(mosaic_crystal_coordinate,mosaic_crystal_orientation,0,mosaic_crystal_shape,0.01)
rc_width = 1e-4
rc_func = XRSA_LinearFunc(lambda x: Gaussian(x,1.0*np.sqrt(2*np.pi)*(rc_width/2.355),0.,rc_width/2.355),-2*rc_width,2*rc_width)
mosaic_crystal = XRSA_Mosaic_Crystal(mosaic_crystal_mesh,3.355,rc_func,name='HOPG')

beta = 53.5
spherical_crystal_coordinate = np.array([3.17*np.cos(np.radians(omega)),0,3.17*np.sin(np.radians(omega))])
spherical_crystal_orientation = np.array([-np.cos(np.radians(90-beta+omega)),0,-np.sin(np.radians(90-beta+omega))])
spherical_crystal_shape = np.array([0.1,0.1])
curvature_radius = 2.7
spacing = 2.457
lc_sph = 0.02
rc_width_sph = 1e-4
rc_func_sph = XRSA_LinearFunc(lambda x: Gaussian(x,1.0*np.sqrt(2*np.pi)*(rc_width_sph/2.355),0.,rc_width_sph/2.355),-2*rc_width_sph,2*rc_width_sph)
spherical_crystal = Generate_Spherical_Crystal(spherical_crystal_coordinate,spherical_crystal_orientation,0,spherical_crystal_shape,lc_sph,spacing,curvature_radius,rc_func_sph,'AlphaSiO2')

detector_orientation = -get_mirror_vector(spherical_crystal_orientation,(mosaic_crystal_coordinate-spherical_crystal_coordinate)/np.linalg.norm(mosaic_crystal_coordinate-spherical_crystal_coordinate))
detector_coordinate = spherical_crystal_coordinate-detector_orientation*curvature_radius*np.sin(np.radians(beta))
detector_shape = np.array([0.1,0.1])
detector_mesh = Generate_Rectangular_Mesh(detector_coordinate,detector_orientation,0,detector_shape,0.02)
detector = XRSA_Detector(detector_mesh,name='CMOS')

mosaic_crystal.Interact(source,'Source')
mosaic_crystal.incident['Source'].Save_to_File('Mosaic_Incident.pkl')
JDsave = Load_JD_From_File('Mosaic_Incident.pkl')
# JDsave.Visualize_Spatial_Distribution(display_anmfs=False)
mosaic_crystal.incident['Source'] = JDsave
mosaic_crystal.Response('Source')
mosaic_crystal.response['Source_HOPG'].Save_to_File('Mosaic_Diffracted.pkl')
JDsave = Load_JD_From_File('Mosaic_Diffracted.pkl')
# JDsave.Visualize_Spatial_Distribution(display_anmfs=True)
mosaic_crystal.response['Source_HOPG'] = JDsave
spherical_crystal.Interact(mosaic_crystal,'Source_HOPG',mode='PB',load_saved_state=False,debug=False)
spherical_crystal.incident['Source_HOPG'].Save_to_File('Spherical_Incident_adaptive.pkl')
JDsave = Load_JD_From_File('Spherical_Incident_adaptive.pkl')
# JDsave.Visualize_Spatial_Distribution(display_anmfs=True)
spherical_crystal.incident['Source_HOPG'] = JDsave
spherical_crystal.Response_self_adaptive('Source_HOPG',debug=True)
spherical_crystal.response['Source_HOPG_AlphaSiO2'].Save_to_File('Spherical_Response.pkl')
JDsave = Load_JD_From_File('Spherical_Response.pkl')
# JDsave.Visualize_Spatial_Distribution(display_anmfs=True)
spherical_crystal.response['Source_HOPG_AlphaSiO2'] = JDsave
detector.Interact(spherical_crystal,'Source_HOPG_AlphaSiO2',mode='PS',dy_factor=10,debug=True)
detector.incident['Source_HOPG_AlphaSiO2'].Save_to_File('Detector_Incident.pkl')
# ax=Visualize_Mesh(detector.spatial_mesh,hold=True,projection=True,coordinate = detector_coordinate,orientation=detector_orientation,gamma=0)
JDsave = Load_JD_From_File('Detector_Incident.pkl')
detector.Display_Pattern('Source_HOPG_AlphaSiO2')