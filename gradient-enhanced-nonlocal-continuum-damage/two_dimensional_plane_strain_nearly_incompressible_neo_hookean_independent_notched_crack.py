# import necessary libraries
from __future__ import division
from dolfin import *
import os
import pathlib
import pickle
import subprocess
from dolfin_utils.meshconvert import meshconvert
import numpy as np
import matplotlib.pyplot as plt
from copy import deepcopy
import textwrap

# Set utility functions

def generate_savedir(namedir):
    savedir = "./"+namedir+"/"
    create_savedir(savedir)

    return savedir

def create_savedir(savedir):
    if MPI.rank(MPI.comm_world) == 0:
        if os.path.isdir(savedir) == False:
            pathlib.Path(savedir).mkdir(parents=True, exist_ok=True)
    MPI.barrier(MPI.comm_world)

def fenics_mesher(fenics_mesh, subdir, mesh_name):

    def generate_mesh_dir(subdir):
        mesh_dir = "./"+subdir+"/"+"meshes"+"/"
        create_savedir(mesh_dir)

        return mesh_dir

    mesh_dir = generate_mesh_dir(subdir)

    geo_mesh = XDMFFile(MPI.comm_world, mesh_dir+mesh_name+".xdmf")
    geo_mesh.write(fenics_mesh)
    fenics_mesh.init()

    return fenics_mesh

def mesh_topologier(fenics_mesh_topology, subdir, mesh_topology_name):

    def generate_mesh_topology_dir(subdir):
        mesh_topology_dir = "./"+subdir+"/"+"mesh_topologies"+"/"
        create_savedir(mesh_topology_dir)

        return mesh_topology_dir

    mesh_topology_dir = generate_mesh_topology_dir(subdir)

    mesh_topology = XDMFFile(MPI.comm_world, mesh_topology_dir+mesh_topology_name+".xdmf")
    mesh_topology.write(fenics_mesh_topology)

def gmsh_mesher(gmsh_file, subdir, mesh_name):

    def generate_mesh_dir(subdir):
        mesh_dir = "./"+subdir+"/"+"meshes"+"/"
        create_savedir(mesh_dir)

        return mesh_dir

    mesh_dir = generate_mesh_dir(subdir)
    temp_mesh = Mesh() # create an empty mesh object

    if not os.path.isfile(mesh_dir+mesh_name+".xdmf"):

        if MPI.rank(MPI.comm_world) == 0:

            # Create a .geo file defining the mesh
            geo_file = open(mesh_dir+mesh_name+".geo", "w")
            geo_file.writelines(gmsh_file)
            geo_file.close()

            # Call gmsh to generate the mesh file and call dolfin-convert to generate the .xml file
            try:
                subprocess.call(["gmsh", "-2", "-o", mesh_dir+mesh_name+".msh", mesh_dir+mesh_name+".geo"])
            except OSError:
                print("-----------------------------------------------------------------------------")
                print(" Error: unable to generate the mesh using gmsh")
                print(" Make sure that you have gmsh installed and have added it to your system PATH")
                print("-----------------------------------------------------------------------------")
                return
            meshconvert.convert2xml(mesh_dir+mesh_name+".msh", mesh_dir+mesh_name+".xml", "gmsh")

        # Convert the .msh file to a .xdmf file
        MPI.barrier(MPI.comm_world)
        mesh = Mesh(mesh_dir+mesh_name+".xml")
        geo_mesh = XDMFFile(MPI.comm_world, mesh_dir+mesh_name+".xdmf")
        geo_mesh.write(mesh)
        geo_mesh.read(temp_mesh)

    else:
        geo_mesh = XDMFFile(MPI.comm_world, mesh_dir+mesh_name+".xdmf")
        geo_mesh.read(temp_mesh)

    return temp_mesh

def save_pickle_object(savedir, object, object_filename):
    object2file = open(savedir+object_filename+'.pickle', 'wb')
    pickle.dump(object, object2file, pickle.HIGHEST_PROTOCOL)
    object2file.close()

def load_pickle_object(savedir, object_filename):
    file2object = open(savedir+object_filename+'.pickle', 'rb')
    object = pickle.load(file2object)
    file2object.close()
    return object

def save_current_figure(savedir, xlabel, xlabelfontsize, ylabel, ylabelfontsize, name):
    plt.xlabel(xlabel, fontsize=xlabelfontsize)
    plt.ylabel(ylabel, fontsize=ylabelfontsize)
    plt.tight_layout()
    plt.savefig(savedir+name+".pdf", transparent=True)
    # plt.savefig(savedir+name+".eps", format='eps', dpi=1000, transparent=True)
    plt.close()

def save_current_figure_no_labels(savedir, name):
    plt.tight_layout()
    plt.savefig(savedir+name+".pdf", transparent=True)
    # plt.savefig(savedir+name+".eps", format='eps', dpi=1000, transparent=True)
    plt.close()

# Edit the define_deformation function here -- look at the composite ufjc network code for assistance in this
def define_deformation(self):
    """
    Define the applied deformation history
    """
    deformation = SimpleNamespace()

    if self.dp.t_step > self.dp.t_max:
        sys.exit('Error: The time step is larger than the total deformation time! Adjust the value of t_step_modify_factor to correct for this.')

    # initialize the chunk counter and associated constants/lists
    chunk_counter  = 0
    chunk_indx_val = 0
    chunk_indx     = []

    # Initialization step: allocate time and stretch results, dependent upon the type of deformation being accounted for 
    t_val    = self.dp.t_min # initialize the time value at zero
    t        = [] # sec
    t_chunks = [] # sec
    lmbda    = self.initialize_lmbda()

    # Append to appropriate lists
    t.append(t_val)
    t_chunks.append(t_val)
    chunk_indx.append(chunk_indx_val)
    lmbda = self.store_initialized_lmbda(lmbda)

    # update the chunk iteration counter
    chunk_counter  += 1
    chunk_indx_val += 1

    # advance to the first time step
    t_val += self.dp.t_step

    while t_val <= (self.dp.t_max+cond_val):
        # Calculate displacement at a particular time step
        lmbda_val = self.calculate_lmbda_func(t_val)

        # Append to appropriate lists
        t.append(t_val)
        lmbda = self.store_calculated_lmbda(lmbda, lmbda_val)

        if chunk_counter == self.dp.t_step_chunk_num:
            # Append to appropriate lists
            t_chunks.append(t_val)
            chunk_indx.append(chunk_indx_val)
            lmbda = self.store_calculated_lmbda_chunk_post_processing(lmbda, lmbda_val)

            # update the time step chunk iteration counter
            chunk_counter = 0

        # advance to the next time step
        t_val          += self.dp.t_step
        chunk_counter  += 1
        chunk_indx_val += 1
    
    u = self.calculate_u_func(lmbda)

    # If the endpoint of the chunked applied deformation is not equal to the true endpoint of the applied deformation, then give the user the option to kill the simulation, or proceed on
    if chunk_indx[-1] != len(t)-1:
        terminal_statement = input('The endpoint of the chunked applied deformation is not equal to the endpoint of the actual applied deformation. Do you wish to kill the simulation here? If no, the simulation will proceed on. ')
        if terminal_statement.lower() == 'yes':
            sys.exit()
        else: pass
    
    deformation.t          = t
    deformation.t_chunks   = t_chunks
    deformation.chunk_indx = chunk_indx
    deformation            = self.save2deformation(deformation, lmbda, u)

    for key, value in vars(deformation).items(): setattr(self, key, value)

# Initializing necessary parameters and finite element mesh

# Set the mpi communicator of the object
comm_rank = MPI.rank(MPI.comm_world)
comm_size = MPI.size(MPI.comm_world)

# Pre-processing parameters
form_compiler_optimize          = True
form_compiler_cpp_optimize      = True
form_compiler_representation    = "uflacs"
form_compiler_quadrature_degree = 4

set_log_level(LogLevel.WARNING)
parameters["form_compiler"]["optimize"]          = form_compiler_optimize
parameters["form_compiler"]["cpp_optimize"]      = form_compiler_cpp_optimize
parameters["form_compiler"]["representation"]    = form_compiler_representation
parameters["form_compiler"]["quadrature_degree"] = form_compiler_quadrature_degree
info(parameters, True)

# Material initialization
physical_dimensionality      = "two_dimensional"
two_dimensional_formulation  = "plane_strain"
incompressibility_assumption = "nearly_incompressible"
phenomenological_model       = "neo_hookean"
rate_dependence              = "rate_independent"
mu = 1 # nondimensionalized shear modulus

# Mesh initialization
mesh_type = "notched_crack"
L, H = 1.0, 1.5
x_notch_point = 0.05 # 0.5
r_notch = 0.05
notch_fine_mesh_layer_level_num = 1
fine_mesh_elem_size = 0.01
coarse_mesh_elem_size = 0.01 # 0.01 # 0.25
l_nl = 1.25*r_notch # 15*r_notch # 10*r_notch # 1.25*r_notch # 0.02

geofile = \
            """
            Mesh.Algorithm = 8;
            coarse_mesh_elem_size = DefineNumber[ %g, Name "Parameters/coarse_mesh_elem_size" ];
            x_notch_point = DefineNumber[ %g, Name "Parameters/x_notch_point" ];
            r_notch = DefineNumber[ %g, Name "Parameters/r_notch" ];
            L = DefineNumber[ %g, Name "Parameters/L"];
            H = DefineNumber[ %g, Name "Parameters/H"];
            Point(1) = {0, 0, 0, coarse_mesh_elem_size};
            Point(2) = {0, -r_notch, 0, coarse_mesh_elem_size};
            Point(3) = {0, -H/2, 0, coarse_mesh_elem_size};
            Point(4) = {L, -H/2, 0, coarse_mesh_elem_size};
            Point(5) = {L, H/2, 0, coarse_mesh_elem_size};
            Point(6) = {0, H/2, 0, coarse_mesh_elem_size};
            Point(7) = {0, r_notch, 0, coarse_mesh_elem_size};
            Point(8) = {r_notch, 0, 0, coarse_mesh_elem_size};
            Line(1) = {2, 3};
            Line(2) = {3, 4};
            Line(3) = {4, 5};
            Line(4) = {5, 6};
            Line(5) = {6, 7};
            Circle(6) = {7, 1, 8};
            Circle(7) = {8, 1, 2};
            Curve Loop(21) = {1, 2, 3, 4, 5, 6, 7};
            Plane Surface(31) = {21};
            Mesh.MshFileVersion = 2.0;
            """ % (coarse_mesh_elem_size, x_notch_point, r_notch, L, H)

geofile = textwrap.dedent(geofile)

L_string           = "{:.1f}".format(L)
H_string           = "{:.1f}".format(H)
x_notch_point_string = "{:.1f}".format(x_notch_point)
r_notch_string     = "{:.1f}".format(r_notch)
coarse_mesh_elem_size_string  = "{:.1f}".format(coarse_mesh_elem_size)

mesh_name = physical_dimensionality+"_"+two_dimensional_formulation+"_"+mesh_type+"_"+L_string+"_"+H_string+"_"+x_notch_point_string+"_"+r_notch_string+"_"+coarse_mesh_elem_size_string

# Create simulation directory
prefix = physical_dimensionality+"_"+two_dimensional_formulation+"_"+incompressibility_assumption+"_"+phenomenological_model+"_"+rate_dependence+"_"+mesh_type
savedir = generate_savedir(prefix)

# Create and save mesh
mesh = gmsh_mesher(geofile, prefix, mesh_name)
dimension = mesh.geometry().dim()

# Geometry parameters
x_notch_surface_point = x_notch_point
y_notch_surface_point = 0

notch_surface_point = (x_notch_surface_point,y_notch_surface_point)

x_notch_surface_point_string = "{:.4f}".format(x_notch_surface_point)
y_notch_surface_point_string = "{:.4f}".format(y_notch_surface_point)

notch_surface_point_label = '('+x_notch_surface_point_string+', '+y_notch_surface_point_string+')'

meshpoints            = [notch_surface_point]
meshpoints_label_list = [r'$'+notch_surface_point_label+'$']
meshpoints_color_list = ['black']
meshpoints_name_list  = ['notch_surface_point']

# Finite element method parameters
solver_algorithm = "monolithic" # "alternate_minimization"
solver_bounded = False # True

u_degree                = 1
scalar_prmtr_degree     = 1
metadata                = {"quadrature_degree": 4}
three_dim_tensor2vector_indx_dict = {"11": 0, "12": 1, "13": 2, "21": 3, "22": 4, "23": 5, "31": 6, "32": 7, "33": 8}
two_dim_tensor2vector_indx_dict = {"11": 0, "12": 1, "21": 2, "22": 3}

# Deformation parameters
k_cond_val = 0.01
K___mu = 10 # ratio of bulk modulus over shear modulus

# Parameters used in F_func
strain_rate = 0.2 # 0.2 # 1/sec
t_min = 0
t_max       = 18 # 100 # sec
t_step      = 0.02 # sec
t_step_chunk_num = 10

# Monolithic solver parameters
solver_monolithic_parameters = {"nonlinear_solver": "snes",
                                "symmetric": True,
                                "snes_solver": {"linear_solver": "mumps",
                                                "method": "newtontr",
                                                "line_search": "cp",
                                                "preconditioner": "hypre_amg",
                                                "maximum_iterations": 200,
                                                "absolute_tolerance": 1e-8,
                                                "relative_tolerance": 1e-7,
                                                "solution_tolerance": 1e-7,
                                                "report": True,
                                                "error_on_nonconvergence": False}}

# Post-processing parameters
ext = "xdmf"
file_results = "results."+ext

save_u                                    = True
save_sigma_mesh                           = True
save_sigma_chunks                         = True
save_F_mesh                               = True
save_F_chunks                             = True

rewrite_function_mesh = False
flush_output          = True
functions_share_mesh  = True

file_results = XDMFFile(MPI.comm_world, savedir + file_results)
file_results.parameters["rewrite_function_mesh"] = rewrite_function_mesh
file_results.parameters["flush_output"]          = flush_output
file_results.parameters["functions_share_mesh"]  = functions_share_mesh

axes_linewidth      = 1.0
font_family         = "sans-serif"
text_usetex         = True
ytick_right         = True
ytick_direction     = "in"
xtick_top           = True
xtick_direction     = "in"
xtick_minor_visible = True

plt.rcParams['axes.linewidth'] = axes_linewidth # set the value globally
plt.rcParams['font.family']    = font_family
plt.rcParams['text.usetex']    = text_usetex # comment this line out in WSL2, uncomment this line in native Linux on workstation
plt.rcParams['ytick.right']     = ytick_right
plt.rcParams['ytick.direction'] = ytick_direction
plt.rcParams['xtick.top']       = xtick_top
plt.rcParams['xtick.direction'] = xtick_direction
plt.rcParams["xtick.minor.visible"] = xtick_minor_visible

# Set deformation here