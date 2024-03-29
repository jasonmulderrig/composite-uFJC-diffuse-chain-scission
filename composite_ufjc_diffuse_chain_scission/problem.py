# import necessary libraries
from __future__ import division
from dolfin import *
from .default_parameters import default_parameters
from .deformation import AppliedDeformation
from .utility import generate_savedir
import numpy as np
import sys
import os
import pathlib
import matplotlib.pyplot as plt
from types import SimpleNamespace

class CompositeuFJCDiffuseChainScissionProblem(object):
    """
    Problem class for composite uFJC diffuse chain scission models
    """
    def __init__(self):

        # Set the mpi communicator of the object
        self.comm_rank = MPI.rank(MPI.comm_world)

        # Parameters
        self.parameters = default_parameters()
        self.set_user_parameters()

        # Setup filesystem
        self.savedir = generate_savedir(self.prefix())

        # Pre-processing
        ppp = self.parameters.pre_processing
        set_log_level(LogLevel.WARNING)
        parameters["form_compiler"]["optimize"]          = ppp.form_compiler_optimize
        parameters["form_compiler"]["cpp_optimize"]      = ppp.form_compiler_cpp_optimize
        parameters["form_compiler"]["representation"]    = ppp.form_compiler_representation
        parameters["form_compiler"]["quadrature_degree"] = ppp.form_compiler_quadrature_degree
        info(parameters, True)

        self.fem = SimpleNamespace()

        # Mesh
        self.fem.mesh      = self.define_mesh()
        self.fem.dimension = self.fem.mesh.geometry().dim() # spatial dimensions of the mesh
        self.fem.solver_algorithm = self.parameters.fem.solver_algorithm
        self.fem.solver_bounded = self.parameters.fem.solver_bounded

        # MeshFunctions and Measures for different blocks and boundaries; may not need this functionality
        self.set_mesh_functions()
        self.set_measures()

        # Material
        self.material = self.define_material()

        # Variational formulation
        self.set_variational_formulation()

        if self.fem.solver_algorithm == "alternate_minimization":
            # Set Dirichlet boundary conditions for displacement
            self.fem.bc_u = self.define_bc_u()
            # Set Dirichlet boundary conditions for non-local chain stretch
            self.fem.bc_lmbda_c_tilde = self.define_bc_lmbda_c_tilde()
        elif self.fem.solver_algorithm == "monolithic":
            # Set boundary conditions for the monolithic solver
            self.fem.bc_monolithic = self.define_bc_monolithic()

        # Deformation
        self.deformation = AppliedDeformation(self.parameters, self.F_func, self.initialize_lmbda, self.store_initialized_lmbda, self.calculate_lmbda_func, self.store_calculated_lmbda, self.store_calculated_lmbda_chunk_post_processing, self.calculate_u_func, self.save2deformation)

        # Post-processing
        ppp = self.parameters.post_processing
        self.file_results = XDMFFile(MPI.comm_world, self.savedir + ppp.file_results)
        self.file_results.parameters["rewrite_function_mesh"] = ppp.rewrite_function_mesh
        self.file_results.parameters["flush_output"]          = ppp.flush_output
        self.file_results.parameters["functions_share_mesh"]  = ppp.functions_share_mesh

        if self.fem.solver_algorithm == "alternate_minimization":
            self.fem.dict_solver_u_parameters = self.define_dict_solver_u_parameters()
            if self.fem.solver_bounded is True:
                self.fem.dict_solver_bounded_lmbda_c_tilde_parameters = self.define_dict_solver_bounded_lmbda_c_tilde_parameters()
            else:
                self.fem.dict_solver_unbounded_lmbda_c_tilde_parameters = self.define_dict_solver_unbounded_lmbda_c_tilde_parameters()
        elif self.fem.solver_algorithm == "monolithic":
            if self.fem.solver_bounded is True:
                self.fem.dict_solver_bounded_monolithic_parameters = self.define_dict_solver_bounded_monolithic_parameters()
            else:
                self.fem.dict_solver_unbounded_monolithic_parameters = self.define_dict_solver_unbounded_monolithic_parameters()
    
    def define_dict_solver_unbounded_monolithic_parameters(self):
        dict_solver_unbounded_monolithic_parameters = vars(self.parameters.solver_unbounded_monolithic)
        dict_solver_unbounded_monolithic_settings   = vars(self.parameters.solver_unbounded_monolithic_settings)

        nonlinear_solver_type = self.parameters.solver_unbounded_monolithic.nonlinear_solver + "_solver"

        dict_solver_unbounded_monolithic_parameters[nonlinear_solver_type] = dict_solver_unbounded_monolithic_settings

        return dict_solver_unbounded_monolithic_parameters
    
    def define_dict_solver_bounded_monolithic_parameters(self):
        dict_solver_bounded_monolithic_parameters = vars(self.parameters.solver_bounded_monolithic)
        dict_solver_bounded_monolithic_settings   = vars(self.parameters.solver_bounded_monolithic_settings)

        nonlinear_solver_type = self.parameters.solver_bounded_monolithic.nonlinear_solver + "_solver"

        dict_solver_bounded_monolithic_parameters[nonlinear_solver_type] = dict_solver_bounded_monolithic_settings

        return dict_solver_bounded_monolithic_parameters

    def define_dict_solver_bounded_lmbda_c_tilde_parameters(self):
        dict_solver_bounded_lmbda_c_tilde_parameters = vars(self.parameters.solver_bounded_lmbda_c_tilde)
        dict_solver_bounded_lmbda_c_tilde_settings   = vars(self.parameters.solver_bounded_lmbda_c_tilde_settings)

        nonlinear_solver_type = self.parameters.solver_bounded_lmbda_c_tilde.nonlinear_solver + "_solver"

        dict_solver_bounded_lmbda_c_tilde_parameters[nonlinear_solver_type] = dict_solver_bounded_lmbda_c_tilde_settings

        return dict_solver_bounded_lmbda_c_tilde_parameters
    
    def define_dict_solver_unbounded_lmbda_c_tilde_parameters(self):
        dict_solver_unbounded_lmbda_c_tilde_parameters = vars(self.parameters.solver_unbounded_lmbda_c_tilde)
        dict_solver_unbounded_lmbda_c_tilde_settings   = vars(self.parameters.solver_unbounded_lmbda_c_tilde_settings)

        nonlinear_solver_type = self.parameters.solver_unbounded_lmbda_c_tilde.nonlinear_solver + "_solver"

        dict_solver_unbounded_lmbda_c_tilde_parameters[nonlinear_solver_type] = dict_solver_unbounded_lmbda_c_tilde_settings

        return dict_solver_unbounded_lmbda_c_tilde_parameters
    
    def define_dict_solver_u_parameters(self):
        dict_solver_u_parameters = vars(self.parameters.solver_u)
        dict_solver_u_settings   = vars(self.parameters.solver_u_settings)

        nonlinear_solver_type = self.parameters.solver_u.nonlinear_solver + "_solver"

        dict_solver_u_parameters[nonlinear_solver_type] = dict_solver_u_settings

        return dict_solver_u_parameters
    
    def print0(self, text):
        """
        Print only from process 0 (for parallel run)
        """
        if self.comm_rank == 0: print(text)

    def set_user_parameters(self):
        """
        Set the user parameters defining the problem
        """
        pass
    
    def define_mesh(self):
        """
        Define the mesh for the problem
        """
        pass
    
    def set_mesh_functions(self):
        """
        Set meshfunctions with boundaries and subdomain indicators
        """
        # self.fem.cells_meshfunction = MeshFunction("size_t", self.fem.mesh, self.fem.dimension)
        # self.fem.cells_meshfunction.set_all(0)
        # self.fem.exterior_facets_meshfunction = MeshFunction("size_t", self.fem.mesh, self.fem.dimension-1)
        # self.fem.exterior_facets_meshfunction.set_all(0)
        pass
    
    def set_measures(self):
        """
        Assign the Measure to get selective integration on boundaries and bulk subdomain
        The Measure is defined using self.fem.cells_meshfunction and self.fem.exterior_facets_meshfunction
        """
        # try:
        #     self.dx = Measure("dx")(subdomain_data=self.cells_meshfunction)
        # except:
        #     self.dx = dx
        # try:
        #     self.ds = Measure("ds")(subdomain_data=self.exterior_facets_meshfunction)
        # except:
        #     self.ds = ds
        pass
    
    def define_material(self):
        """
        Return material that will be set in the model
        """
        pass
    
    def set_variational_formulation(self):
        """
        Define the variational formulation problem to be solved
        """
        self.material.fenics_variational_formulation(self.parameters, self.fem)
    
    def define_bc_monolithic(self):
        """
        Return a list of boundary conditions
        """
        return []
    
    def define_bc_u(self):
        """
        Return a list of boundary conditions on the displacement
        """
        return []
    
    def define_bc_lmbda_c_tilde(self):
        """
        Return a list of boundary conditions on the non-local chain stretch
        """
        return []
    
    def F_func(self):
        """
        Function defining the deformation
        """
        pass

    def initialize_lmbda(self):
        pass
    
    def store_initialized_lmbda(self):
        pass
    
    def calculate_lmbda_func(self):
        pass
    
    def store_calculated_lmbda(self):
        pass
    
    def store_calculated_lmbda_chunk_post_processing(self):
        pass
    
    def calculate_u_func(self):
        pass
    
    def save2deformation(self):
        pass
    
    def prefix(self):
        return "problem"
    
    def strong_form_initialize_sigma_chunks(self):
        pass
    
    def lr_cg_deformation_gradient_func(self):
        pass
    
    def strong_form_calculate_sigma_func(self):
        pass
    
    def strong_form_store_calculated_sigma_chunks(self):
        pass
    
    def set_homogeneous_strong_form_deformation_finalization(self):
        """
        Plot the chunked results from the homogeneous strong form deformation
        """
        pass

    def solve_homogeneous_strong_form_deformation(self):
        """
        Solve the evolution problem for homogeneous strong form through the applied deformation history at each time step
        """
        if self.comm_rank == 0:
            # initialization
            strong_form_results, strong_form_chunks = self.material.homogeneous_strong_form_initialization()

            # grab deformation
            deformation = self.deformation

            # time stepping
            for t_indx, t_val in enumerate(deformation.t):
                
                # Update time stepping
                deformation.t_indx = t_indx
                deformation.t_val  = t_val
                print("\033[1;32m--- Time step # {0:2d}: t = {1:.3f} for the homogeneous strong form network deformation ---\033[1;m".format(t_indx, t_val))

                # Solve homogeneous strong form
                strong_form_results = self.material.homogeneous_strong_form_solve_step(deformation, strong_form_results)

                # Post-processing
                if deformation.t_indx in deformation.chunk_indx:
                    strong_form_chunks = self.material.homogeneous_strong_form_chunk_post_processing(deformation, strong_form_results, strong_form_chunks)
            
            # Store chunks and perform any finalizations, such as data visualization
            self.strong_form_chunks = strong_form_chunks
            self.set_homogeneous_strong_form_deformation_finalization()
    
    def set_loading(self):
        """
        Update Dirichlet boundary conditions"
        """
        pass

    def weak_form_initialize_deformation_sigma_chunks(self):
        pass
    
    def weak_form_store_calculated_sigma_chunks(self):
        pass
    
    def weak_form_store_calculated_deformation_chunks(self):
        pass

    def set_user_fenics_weak_form_post_processing(self):
        """
        User post-processing
        """
        pass

    def set_fenics_weak_form_deformation_finalization(self):
        """
        Plot the chunked results from the weak form deformation
        """
        pass

    def solve_fenics_weak_form_deformation(self):
        """
        Solve the evolution problem for the weak form in FEniCS through the applied deformation history at each time step
        """
        # initialization
        self.fem, weak_form_chunks = self.material.fenics_weak_form_initialization(self.fem, self.parameters)

        # grab deformation
        deformation = self.deformation

        # weak form solver setup
        self.fem = self.material.fenics_weak_form_solver_setup(self.fem)

        # time stepping
        for t_indx, t_val in enumerate(deformation.t):
            
            # Update time stepping
            deformation.t_indx = t_indx
            deformation.t_val  = t_val
            self.print0("\033[1;32m--- Time step # {0:2d}: t = {1:.3f} for the weak form network deformation ---\033[1;m".format(t_indx, t_val))
            self.set_loading()

            # Solve weak form for displacements, and account for network irreversibility
            self.fem = self.material.fenics_weak_form_solve_step(self.fem)
            
            # Enforce all parallelization to unify after the weak form
            # solve step and before post-processing
            MPI.barrier(MPI.comm_world)

            # Post-processing
            if deformation.t_indx in deformation.chunk_indx: # the elements in chunk_indx are guaranteed to be unique
                weak_form_chunks = self.material.fenics_weak_form_chunk_post_processing(deformation, weak_form_chunks, self.fem, self.file_results, self.parameters)
                MPI.barrier(MPI.comm_world)

            self.set_user_fenics_weak_form_post_processing()
        
        # Store chunks and perform any finalizations, such as data visualization
        self.weak_form_chunks = weak_form_chunks
        self.set_fenics_weak_form_deformation_finalization()




