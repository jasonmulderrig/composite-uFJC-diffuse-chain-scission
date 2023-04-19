# import necessary libraries
from __future__ import division
from dolfin import *
from composite_ufjc_diffuse_chain_scission import (
    CompositeuFJCDiffuseChainScissionProblem,
    TwoDimensionalPlaneStrainNearlyIncompressibleNonaffineEightChainModelEqualStrainRateIndependentCompositeuFJCNetwork,
    gmsh_mesher,
    mesh_topologier,
    latex_formatting_figure,
    save_current_figure
)
import numpy as np
import matplotlib.pyplot as plt
from copy import deepcopy
import textwrap


class TwoDimensionalPlaneStrainNearlyIncompressibleNonaffineEightChainModelEqualStrainRateIndependentNotchedCrack(CompositeuFJCDiffuseChainScissionProblem):

    def __init__(self, L, H, x_notch_point, r_notch, notch_fine_mesh_layer_level_num=2, fine_mesh_elem_size=0.001, coarse_mesh_elem_size=0.1, l_nl=0.1):

        self.L = L
        self.H = H
        self.x_notch_point = x_notch_point
        self.r_notch = r_notch
        self.notch_fine_mesh_layer_level_num = notch_fine_mesh_layer_level_num
        self.fine_mesh_elem_size = fine_mesh_elem_size
        self.coarse_mesh_elem_size  = coarse_mesh_elem_size
        self.l_nl = l_nl
        
        CompositeuFJCDiffuseChainScissionProblem.__init__(self)
    
    def set_user_parameters(self):
        """
        Set the user parameters defining the problem
        """

        p = self.parameters

        # x_notch_center_point = self.x_notch_point - self.r_notch

        x_notch_surface_point = self.x_notch_point # x_notch_center_point + np.cos(np.pi/4)*self.r_notch
        y_notch_surface_point = 0 # np.sin(np.pi/4)*self.r_notch

        notch_surface_point = (x_notch_surface_point,y_notch_surface_point)

        x_notch_surface_point_string = "{:.4f}".format(x_notch_surface_point)
        y_notch_surface_point_string = "{:.4f}".format(y_notch_surface_point)

        notch_surface_point_label = '('+x_notch_surface_point_string+', '+y_notch_surface_point_string+')'

        p.geometry.meshpoints            = [notch_surface_point]
        p.geometry.meshpoints_label_list = [r'$'+notch_surface_point_label+'$']
        p.geometry.meshpoints_color_list = ['black']
        p.geometry.meshpoints_name_list  = ['notch_surface_point']

        p.geometry.meshtype = "notched_crack" # "notch_on_edgeface"

        p.material.l_nl = self.l_nl

        # Define a rather brittle material for debugging purposes
        p.material.zeta_nu_char = 298.9 # 537.6 # 298.9 # 100 # 50
        p.material.kappa_nu     = 912.2 # 3197.5 # 912.2 # 2300 # 7500

        p.material.macro2micro_deformation_assumption = "nonaffine"

        # Define the chain length statistics in the network
        nu_distribution = "uniform"
        nu_list         = [6] # [i for i in range(5, 16)] # nu = 5 -> nu = 15
        nu_min          = min(nu_list)
        nu_bar          = 6 # 8
        Delta_nu        = nu_bar-nu_min
        nu_list         = nu_list
        nu_min          = nu_min
        nu_bar          = nu_bar
        Delta_nu        = Delta_nu

        p.material.nu_distribution = nu_distribution
        p.material.nu_list         = nu_list
        p.material.nu_min          = nu_min
        p.material.nu_bar          = nu_bar
        p.material.Delta_nu        = Delta_nu
        p.material.nu_list         = nu_list
        p.material.nu_min          = nu_min
        p.material.nu_bar          = nu_bar
        p.material.Delta_nu        = Delta_nu

        # Define chain lengths to chunk during deformation
        nu_chunks_list = nu_list # nu_list[0:3]
        nu_chunks_indx_list = nu_chunks_list.copy()
        for i in range(len(nu_chunks_list)):
            nu_chunks_indx_list[i] = nu_list.index(nu_chunks_list[i])
        nu_chunks_label_list = [r'$\nu='+str(nu_list[nu_chunks_indx_list[i]])+'$' for i in range(len(nu_chunks_list))]
        nu_chunks_color_list = ['blue'] # ['orange', 'blue', 'green'] # , 'red', 'purple', 'brown']
        
        p.material.nu_chunks_list       = nu_chunks_list
        p.material.nu_chunks_indx_list  = nu_chunks_indx_list
        p.material.nu_chunks_label_list = nu_chunks_label_list
        p.material.nu_chunks_color_list = nu_chunks_color_list

        p.fem.solver_algorithm = "monolithic" # "alternate_minimization"
        p.fem.solver_bounded = False # True

        # General network deformation parameters
        k_cond_val = 1e-4 # 0.01
        K_G = 10

        # Parameters used in F_func
        strain_rate = 0.1 # 0.2 # 1/sec
        t_max       = 30 # 13.6 # 13.5 # 16 # 100 # sec
        t_step      = 0.02 # 0.01 # 0.02 # sec
        t_step_chunk_num = 10

        p.deformation.k_cond_val       = k_cond_val
        p.deformation.K_G              = K_G
        p.deformation.deformation_type = "uniaxial"
        p.deformation.strain_rate      = strain_rate
        p.deformation.t_max            = t_max
        p.deformation.t_step           = t_step
        p.deformation.t_step_chunk_num = t_step_chunk_num

        p.deformation.tol_lmbda_c_tilde_val = 1e-3

        p.post_processing.save_lmbda_c_eq_chunks           = True
        p.post_processing.save_lmbda_nu_chunks             = True
        p.post_processing.save_lmbda_c_eq_tilde_chunks     = True
        p.post_processing.save_lmbda_nu_tilde_chunks       = True
        p.post_processing.save_lmbda_c_eq_tilde_max_chunks = True
        p.post_processing.save_lmbda_nu_tilde_max_chunks   = True
        p.post_processing.save_upsilon_c_chunks            = True
        p.post_processing.save_d_c_mesh                    = True
        p.post_processing.save_d_c_chunks                  = True
    
    def prefix(self):
        mp = self.parameters.material
        gp = self.parameters.geometry
        return mp.physical_dimensionality+"_"+mp.two_dimensional_formulation+"_"+mp.incompressibility_assumption+"_"+mp.macro2micro_deformation_assumption+"_"+mp.micro2macro_homogenization_scheme+"_"+mp.chain_level_load_sharing+"_"+mp.rate_dependence+"_"+gp.meshtype
    
    def define_mesh(self):
        """
        Define the mesh for the problem
        """
        # use obsolete version of string formatting here because brackets are essential for use in the gmsh script
        # geofile = \
        #     """
        #     Mesh.Algorithm = 8;
        #     notch_fine_mesh_layer_level_num = DefineNumber[ %g, Name "Parameters/notch_fine_mesh_layer_level_num" ];
        #     fine_mesh_elem_size = DefineNumber[ %g, Name "Parameters/fine_mesh_elem_size" ];
        #     coarse_mesh_elem_size = DefineNumber[ %g, Name "Parameters/coarse_mesh_elem_size" ];
        #     x_notch_point = DefineNumber[ %g, Name "Parameters/x_notch_point" ];
        #     r_notch = DefineNumber[ %g, Name "Parameters/r_notch" ];
        #     L = DefineNumber[ %g, Name "Parameters/L"];
        #     H = DefineNumber[ %g, Name "Parameters/H"];
        #     Point(1) = {0, 0, 0, fine_mesh_elem_size};
        #     Point(2) = {x_notch_point-r_notch, 0, 0, fine_mesh_elem_size};
        #     Point(3) = {x_notch_point, 0, 0, fine_mesh_elem_size};
        #     Point(4) = {L, 0, 0, fine_mesh_elem_size};
        #     Point(5) = {L, r_notch+notch_fine_mesh_layer_level_num*fine_mesh_elem_size, 0, fine_mesh_elem_size};
        #     Point(6) = {L, H, 0, coarse_mesh_elem_size};
        #     Point(7) = {0, H, 0, coarse_mesh_elem_size};
        #     Point(8) = {0, r_notch+notch_fine_mesh_layer_level_num*fine_mesh_elem_size, 0, fine_mesh_elem_size};
        #     Point(9) = {0, r_notch, 0, fine_mesh_elem_size};
        #     Point(10) = {x_notch_point-r_notch, r_notch, 0, fine_mesh_elem_size};
        #     Line(1) = {3, 4};
        #     Line(2) = {4, 5};
        #     Line(3) = {5, 6};
        #     Line(4) = {6, 7};
        #     Line(5) = {7, 8};
        #     Line(6) = {8, 5};
        #     Line(7) = {8, 9};
        #     Line(8) = {9, 10};
        #     Circle(9) = {10, 2, 3};
        #     Curve Loop(21) = {1, 2, -6, 7, 8, 9};
        #     Curve Loop(22) = {3, 4, 5, 6};
        #     Plane Surface(31) = {21};
        #     Plane Surface(32) = {21, 22};
        #     Mesh.MshFileVersion = 2.0;
        #     """ % (self.notch_fine_mesh_layer_level_num, self.fine_mesh_elem_size, self.coarse_mesh_elem_size, self.x_notch_point, self.r_notch, self.L, self.H)
        
        # geofile = textwrap.dedent(geofile)

        # L_string           = "{:.1f}".format(self.L)
        # H_string           = "{:.1f}".format(self.H)
        # x_notch_point_string = "{:.1f}".format(self.x_notch_point)
        # r_notch_string     = "{:.1f}".format(self.r_notch)
        # notch_fine_mesh_layer_level_num_string = "{:d}".format(self.notch_fine_mesh_layer_level_num)
        # fine_mesh_elem_size_string = "{:.3f}".format(self.fine_mesh_elem_size)
        # coarse_mesh_elem_size_string  = "{:.1f}".format(self.coarse_mesh_elem_size)

        # mesh_type = "two_dimensional_plane_strain_notched_crack"
        # mesh_name = mesh_type+"_"+L_string+"_"+H_string+"_"+x_notch_point_string+"_"+r_notch_string+"_"+notch_fine_mesh_layer_level_num_string+"_"+fine_mesh_elem_size_string+"_"+coarse_mesh_elem_size_string

        # return gmsh_mesher(geofile, self.prefix(), mesh_name)
        
        geofile = \
            """
            Mesh.Algorithm = 8;
            coarse_mesh_elem_size = DefineNumber[ %g, Name "Parameters/coarse_mesh_elem_size" ];
            x_notch_point = DefineNumber[ %g, Name "Parameters/x_notch_point" ];
            r_notch = DefineNumber[ %g, Name "Parameters/r_notch" ];
            L = DefineNumber[ %g, Name "Parameters/L"];
            H = DefineNumber[ %g, Name "Parameters/H"];
            Point(1) = {0, 0, 0, coarse_mesh_elem_size};
            Point(2) = {x_notch_point-r_notch, 0, 0, coarse_mesh_elem_size};
            Point(3) = {0, -r_notch, 0, coarse_mesh_elem_size};
            Point(4) = {0, -H/2, 0, coarse_mesh_elem_size};
            Point(5) = {L, -H/2, 0, coarse_mesh_elem_size};
            Point(6) = {L, H/2, 0, coarse_mesh_elem_size};
            Point(7) = {0, H/2, 0, coarse_mesh_elem_size};
            Point(8) = {0, r_notch, 0, coarse_mesh_elem_size};
            Point(9) = {x_notch_point-r_notch, r_notch, 0, coarse_mesh_elem_size};
            Point(10) = {x_notch_point, 0, 0, coarse_mesh_elem_size};
            Point(11) = {x_notch_point-r_notch, -r_notch, 0, coarse_mesh_elem_size};
            Line(1) = {11, 3};
            Line(2) = {3, 4};
            Line(3) = {4, 5};
            Line(4) = {5, 6};
            Line(5) = {6, 7};
            Line(6) = {7, 8};
            Line(7) = {8, 9};
            Circle(8) = {9, 2, 10};
            Circle(9) = {10, 2, 11};
            Curve Loop(21) = {1, 2, 3, 4, 5, 6, 7, 8, 9};
            Plane Surface(31) = {21};
            Mesh.MshFileVersion = 2.0;
            """ % (self.coarse_mesh_elem_size, self.x_notch_point, self.r_notch, self.L, self.H)

        # geofile = \
        #     """
        #     Mesh.Algorithm = 8;
        #     coarse_mesh_elem_size = DefineNumber[ %g, Name "Parameters/coarse_mesh_elem_size" ];
        #     x_notch_point = DefineNumber[ %g, Name "Parameters/x_notch_point" ];
        #     r_notch = DefineNumber[ %g, Name "Parameters/r_notch" ];
        #     L = DefineNumber[ %g, Name "Parameters/L"];
        #     H = DefineNumber[ %g, Name "Parameters/H"];
        #     Point(1) = {0, 0, 0, coarse_mesh_elem_size};
        #     Point(2) = {0, -r_notch, 0, coarse_mesh_elem_size};
        #     Point(3) = {0, -H/2, 0, coarse_mesh_elem_size};
        #     Point(4) = {L, -H/2, 0, coarse_mesh_elem_size};
        #     Point(5) = {L, H/2, 0, coarse_mesh_elem_size};
        #     Point(6) = {0, H/2, 0, coarse_mesh_elem_size};
        #     Point(7) = {0, r_notch, 0, coarse_mesh_elem_size};
        #     Point(8) = {r_notch, 0, 0, coarse_mesh_elem_size};
        #     Line(1) = {2, 3};
        #     Line(2) = {3, 4};
        #     Line(3) = {4, 5};
        #     Line(4) = {5, 6};
        #     Line(5) = {6, 7};
        #     Circle(6) = {7, 1, 8};
        #     Circle(7) = {8, 1, 2};
        #     Curve Loop(21) = {1, 2, 3, 4, 5, 6, 7};
        #     Plane Surface(31) = {21};
        #     Mesh.MshFileVersion = 2.0;
        #     """ % (self.coarse_mesh_elem_size, self.x_notch_point, self.r_notch, self.L, self.H)

        geofile = textwrap.dedent(geofile)

        L_string           = "{:.1f}".format(self.L)
        H_string           = "{:.1f}".format(self.H)
        x_notch_point_string = "{:.1f}".format(self.x_notch_point)
        r_notch_string     = "{:.1f}".format(self.r_notch)
        coarse_mesh_elem_size_string  = "{:.1f}".format(self.coarse_mesh_elem_size)

        mp = self.parameters.material
        gp = self.parameters.geometry
        mesh_name = mp.physical_dimensionality+"_"+mp.two_dimensional_formulation+"_"+gp.meshtype+"_"+L_string+"_"+H_string+"_"+x_notch_point_string+"_"+r_notch_string+"_"+coarse_mesh_elem_size_string

        return gmsh_mesher(geofile, self.prefix(), mesh_name)
    
    def F_func(self, t):
        """
        Function defining the deformation
        """
        dp = self.parameters.deformation

        return 1 + dp.strain_rate*(t-dp.t_min)
    
    def initialize_lmbda(self):
        lmbda_2        = [] # unitless
        lmbda_2_chunks = [] # unitless

        return lmbda_2, lmbda_2_chunks
    
    def store_initialized_lmbda(self, lmbda):
        lmbda_2_val = 1 # assuming no pre-stretching
        
        lmbda_2        = lmbda[0]
        lmbda_2_chunks = lmbda[1]
        
        lmbda_2.append(lmbda_2_val)
        lmbda_2_chunks.append(lmbda_2_val)
        
        return lmbda_2, lmbda_2_chunks
    
    def calculate_lmbda_func(self, t_val):
        lmbda_2_val = self.F_func(t_val)

        return lmbda_2_val
    
    def store_calculated_lmbda(self, lmbda, lmbda_val):
        lmbda_2        = lmbda[0]
        lmbda_2_chunks = lmbda[1]
        lmbda_2_val    = lmbda_val
        
        lmbda_2.append(lmbda_2_val)
        
        return lmbda_2, lmbda_2_chunks
    
    def store_calculated_lmbda_chunk_post_processing(self, lmbda, lmbda_val):
        lmbda_2        = lmbda[0]
        lmbda_2_chunks = lmbda[1]
        lmbda_2_val    = lmbda_val
        
        lmbda_2_chunks.append(lmbda_2_val)
        
        return lmbda_2, lmbda_2_chunks
    
    def calculate_u_func(self, lmbda):
        lmbda_2        = lmbda[0]
        lmbda_2_chunks = lmbda[1]

        u_2        = [lmbda_2_val-1 for lmbda_2_val in lmbda_2]
        u_2_chunks = [lmbda_2_chunks_val-1 for lmbda_2_chunks_val in lmbda_2_chunks]

        return u_2, u_2_chunks
    
    def save2deformation(self, deformation, lmbda, u):
        lmbda_2        = lmbda[0]
        lmbda_2_chunks = lmbda[1]

        u_2        = u[0]
        u_2_chunks = u[1]

        deformation.lmbda_2        = lmbda_2
        deformation.lmbda_2_chunks = lmbda_2_chunks
        deformation.u_2            = u_2
        deformation.u_2_chunks     = u_2_chunks

        return deformation
    
    def strong_form_initialize_sigma_chunks(self, chunks):
        sigma_22_chunks = [] # unitless
        chunks.sigma_22_chunks = sigma_22_chunks

        return chunks
    
    def lr_cg_deformation_gradient_func(self, deformation):
        lmbda_2_val = deformation.lmbda_2[deformation.t_indx]
        lmbda_1_val = 1./lmbda_2_val
        F_val = np.diagflat([lmbda_1_val, lmbda_2_val])
        C_val = np.einsum('jJ,jK->JK', F_val, F_val)
        b_val = np.einsum('jJ,kJ->jk', F_val, F_val)

        return F_val, C_val, b_val
    
    def strong_form_calculate_sigma_func(self, sigma_hyp_val, deformation):
        dp = self.parameters.deformation

        F_val, C_val, b_val = self.lr_cg_deformation_gradient_func(deformation)
        
        sigma_22_val = sigma_hyp_val*b_val[1][1] + dp.K_G*( np.linalg.det(F_val) - 1. )

        return sigma_22_val
    
    def strong_form_store_calculated_sigma_chunks(self, sigma_val, chunks):
        sigma_22_val = sigma_val
        chunks.sigma_22_chunks.append(sigma_22_val)

        return chunks
    
    def weak_form_initialize_deformation_sigma_chunks(self, meshpoints, chunks):
        chunks.F_22_chunks         = []
        chunks.F_22_chunks_val     = [0. for meshpoint_indx in range(len(meshpoints))]
        chunks.sigma_22_chunks     = []
        chunks.sigma_22_chunks_val = [0. for meshpoint_indx in range(len(meshpoints))]
        chunks.sigma_22_penalty_term_chunks     = []
        chunks.sigma_22_penalty_term_chunks_val = [0. for meshpoint_indx in range(len(meshpoints))]
        chunks.sigma_22_less_penalty_term_chunks     = []
        chunks.sigma_22_less_penalty_term_chunks_val = [0. for meshpoint_indx in range(len(meshpoints))]

        return chunks
    
    def weak_form_store_calculated_sigma_chunks(self, sigma_val, sigma_penalty_term_val, sigma_less_penalty_term_val, two_dim_tensor2vector_indx_dict, meshpoints, chunks):
        for meshpoint_indx in range(len(meshpoints)):
            chunks.sigma_22_chunks_val[meshpoint_indx] = sigma_val(meshpoints[meshpoint_indx])[two_dim_tensor2vector_indx_dict["22"]]
            chunks.sigma_22_penalty_term_chunks_val[meshpoint_indx] = sigma_penalty_term_val(meshpoints[meshpoint_indx])[two_dim_tensor2vector_indx_dict["22"]]
            chunks.sigma_22_less_penalty_term_chunks_val[meshpoint_indx] = sigma_less_penalty_term_val(meshpoints[meshpoint_indx])[two_dim_tensor2vector_indx_dict["22"]]
        chunks.sigma_22_chunks.append(deepcopy(chunks.sigma_22_chunks_val))
        chunks.sigma_22_penalty_term_chunks.append(deepcopy(chunks.sigma_22_penalty_term_chunks_val))
        chunks.sigma_22_less_penalty_term_chunks.append(deepcopy(chunks.sigma_22_less_penalty_term_chunks_val))

        return chunks
    
    def weak_form_store_calculated_deformation_chunks(self, F_val, two_dim_tensor2vector_indx_dict, meshpoints, chunks):
        for meshpoint_indx in range(len(meshpoints)):
            chunks.F_22_chunks_val[meshpoint_indx] = F_val(meshpoints[meshpoint_indx])[two_dim_tensor2vector_indx_dict["22"]]
        chunks.F_22_chunks.append(deepcopy(chunks.F_22_chunks_val))

        return chunks

    def define_material(self):
        """
        Return material that will be set in the model
        """
        material = TwoDimensionalPlaneStrainNearlyIncompressibleNonaffineEightChainModelEqualStrainRateIndependentCompositeuFJCNetwork(self.parameters, self.strong_form_initialize_sigma_chunks, self.lr_cg_deformation_gradient_func, self.strong_form_calculate_sigma_func, self.strong_form_store_calculated_sigma_chunks, self.weak_form_initialize_deformation_sigma_chunks, self.weak_form_store_calculated_sigma_chunks, self.weak_form_store_calculated_deformation_chunks)
        
        return material
    
    def define_bc_monolithic(self):
        """
        Return a list of boundary conditions
        """
        self.fem.lines = MeshFunction("size_t", self.fem.mesh, self.fem.mesh.topology().dim()-1)
        self.fem.lines.set_all(0)

        L = self.L
        H = self.H
        x_notch_point = self.x_notch_point
        r_notch = self.r_notch

        class LeftBoundary(SubDomain):
            def inside(self, x, on_boundary):
                return near(x[0], 0., DOLFIN_EPS)
        
        class RightBoundary(SubDomain):
            def inside(self, x, on_boundary):
                return near(x[0], L, DOLFIN_EPS)

        class BottomBoundary(SubDomain):
            def inside(self, x, on_boundary):
                return near(x[1], -H/2., DOLFIN_EPS)
        
        class TopBoundary(SubDomain):
            def inside(self, x, on_boundary):
                return near(x[1], H/2., DOLFIN_EPS)

        class Notch(SubDomain):
            def inside(self, x, on_boundary):
                r_notch_sq = (x[0]-(x_notch_point-r_notch))**2 + x[1]**2
                return r_notch_sq <= (r_notch + DOLFIN_EPS)**2

        LeftBoundary().mark(self.fem.lines, 1)
        RightBoundary().mark(self.fem.lines, 2)
        BottomBoundary().mark(self.fem.lines, 3)
        TopBoundary().mark(self.fem.lines, 4)
        Notch().mark(self.fem.lines, 5)

        mesh_topologier(self.fem.lines, self.prefix(), "lines")

        self.fem.u_y_expression = Expression("u_y", u_y=0., degree=0)

        bc_I = DirichletBC(self.fem.V.sub(0).sub(1), Constant(0.), BottomBoundary())
        bc_II = DirichletBC(self.fem.V.sub(0).sub(0), Constant(0.), RightBoundary())
        bc_III  = DirichletBC(self.fem.V.sub(0).sub(1), self.fem.u_y_expression, TopBoundary())

        return [bc_I, bc_II, bc_III]

        # bc_I   = DirichletBC(self.fem.V.sub(0).sub(0), Constant(0.), BottomBoundary())
        # bc_II  = DirichletBC(self.fem.V.sub(0).sub(1), Constant(0.), BottomBoundary())
        # bc_III = DirichletBC(self.fem.V.sub(0).sub(0), Constant(0.), TopBoundary())
        # bc_IV  = DirichletBC(self.fem.V.sub(0).sub(1), self.fem.u_y_expression, TopBoundary())

        # return [bc_I, bc_II, bc_III, bc_IV]
    
        # pinned_support_expression = Expression(["0.0", "0.0"], degree=0)
        # self.fem.u_y_expression = Expression(["0.0", "u_y"], u_y=0., degree=0)

        # bc_I   = DirichletBC(self.fem.V.sub(0), pinned_support_expression, BottomBoundary())
        # bc_II  = DirichletBC(self.fem.V.sub(0), self.fem.u_y_expression, TopBoundary())

        # return [bc_I, bc_II]
    
    def define_bc_u(self):
        """
        Return a list of displacement-controlled (Dirichlet) boundary conditions
        """
        self.fem.lines = MeshFunction("size_t", self.fem.mesh, self.fem.mesh.topology().dim()-1)
        self.fem.lines.set_all(0)

        L = self.L
        H = self.H
        x_notch_point = self.x_notch_point
        r_notch = self.r_notch

        class LeftBoundary(SubDomain):
            def inside(self, x, on_boundary):
                return near(x[0], 0., DOLFIN_EPS)
        
        class RightBoundary(SubDomain):
            def inside(self, x, on_boundary):
                return near(x[0], L, DOLFIN_EPS)

        class BottomBoundary(SubDomain):
            def inside(self, x, on_boundary):
                return near(x[1], -H/2., DOLFIN_EPS)
        
        class TopBoundary(SubDomain):
            def inside(self, x, on_boundary):
                return near(x[1], H/2., DOLFIN_EPS)

        class Notch(SubDomain):
            def inside(self, x, on_boundary):
                r_notch_sq = (x[0]-(x_notch_point-r_notch))**2 + x[1]**2
                return r_notch_sq <= (r_notch + DOLFIN_EPS)**2

        LeftBoundary().mark(self.fem.lines, 1)
        RightBoundary().mark(self.fem.lines, 2)
        BottomBoundary().mark(self.fem.lines, 3)
        TopBoundary().mark(self.fem.lines, 4)
        Notch().mark(self.fem.lines, 5)

        mesh_topologier(self.fem.lines, self.prefix(), "lines")

        self.fem.u_y_expression = Expression("u_y", u_y=0., degree=0)

        bc_I = DirichletBC(self.fem.V_u.sub(1), Constant(0.), BottomBoundary())
        bc_II = DirichletBC(self.fem.V_u.sub(0), Constant(0.), RightBoundary())
        bc_III  = DirichletBC(self.fem.V_u.sub(1), self.fem.u_y_expression, TopBoundary())

        return [bc_I, bc_II, bc_III]

        # bc_I   = DirichletBC(self.fem.V_u.sub(0), Constant(0.), BottomBoundary())
        # bc_II  = DirichletBC(self.fem.V_u.sub(1), Constant(0.), BottomBoundary())
        # bc_III = DirichletBC(self.fem.V_u.sub(0), Constant(0.), TopBoundary())
        # bc_IV  = DirichletBC(self.fem.V_u.sub(1), self.fem.u_y_expression, TopBoundary())

        # return [bc_I, bc_II, bc_III, bc_IV]

    def set_loading(self):
        """
        Update Dirichlet boundary conditions"
        """
        self.fem.u_y_expression.u_y = self.H*self.deformation.u_2[self.deformation.t_indx]

    def set_fenics_weak_form_deformation_finalization(self):
        """
        Plot the chunked results from the weak form deformation in FEniCS
        """

        gp               = self.parameters.geometry
        mp               = self.parameters.material
        ppp              = self.parameters.post_processing
        deformation      = self.deformation
        weak_form_chunks = self.weak_form_chunks
        
        # plot results
        latex_formatting_figure(ppp)

        # lmbda_c
        if ppp.save_lmbda_c_chunks:
            fig = plt.figure()
            for meshpoint_indx in range(len(gp.meshpoints)):
                lmbda_c___meshpoint_chunk = [lmbda_c_chunk[meshpoint_indx] for lmbda_c_chunk in weak_form_chunks.lmbda_c_chunks]
                plt.plot(deformation.t_chunks, lmbda_c___meshpoint_chunk, linestyle='-', color=gp.meshpoints_color_list[meshpoint_indx], alpha=1, linewidth=2.5, label=gp.meshpoints_label_list[meshpoint_indx])
            plt.legend(loc='best')
            plt.grid(True, alpha=0.25)
            save_current_figure(self.savedir, r'$t$', 30, r'$\lambda_c$', 30, "fenics-weak-form-uniaxial-rate-independent-t-vs-lmbda_c")

            fig = plt.figure()
            for meshpoint_indx in range(len(gp.meshpoints)):
                lmbda_c___meshpoint_chunk = [lmbda_c_chunk[meshpoint_indx] for lmbda_c_chunk in weak_form_chunks.lmbda_c_chunks]
                plt.plot(deformation.u_2_chunks, lmbda_c___meshpoint_chunk, linestyle='-', color=gp.meshpoints_color_list[meshpoint_indx], alpha=1, linewidth=2.5, label=gp.meshpoints_label_list[meshpoint_indx])
            plt.legend(loc='best')
            plt.grid(True, alpha=0.25)
            save_current_figure(self.savedir, r'$u_2$', 30, r'$\lambda_c$', 30, "fenics-weak-form-uniaxial-rate-independent-u_2-vs-lmbda_c")

        # lmbda_c_eq
        if ppp.save_lmbda_c_eq_chunks:
            for meshpoint_indx in range(len(gp.meshpoints)):
                fig = plt.figure()
                lmbda_c_eq___meshpoint_chunk = [lmbda_c_eq_chunk[meshpoint_indx] for lmbda_c_eq_chunk in weak_form_chunks.lmbda_c_eq_chunks]
                for nu_chunk_indx in range(len(mp.nu_chunks_list)):
                    lmbda_c_eq___nu_chunk = [lmbda_c_eq_chunk[nu_chunk_indx] for lmbda_c_eq_chunk in lmbda_c_eq___meshpoint_chunk]
                    plt.plot(deformation.t_chunks, lmbda_c_eq___nu_chunk, linestyle='-', color=mp.nu_chunks_color_list[nu_chunk_indx], alpha=1, linewidth=2.5, label=mp.nu_chunks_label_list[nu_chunk_indx])
                plt.legend(loc='best')
                plt.grid(True, alpha=0.25)
                save_current_figure(self.savedir, r'$t$', 30, r'$\lambda_c^{eq}$', 30, "fenics-weak-form-uniaxial-rate-independent-t-vs-lmbda_c_eq"+"_"+gp.meshpoints_name_list[meshpoint_indx])
            
            for meshpoint_indx in range(len(gp.meshpoints)):
                fig = plt.figure()
                lmbda_c_eq___meshpoint_chunk = [lmbda_c_eq_chunk[meshpoint_indx] for lmbda_c_eq_chunk in weak_form_chunks.lmbda_c_eq_chunks]
                for nu_chunk_indx in range(len(mp.nu_chunks_list)):
                    lmbda_c_eq___nu_chunk = [lmbda_c_eq_chunk[nu_chunk_indx] for lmbda_c_eq_chunk in lmbda_c_eq___meshpoint_chunk]
                    plt.plot(deformation.u_2_chunks, lmbda_c_eq___nu_chunk, linestyle='-', color=mp.nu_chunks_color_list[nu_chunk_indx], alpha=1, linewidth=2.5, label=mp.nu_chunks_label_list[nu_chunk_indx])
                plt.legend(loc='best')
                plt.grid(True, alpha=0.25)
                save_current_figure(self.savedir, r'$u_2$', 30, r'$\lambda_c^{eq}$', 30, "fenics-weak-form-uniaxial-rate-independent-u_2-vs-lmbda_c_eq"+"_"+gp.meshpoints_name_list[meshpoint_indx])
        
        # lmbda_nu
        if ppp.save_lmbda_nu_chunks:
            for meshpoint_indx in range(len(gp.meshpoints)):
                fig = plt.figure()
                lmbda_nu___meshpoint_chunk = [lmbda_nu_chunk[meshpoint_indx] for lmbda_nu_chunk in weak_form_chunks.lmbda_nu_chunks]
                for nu_chunk_indx in range(len(mp.nu_chunks_list)):
                    lmbda_nu___nu_chunk = [lmbda_nu_chunk[nu_chunk_indx] for lmbda_nu_chunk in lmbda_nu___meshpoint_chunk]
                    plt.plot(deformation.t_chunks, lmbda_nu___nu_chunk, linestyle='-', color=mp.nu_chunks_color_list[nu_chunk_indx], alpha=1, linewidth=2.5, label=mp.nu_chunks_label_list[nu_chunk_indx])
                plt.legend(loc='best')
                plt.grid(True, alpha=0.25)
                save_current_figure(self.savedir, r'$t$', 30, r'$\lambda_{\nu}$', 30, "fenics-weak-form-uniaxial-rate-independent-t-vs-lmbda_nu"+"_"+gp.meshpoints_name_list[meshpoint_indx])
            
            for meshpoint_indx in range(len(gp.meshpoints)):
                fig = plt.figure()
                lmbda_nu___meshpoint_chunk = [lmbda_nu_chunk[meshpoint_indx] for lmbda_nu_chunk in weak_form_chunks.lmbda_nu_chunks]
                for nu_chunk_indx in range(len(mp.nu_chunks_list)):
                    lmbda_nu___nu_chunk = [lmbda_nu_chunk[nu_chunk_indx] for lmbda_nu_chunk in lmbda_nu___meshpoint_chunk]
                    plt.plot(deformation.u_2_chunks, lmbda_nu___nu_chunk, linestyle='-', color=mp.nu_chunks_color_list[nu_chunk_indx], alpha=1, linewidth=2.5, label=mp.nu_chunks_label_list[nu_chunk_indx])
                plt.legend(loc='best')
                plt.grid(True, alpha=0.25)
                save_current_figure(self.savedir, r'$u_2$', 30, r'$\lambda_{\nu}$', 30, "fenics-weak-form-uniaxial-rate-independent-u_2-vs-lmbda_nu"+"_"+gp.meshpoints_name_list[meshpoint_indx])
        
        # lmbda_c_tilde
        if ppp.save_lmbda_c_tilde_chunks:
            fig = plt.figure()
            for meshpoint_indx in range(len(gp.meshpoints)):
                lmbda_c_tilde___meshpoint_chunk = [lmbda_c_tilde_chunk[meshpoint_indx] for lmbda_c_tilde_chunk in weak_form_chunks.lmbda_c_tilde_chunks]
                plt.plot(deformation.t_chunks, lmbda_c_tilde___meshpoint_chunk, linestyle='-', color=gp.meshpoints_color_list[meshpoint_indx], alpha=1, linewidth=2.5, label=gp.meshpoints_label_list[meshpoint_indx])
            plt.legend(loc='best')
            plt.grid(True, alpha=0.25)
            save_current_figure(self.savedir, r'$t$', 30, r'$\tilde{\lambda}_c$', 30, "fenics-weak-form-uniaxial-rate-independent-t-vs-lmbda_c_tilde")

            fig = plt.figure()
            for meshpoint_indx in range(len(gp.meshpoints)):
                lmbda_c_tilde___meshpoint_chunk = [lmbda_c_tilde_chunk[meshpoint_indx] for lmbda_c_tilde_chunk in weak_form_chunks.lmbda_c_tilde_chunks]
                plt.plot(deformation.u_2_chunks, lmbda_c_tilde___meshpoint_chunk, linestyle='-', color=gp.meshpoints_color_list[meshpoint_indx], alpha=1, linewidth=2.5, label=gp.meshpoints_label_list[meshpoint_indx])
            plt.legend(loc='best')
            plt.grid(True, alpha=0.25)
            save_current_figure(self.savedir, r'$u_2$', 30, r'$\tilde{\lambda}_c$', 30, "fenics-weak-form-uniaxial-rate-independent-u_2-vs-lmbda_c_tilde")

        # lmbda_c_eq_tilde
        if ppp.save_lmbda_c_eq_tilde_chunks:
            for meshpoint_indx in range(len(gp.meshpoints)):
                fig = plt.figure()
                lmbda_c_eq_tilde___meshpoint_chunk = [lmbda_c_eq_tilde_chunk[meshpoint_indx] for lmbda_c_eq_tilde_chunk in weak_form_chunks.lmbda_c_eq_tilde_chunks]
                for nu_chunk_indx in range(len(mp.nu_chunks_list)):
                    lmbda_c_eq_tilde___nu_chunk = [lmbda_c_eq_tilde_chunk[nu_chunk_indx] for lmbda_c_eq_tilde_chunk in lmbda_c_eq_tilde___meshpoint_chunk]
                    plt.plot(deformation.t_chunks, lmbda_c_eq_tilde___nu_chunk, linestyle='-', color=mp.nu_chunks_color_list[nu_chunk_indx], alpha=1, linewidth=2.5, label=mp.nu_chunks_label_list[nu_chunk_indx])
                plt.legend(loc='best')
                plt.grid(True, alpha=0.25)
                save_current_figure(self.savedir, r'$t$', 30, r'$\tilde{\lambda}_c^{eq}$', 30, "fenics-weak-form-uniaxial-rate-independent-t-vs-lmbda_c_eq_tilde"+"_"+gp.meshpoints_name_list[meshpoint_indx])
            
            for meshpoint_indx in range(len(gp.meshpoints)):
                fig = plt.figure()
                lmbda_c_eq_tilde___meshpoint_chunk = [lmbda_c_eq_tilde_chunk[meshpoint_indx] for lmbda_c_eq_tilde_chunk in weak_form_chunks.lmbda_c_eq_tilde_chunks]
                for nu_chunk_indx in range(len(mp.nu_chunks_list)):
                    lmbda_c_eq_tilde___nu_chunk = [lmbda_c_eq_tilde_chunk[nu_chunk_indx] for lmbda_c_eq_tilde_chunk in lmbda_c_eq_tilde___meshpoint_chunk]
                    plt.plot(deformation.u_2_chunks, lmbda_c_eq_tilde___nu_chunk, linestyle='-', color=mp.nu_chunks_color_list[nu_chunk_indx], alpha=1, linewidth=2.5, label=mp.nu_chunks_label_list[nu_chunk_indx])
                plt.legend(loc='best')
                plt.grid(True, alpha=0.25)
                save_current_figure(self.savedir, r'$u_2$', 30, r'$\tilde{\lambda}_c^{eq}$', 30, "fenics-weak-form-uniaxial-rate-independent-u_2-vs-lmbda_c_eq_tilde"+"_"+gp.meshpoints_name_list[meshpoint_indx])
        
        # lmbda_nu_tilde
        if ppp.save_lmbda_nu_tilde_chunks:
            for meshpoint_indx in range(len(gp.meshpoints)):
                fig = plt.figure()
                lmbda_nu_tilde___meshpoint_chunk = [lmbda_nu_tilde_chunk[meshpoint_indx] for lmbda_nu_tilde_chunk in weak_form_chunks.lmbda_nu_tilde_chunks]
                for nu_chunk_indx in range(len(mp.nu_chunks_list)):
                    lmbda_nu_tilde___nu_chunk = [lmbda_nu_tilde_chunk[nu_chunk_indx] for lmbda_nu_tilde_chunk in lmbda_nu_tilde___meshpoint_chunk]
                    plt.plot(deformation.t_chunks, lmbda_nu_tilde___nu_chunk, linestyle='-', color=mp.nu_chunks_color_list[nu_chunk_indx], alpha=1, linewidth=2.5, label=mp.nu_chunks_label_list[nu_chunk_indx])
                plt.legend(loc='best')
                plt.grid(True, alpha=0.25)
                save_current_figure(self.savedir, r'$t$', 30, r'$\tilde{\lambda}_{\nu}$', 30, "fenics-weak-form-uniaxial-rate-independent-t-vs-lmbda_nu_tilde"+"_"+gp.meshpoints_name_list[meshpoint_indx])
            
            for meshpoint_indx in range(len(gp.meshpoints)):
                fig = plt.figure()
                lmbda_nu_tilde___meshpoint_chunk = [lmbda_nu_tilde_chunk[meshpoint_indx] for lmbda_nu_tilde_chunk in weak_form_chunks.lmbda_nu_tilde_chunks]
                for nu_chunk_indx in range(len(mp.nu_chunks_list)):
                    lmbda_nu_tilde___nu_chunk = [lmbda_nu_tilde_chunk[nu_chunk_indx] for lmbda_nu_tilde_chunk in lmbda_nu_tilde___meshpoint_chunk]
                    plt.plot(deformation.u_2_chunks, lmbda_nu_tilde___nu_chunk, linestyle='-', color=mp.nu_chunks_color_list[nu_chunk_indx], alpha=1, linewidth=2.5, label=mp.nu_chunks_label_list[nu_chunk_indx])
                plt.legend(loc='best')
                plt.grid(True, alpha=0.25)
                save_current_figure(self.savedir, r'$u_2$', 30, r'$\tilde{\lambda}_{\nu}$', 30, "fenics-weak-form-uniaxial-rate-independent-u_2-vs-lmbda_nu_tilde"+"_"+gp.meshpoints_name_list[meshpoint_indx])
        
        # lmbda_c_tilde_max
        if ppp.save_lmbda_c_tilde_max_chunks:
            fig = plt.figure()
            for meshpoint_indx in range(len(gp.meshpoints)):
                lmbda_c_tilde_max___meshpoint_chunk = [lmbda_c_tilde_max_chunk[meshpoint_indx] for lmbda_c_tilde_max_chunk in weak_form_chunks.lmbda_c_tilde_max_chunks]
                plt.plot(deformation.t_chunks, lmbda_c_tilde_max___meshpoint_chunk, linestyle='-', color=gp.meshpoints_color_list[meshpoint_indx], alpha=1, linewidth=2.5, label=gp.meshpoints_label_list[meshpoint_indx])
            plt.legend(loc='best')
            plt.grid(True, alpha=0.25)
            save_current_figure(self.savedir, r'$t$', 30, r'$\tilde{\lambda}_c^{max}$', 30, "fenics-weak-form-uniaxial-rate-independent-t-vs-lmbda_c_tilde_max")

            fig = plt.figure()
            for meshpoint_indx in range(len(gp.meshpoints)):
                lmbda_c_tilde_max___meshpoint_chunk = [lmbda_c_tilde_max_chunk[meshpoint_indx] for lmbda_c_tilde_max_chunk in weak_form_chunks.lmbda_c_tilde_max_chunks]
                plt.plot(deformation.u_2_chunks, lmbda_c_tilde_max___meshpoint_chunk, linestyle='-', color=gp.meshpoints_color_list[meshpoint_indx], alpha=1, linewidth=2.5, label=gp.meshpoints_label_list[meshpoint_indx])
            plt.legend(loc='best')
            plt.grid(True, alpha=0.25)
            save_current_figure(self.savedir, r'$u_2$', 30, r'$\tilde{\lambda}_c^{max}$', 30, "fenics-weak-form-uniaxial-rate-independent-u_2-vs-lmbda_c_tilde_max")

        # lmbda_c_eq_tilde_max
        if ppp.save_lmbda_c_eq_tilde_max_chunks:
            for meshpoint_indx in range(len(gp.meshpoints)):
                fig = plt.figure()
                lmbda_c_eq_tilde_max___meshpoint_chunk = [lmbda_c_eq_tilde_max_chunk[meshpoint_indx] for lmbda_c_eq_tilde_max_chunk in weak_form_chunks.lmbda_c_eq_tilde_max_chunks]
                for nu_chunk_indx in range(len(mp.nu_chunks_list)):
                    lmbda_c_eq_tilde_max___nu_chunk = [lmbda_c_eq_tilde_max_chunk[nu_chunk_indx] for lmbda_c_eq_tilde_max_chunk in lmbda_c_eq_tilde_max___meshpoint_chunk]
                    plt.plot(deformation.t_chunks, lmbda_c_eq_tilde_max___nu_chunk, linestyle='-', color=mp.nu_chunks_color_list[nu_chunk_indx], alpha=1, linewidth=2.5, label=mp.nu_chunks_label_list[nu_chunk_indx])
                plt.legend(loc='best')
                plt.grid(True, alpha=0.25)
                save_current_figure(self.savedir, r'$t$', 30, r'$(\tilde{\lambda}_c^{eq})^{max}$', 30, "fenics-weak-form-uniaxial-rate-independent-t-vs-lmbda_c_eq_tilde_max"+"_"+gp.meshpoints_name_list[meshpoint_indx])
            
            for meshpoint_indx in range(len(gp.meshpoints)):
                fig = plt.figure()
                lmbda_c_eq_tilde_max___meshpoint_chunk = [lmbda_c_eq_tilde_max_chunk[meshpoint_indx] for lmbda_c_eq_tilde_max_chunk in weak_form_chunks.lmbda_c_eq_tilde_max_chunks]
                for nu_chunk_indx in range(len(mp.nu_chunks_list)):
                    lmbda_c_eq_tilde_max___nu_chunk = [lmbda_c_eq_tilde_max_chunk[nu_chunk_indx] for lmbda_c_eq_tilde_max_chunk in lmbda_c_eq_tilde_max___meshpoint_chunk]
                    plt.plot(deformation.u_2_chunks, lmbda_c_eq_tilde_max___nu_chunk, linestyle='-', color=mp.nu_chunks_color_list[nu_chunk_indx], alpha=1, linewidth=2.5, label=mp.nu_chunks_label_list[nu_chunk_indx])
                plt.legend(loc='best')
                plt.grid(True, alpha=0.25)
                save_current_figure(self.savedir, r'$u_2$', 30, r'$(\tilde{\lambda}_c^{eq})^{max}$', 30, "fenics-weak-form-uniaxial-rate-independent-u_2-vs-lmbda_c_eq_tilde_max"+"_"+gp.meshpoints_name_list[meshpoint_indx])
        
        # lmbda_nu_tilde_max
        if ppp.save_lmbda_nu_tilde_max_chunks:
            for meshpoint_indx in range(len(gp.meshpoints)):
                fig = plt.figure()
                lmbda_nu_tilde_max___meshpoint_chunk = [lmbda_nu_tilde_max_chunk[meshpoint_indx] for lmbda_nu_tilde_max_chunk in weak_form_chunks.lmbda_nu_tilde_max_chunks]
                for nu_chunk_indx in range(len(mp.nu_chunks_list)):
                    lmbda_nu_tilde_max___nu_chunk = [lmbda_nu_tilde_max_chunk[nu_chunk_indx] for lmbda_nu_tilde_max_chunk in lmbda_nu_tilde_max___meshpoint_chunk]
                    plt.plot(deformation.t_chunks, lmbda_nu_tilde_max___nu_chunk, linestyle='-', color=mp.nu_chunks_color_list[nu_chunk_indx], alpha=1, linewidth=2.5, label=mp.nu_chunks_label_list[nu_chunk_indx])
                plt.legend(loc='best')
                plt.grid(True, alpha=0.25)
                save_current_figure(self.savedir, r'$t$', 30, r'$\tilde{\lambda}_{\nu}^{max}$', 30, "fenics-weak-form-uniaxial-rate-independent-t-vs-lmbda_nu_tilde_max"+"_"+gp.meshpoints_name_list[meshpoint_indx])
            
            for meshpoint_indx in range(len(gp.meshpoints)):
                fig = plt.figure()
                lmbda_nu_tilde_max___meshpoint_chunk = [lmbda_nu_tilde_max_chunk[meshpoint_indx] for lmbda_nu_tilde_max_chunk in weak_form_chunks.lmbda_nu_tilde_max_chunks]
                for nu_chunk_indx in range(len(mp.nu_chunks_list)):
                    lmbda_nu_tilde_max___nu_chunk = [lmbda_nu_tilde_max_chunk[nu_chunk_indx] for lmbda_nu_tilde_max_chunk in lmbda_nu_tilde_max___meshpoint_chunk]
                    plt.plot(deformation.u_2_chunks, lmbda_nu_tilde_max___nu_chunk, linestyle='-', color=mp.nu_chunks_color_list[nu_chunk_indx], alpha=1, linewidth=2.5, label=mp.nu_chunks_label_list[nu_chunk_indx])
                plt.legend(loc='best')
                plt.grid(True, alpha=0.25)
                save_current_figure(self.savedir, r'$u_2$', 30, r'$\tilde{\lambda}_{\nu}^{max}$', 30, "fenics-weak-form-uniaxial-rate-independent-u_2-vs-lmbda_nu_tilde_max"+"_"+gp.meshpoints_name_list[meshpoint_indx])
        
        # upsilon_c
        if ppp.save_upsilon_c_chunks:
            for meshpoint_indx in range(len(gp.meshpoints)):
                fig = plt.figure()
                upsilon_c___meshpoint_chunk = [upsilon_c_chunk[meshpoint_indx] for upsilon_c_chunk in weak_form_chunks.upsilon_c_chunks]
                for nu_chunk_indx in range(len(mp.nu_chunks_list)):
                    upsilon_c___nu_chunk = [upsilon_c_chunk[nu_chunk_indx] for upsilon_c_chunk in upsilon_c___meshpoint_chunk]
                    plt.plot(deformation.t_chunks, upsilon_c___nu_chunk, linestyle='-', color=mp.nu_chunks_color_list[nu_chunk_indx], alpha=1, linewidth=2.5, label=mp.nu_chunks_label_list[nu_chunk_indx])
                plt.legend(loc='best')
                plt.grid(True, alpha=0.25)
                save_current_figure(self.savedir, r'$t$', 30, r'$\upsilon_c$', 30, "fenics-weak-form-uniaxial-rate-independent-t-vs-upsilon_c"+"_"+gp.meshpoints_name_list[meshpoint_indx])
            
            for meshpoint_indx in range(len(gp.meshpoints)):
                fig = plt.figure()
                upsilon_c___meshpoint_chunk = [upsilon_c_chunk[meshpoint_indx] for upsilon_c_chunk in weak_form_chunks.upsilon_c_chunks]
                for nu_chunk_indx in range(len(mp.nu_chunks_list)):
                    upsilon_c___nu_chunk = [upsilon_c_chunk[nu_chunk_indx] for upsilon_c_chunk in upsilon_c___meshpoint_chunk]
                    plt.plot(deformation.u_2_chunks, upsilon_c___nu_chunk, linestyle='-', color=mp.nu_chunks_color_list[nu_chunk_indx], alpha=1, linewidth=2.5, label=mp.nu_chunks_label_list[nu_chunk_indx])
                plt.legend(loc='best')
                plt.grid(True, alpha=0.25)
                save_current_figure(self.savedir, r'$u_2$', 30, r'$\upsilon_c$', 30, "fenics-weak-form-uniaxial-rate-independent-u_2-vs-upsilon_c"+"_"+gp.meshpoints_name_list[meshpoint_indx])
        
        # Upsilon_c
        if ppp.save_Upsilon_c_chunks:
            fig = plt.figure()
            for meshpoint_indx in range(len(gp.meshpoints)):
                Upsilon_c___meshpoint_chunk = [Upsilon_c_chunk[meshpoint_indx] for Upsilon_c_chunk in weak_form_chunks.Upsilon_c_chunks]
                plt.plot(deformation.t_chunks, Upsilon_c___meshpoint_chunk, linestyle='-', color=gp.meshpoints_color_list[meshpoint_indx], alpha=1, linewidth=2.5, label=gp.meshpoints_label_list[meshpoint_indx])
            plt.legend(loc='best')
            plt.grid(True, alpha=0.25)
            save_current_figure(self.savedir, r'$t$', 30, r'$\Upsilon_c$', 30, "fenics-weak-form-uniaxial-rate-independent-t-vs-Upsilon_c")

            fig = plt.figure()
            for meshpoint_indx in range(len(gp.meshpoints)):
                Upsilon_c___meshpoint_chunk = [Upsilon_c_chunk[meshpoint_indx] for Upsilon_c_chunk in weak_form_chunks.Upsilon_c_chunks]
                plt.plot(deformation.u_2_chunks, Upsilon_c___meshpoint_chunk, linestyle='-', color=gp.meshpoints_color_list[meshpoint_indx], alpha=1, linewidth=2.5, label=gp.meshpoints_label_list[meshpoint_indx])
            plt.legend(loc='best')
            plt.grid(True, alpha=0.25)
            save_current_figure(self.savedir, r'$u_2$', 30, r'$\Upsilon_c$', 30, "fenics-weak-form-uniaxial-rate-independent-u_2-vs-Upsilon_c")

        # d_c
        if ppp.save_d_c_chunks:
            for meshpoint_indx in range(len(gp.meshpoints)):
                fig = plt.figure()
                d_c___meshpoint_chunk = [d_c_chunk[meshpoint_indx] for d_c_chunk in weak_form_chunks.d_c_chunks]
                for nu_chunk_indx in range(len(mp.nu_chunks_list)):
                    d_c___nu_chunk = [d_c_chunk[nu_chunk_indx] for d_c_chunk in d_c___meshpoint_chunk]
                    plt.plot(deformation.t_chunks, d_c___nu_chunk, linestyle='-', color=mp.nu_chunks_color_list[nu_chunk_indx], alpha=1, linewidth=2.5, label=mp.nu_chunks_label_list[nu_chunk_indx])
                plt.legend(loc='best')
                plt.grid(True, alpha=0.25)
                save_current_figure(self.savedir, r'$t$', 30, r'$d_c$', 30, "fenics-weak-form-uniaxial-rate-independent-t-vs-d_c"+"_"+gp.meshpoints_name_list[meshpoint_indx])
            
            for meshpoint_indx in range(len(gp.meshpoints)):
                fig = plt.figure()
                d_c___meshpoint_chunk = [d_c_chunk[meshpoint_indx] for d_c_chunk in weak_form_chunks.d_c_chunks]
                for nu_chunk_indx in range(len(mp.nu_chunks_list)):
                    d_c___nu_chunk = [d_c_chunk[nu_chunk_indx] for d_c_chunk in d_c___meshpoint_chunk]
                    plt.plot(deformation.u_2_chunks, d_c___nu_chunk, linestyle='-', color=mp.nu_chunks_color_list[nu_chunk_indx], alpha=1, linewidth=2.5, label=mp.nu_chunks_label_list[nu_chunk_indx])
                plt.legend(loc='best')
                plt.grid(True, alpha=0.25)
                save_current_figure(self.savedir, r'$u_2$', 30, r'$d_c$', 30, "fenics-weak-form-uniaxial-rate-independent-u_2-vs-d_c"+"_"+gp.meshpoints_name_list[meshpoint_indx])
        
        # D_c
        if ppp.save_D_c_chunks:
            fig = plt.figure()
            for meshpoint_indx in range(len(gp.meshpoints)):
                D_c___meshpoint_chunk = [D_c_chunk[meshpoint_indx] for D_c_chunk in weak_form_chunks.D_c_chunks]
                plt.plot(deformation.t_chunks, D_c___meshpoint_chunk, linestyle='-', color=gp.meshpoints_color_list[meshpoint_indx], alpha=1, linewidth=2.5, label=gp.meshpoints_label_list[meshpoint_indx])
            plt.legend(loc='best')
            plt.grid(True, alpha=0.25)
            save_current_figure(self.savedir, r'$t$', 30, r'$D_c$', 30, "fenics-weak-form-uniaxial-rate-independent-t-vs-D_c")

            fig = plt.figure()
            for meshpoint_indx in range(len(gp.meshpoints)):
                D_c___meshpoint_chunk = [D_c_chunk[meshpoint_indx] for D_c_chunk in weak_form_chunks.D_c_chunks]
                plt.plot(deformation.u_2_chunks, D_c___meshpoint_chunk, linestyle='-', color=gp.meshpoints_color_list[meshpoint_indx], alpha=1, linewidth=2.5, label=gp.meshpoints_label_list[meshpoint_indx])
            plt.legend(loc='best')
            plt.grid(True, alpha=0.25)
            save_current_figure(self.savedir, r'$u_2$', 30, r'$D_c$', 30, "fenics-weak-form-uniaxial-rate-independent-u_2-vs-D_c")
        
        # epsilon_cnu_diss_hat
        if ppp.save_epsilon_cnu_diss_hat_chunks:
            for meshpoint_indx in range(len(gp.meshpoints)):
                fig = plt.figure()
                epsilon_cnu_diss_hat___meshpoint_chunk = [epsilon_cnu_diss_hat_chunk[meshpoint_indx] for epsilon_cnu_diss_hat_chunk in weak_form_chunks.epsilon_cnu_diss_hat_chunks]
                for nu_chunk_indx in range(len(mp.nu_chunks_list)):
                    epsilon_cnu_diss_hat___nu_chunk = [epsilon_cnu_diss_hat_chunk[nu_chunk_indx] for epsilon_cnu_diss_hat_chunk in epsilon_cnu_diss_hat___meshpoint_chunk]
                    plt.plot(deformation.t_chunks, epsilon_cnu_diss_hat___nu_chunk, linestyle='-', color=mp.nu_chunks_color_list[nu_chunk_indx], alpha=1, linewidth=2.5, label=mp.nu_chunks_label_list[nu_chunk_indx])
                plt.legend(loc='best')
                plt.grid(True, alpha=0.25)
                save_current_figure(self.savedir, r'$t$', 30, r'$\hat{\varepsilon}_{c\nu}^{diss}$', 30, "fenics-weak-form-uniaxial-rate-independent-t-vs-epsilon_cnu_diss_hat"+"_"+gp.meshpoints_name_list[meshpoint_indx])
            
            for meshpoint_indx in range(len(gp.meshpoints)):
                fig = plt.figure()
                epsilon_cnu_diss_hat___meshpoint_chunk = [epsilon_cnu_diss_hat_chunk[meshpoint_indx] for epsilon_cnu_diss_hat_chunk in weak_form_chunks.epsilon_cnu_diss_hat_chunks]
                for nu_chunk_indx in range(len(mp.nu_chunks_list)):
                    epsilon_cnu_diss_hat___nu_chunk = [epsilon_cnu_diss_hat_chunk[nu_chunk_indx] for epsilon_cnu_diss_hat_chunk in epsilon_cnu_diss_hat___meshpoint_chunk]
                    plt.plot(deformation.u_2_chunks, epsilon_cnu_diss_hat___nu_chunk, linestyle='-', color=mp.nu_chunks_color_list[nu_chunk_indx], alpha=1, linewidth=2.5, label=mp.nu_chunks_label_list[nu_chunk_indx])
                plt.legend(loc='best')
                plt.grid(True, alpha=0.25)
                save_current_figure(self.savedir, r'$u_2$', 30, r'$\hat{\varepsilon}_{c\nu}^{diss}$', 30, "fenics-weak-form-uniaxial-rate-independent-u_2-vs-epsilon_cnu_diss_hat"+"_"+gp.meshpoints_name_list[meshpoint_indx])
        
        # Epsilon_cnu_diss_hat
        if ppp.save_Epsilon_cnu_diss_hat_chunks:
            fig = plt.figure()
            for meshpoint_indx in range(len(gp.meshpoints)):
                Epsilon_cnu_diss_hat___meshpoint_chunk = [Epsilon_cnu_diss_hat_chunk[meshpoint_indx] for Epsilon_cnu_diss_hat_chunk in weak_form_chunks.Epsilon_cnu_diss_hat_chunks]
                plt.plot(deformation.t_chunks, Epsilon_cnu_diss_hat___meshpoint_chunk, linestyle='-', color=gp.meshpoints_color_list[meshpoint_indx], alpha=1, linewidth=2.5, label=gp.meshpoints_label_list[meshpoint_indx])
            plt.legend(loc='best')
            plt.grid(True, alpha=0.25)
            save_current_figure(self.savedir, r'$t$', 30, r'$\hat{E}_{c\nu}^{diss}$', 30, "fenics-weak-form-uniaxial-rate-independent-t-vs-Epsilon_cnu_diss_hat")

            fig = plt.figure()
            for meshpoint_indx in range(len(gp.meshpoints)):
                Epsilon_cnu_diss_hat___meshpoint_chunk = [Epsilon_cnu_diss_hat_chunk[meshpoint_indx] for Epsilon_cnu_diss_hat_chunk in weak_form_chunks.Epsilon_cnu_diss_hat_chunks]
                plt.plot(deformation.u_2_chunks, Epsilon_cnu_diss_hat___meshpoint_chunk, linestyle='-', color=gp.meshpoints_color_list[meshpoint_indx], alpha=1, linewidth=2.5, label=gp.meshpoints_label_list[meshpoint_indx])
            plt.legend(loc='best')
            plt.grid(True, alpha=0.25)
            save_current_figure(self.savedir, r'$u_2$', 30, r'$\hat{E}_{c\nu}^{diss}$', 30, "fenics-weak-form-uniaxial-rate-independent-u_2-vs-Epsilon_cnu_diss_hat")
        
        # epsilon_c_diss_hat
        if ppp.save_epsilon_c_diss_hat_chunks:
            for meshpoint_indx in range(len(gp.meshpoints)):
                fig = plt.figure()
                epsilon_c_diss_hat___meshpoint_chunk = [epsilon_c_diss_hat_chunk[meshpoint_indx] for epsilon_c_diss_hat_chunk in weak_form_chunks.epsilon_c_diss_hat_chunks]
                for nu_chunk_indx in range(len(mp.nu_chunks_list)):
                    epsilon_c_diss_hat___nu_chunk = [epsilon_c_diss_hat_chunk[nu_chunk_indx] for epsilon_c_diss_hat_chunk in epsilon_c_diss_hat___meshpoint_chunk]
                    plt.plot(deformation.t_chunks, epsilon_c_diss_hat___nu_chunk, linestyle='-', color=mp.nu_chunks_color_list[nu_chunk_indx], alpha=1, linewidth=2.5, label=mp.nu_chunks_label_list[nu_chunk_indx])
                plt.legend(loc='best')
                plt.grid(True, alpha=0.25)
                save_current_figure(self.savedir, r'$t$', 30, r'$\hat{\varepsilon}_c^{diss}$', 30, "fenics-weak-form-uniaxial-rate-independent-t-vs-epsilon_c_diss_hat"+"_"+gp.meshpoints_name_list[meshpoint_indx])
            
            for meshpoint_indx in range(len(gp.meshpoints)):
                fig = plt.figure()
                epsilon_c_diss_hat___meshpoint_chunk = [epsilon_c_diss_hat_chunk[meshpoint_indx] for epsilon_c_diss_hat_chunk in weak_form_chunks.epsilon_c_diss_hat_chunks]
                for nu_chunk_indx in range(len(mp.nu_chunks_list)):
                    epsilon_c_diss_hat___nu_chunk = [epsilon_c_diss_hat_chunk[nu_chunk_indx] for epsilon_c_diss_hat_chunk in epsilon_c_diss_hat___meshpoint_chunk]
                    plt.plot(deformation.u_2_chunks, epsilon_c_diss_hat___nu_chunk, linestyle='-', color=mp.nu_chunks_color_list[nu_chunk_indx], alpha=1, linewidth=2.5, label=mp.nu_chunks_label_list[nu_chunk_indx])
                plt.legend(loc='best')
                plt.grid(True, alpha=0.25)
                save_current_figure(self.savedir, r'$u_2$', 30, r'$\hat{\varepsilon}_c^{diss}$', 30, "fenics-weak-form-uniaxial-rate-independent-u_2-vs-epsilon_c_diss_hat"+"_"+gp.meshpoints_name_list[meshpoint_indx])
        
        # Epsilon_c_diss_hat
        if ppp.save_Epsilon_c_diss_hat_chunks:
            fig = plt.figure()
            for meshpoint_indx in range(len(gp.meshpoints)):
                Epsilon_c_diss_hat___meshpoint_chunk = [Epsilon_c_diss_hat_chunk[meshpoint_indx] for Epsilon_c_diss_hat_chunk in weak_form_chunks.Epsilon_c_diss_hat_chunks]
                plt.plot(deformation.t_chunks, Epsilon_c_diss_hat___meshpoint_chunk, linestyle='-', color=gp.meshpoints_color_list[meshpoint_indx], alpha=1, linewidth=2.5, label=gp.meshpoints_label_list[meshpoint_indx])
            plt.legend(loc='best')
            plt.grid(True, alpha=0.25)
            save_current_figure(self.savedir, r'$t$', 30, r'$\hat{E}_c^{diss}$', 30, "fenics-weak-form-uniaxial-rate-independent-t-vs-Epsilon_c_diss_hat")

            fig = plt.figure()
            for meshpoint_indx in range(len(gp.meshpoints)):
                Epsilon_c_diss_hat___meshpoint_chunk = [Epsilon_c_diss_hat_chunk[meshpoint_indx] for Epsilon_c_diss_hat_chunk in weak_form_chunks.Epsilon_c_diss_hat_chunks]
                plt.plot(deformation.u_2_chunks, Epsilon_c_diss_hat___meshpoint_chunk, linestyle='-', color=gp.meshpoints_color_list[meshpoint_indx], alpha=1, linewidth=2.5, label=gp.meshpoints_label_list[meshpoint_indx])
            plt.legend(loc='best')
            plt.grid(True, alpha=0.25)
            save_current_figure(self.savedir, r'$u_2$', 30, r'$\hat{E}_c^{diss}$', 30, "fenics-weak-form-uniaxial-rate-independent-u_2-vs-Epsilon_c_diss_hat")
        
        # overline_epsilon_cnu_diss_hat
        if ppp.save_overline_epsilon_cnu_diss_hat_chunks:
            for meshpoint_indx in range(len(gp.meshpoints)):
                fig = plt.figure()
                overline_epsilon_cnu_diss_hat___meshpoint_chunk = [overline_epsilon_cnu_diss_hat_chunk[meshpoint_indx] for overline_epsilon_cnu_diss_hat_chunk in weak_form_chunks.overline_epsilon_cnu_diss_hat_chunks]
                for nu_chunk_indx in range(len(mp.nu_chunks_list)):
                    overline_epsilon_cnu_diss_hat___nu_chunk = [overline_epsilon_cnu_diss_hat_chunk[nu_chunk_indx] for overline_epsilon_cnu_diss_hat_chunk in overline_epsilon_cnu_diss_hat___meshpoint_chunk]
                    plt.plot(deformation.t_chunks, overline_epsilon_cnu_diss_hat___nu_chunk, linestyle='-', color=mp.nu_chunks_color_list[nu_chunk_indx], alpha=1, linewidth=2.5, label=mp.nu_chunks_label_list[nu_chunk_indx])
                plt.legend(loc='best')
                plt.grid(True, alpha=0.25)
                save_current_figure(self.savedir, r'$t$', 30, r'$\overline{\hat{\varepsilon}_{c\nu}^{diss}}$', 30, "fenics-weak-form-uniaxial-rate-independent-t-vs-overline_epsilon_cnu_diss_hat"+"_"+gp.meshpoints_name_list[meshpoint_indx])
            
            for meshpoint_indx in range(len(gp.meshpoints)):
                fig = plt.figure()
                overline_epsilon_cnu_diss_hat___meshpoint_chunk = [overline_epsilon_cnu_diss_hat_chunk[meshpoint_indx] for overline_epsilon_cnu_diss_hat_chunk in weak_form_chunks.overline_epsilon_cnu_diss_hat_chunks]
                for nu_chunk_indx in range(len(mp.nu_chunks_list)):
                    overline_epsilon_cnu_diss_hat___nu_chunk = [overline_epsilon_cnu_diss_hat_chunk[nu_chunk_indx] for overline_epsilon_cnu_diss_hat_chunk in overline_epsilon_cnu_diss_hat___meshpoint_chunk]
                    plt.plot(deformation.u_2_chunks, overline_epsilon_cnu_diss_hat___nu_chunk, linestyle='-', color=mp.nu_chunks_color_list[nu_chunk_indx], alpha=1, linewidth=2.5, label=mp.nu_chunks_label_list[nu_chunk_indx])
                plt.legend(loc='best')
                plt.grid(True, alpha=0.25)
                save_current_figure(self.savedir, r'$u_2$', 30, r'$\overline{\hat{\varepsilon}_{c\nu}^{diss}}$', 30, "fenics-weak-form-uniaxial-rate-independent-u_2-vs-overline_epsilon_cnu_diss_hat"+"_"+gp.meshpoints_name_list[meshpoint_indx])
        
        # overline_Epsilon_cnu_diss_hat
        if ppp.save_overline_Epsilon_cnu_diss_hat_chunks:
            fig = plt.figure()
            for meshpoint_indx in range(len(gp.meshpoints)):
                overline_Epsilon_cnu_diss_hat___meshpoint_chunk = [overline_Epsilon_cnu_diss_hat_chunk[meshpoint_indx] for overline_Epsilon_cnu_diss_hat_chunk in weak_form_chunks.overline_Epsilon_cnu_diss_hat_chunks]
                plt.plot(deformation.t_chunks, overline_Epsilon_cnu_diss_hat___meshpoint_chunk, linestyle='-', color=gp.meshpoints_color_list[meshpoint_indx], alpha=1, linewidth=2.5, label=gp.meshpoints_label_list[meshpoint_indx])
            plt.legend(loc='best')
            plt.grid(True, alpha=0.25)
            save_current_figure(self.savedir, r'$t$', 30, r'$\overline{\hat{E}_{c\nu}^{diss}}$', 30, "fenics-weak-form-uniaxial-rate-independent-t-vs-overline_Epsilon_cnu_diss_hat")

            fig = plt.figure()
            for meshpoint_indx in range(len(gp.meshpoints)):
                overline_Epsilon_cnu_diss_hat___meshpoint_chunk = [overline_Epsilon_cnu_diss_hat_chunk[meshpoint_indx] for overline_Epsilon_cnu_diss_hat_chunk in weak_form_chunks.overline_Epsilon_cnu_diss_hat_chunks]
                plt.plot(deformation.u_2_chunks, overline_Epsilon_cnu_diss_hat___meshpoint_chunk, linestyle='-', color=gp.meshpoints_color_list[meshpoint_indx], alpha=1, linewidth=2.5, label=gp.meshpoints_label_list[meshpoint_indx])
            plt.legend(loc='best')
            plt.grid(True, alpha=0.25)
            save_current_figure(self.savedir, r'$u_2$', 30, r'$\overline{\hat{E}_{c\nu}^{diss}}$', 30, "fenics-weak-form-uniaxial-rate-independent-u_2-vs-overline_Epsilon_cnu_diss_hat")
        

        # overline_epsilon_c_diss_hat
        if ppp.save_overline_epsilon_c_diss_hat_chunks:
            for meshpoint_indx in range(len(gp.meshpoints)):
                fig = plt.figure()
                overline_epsilon_c_diss_hat___meshpoint_chunk = [overline_epsilon_c_diss_hat_chunk[meshpoint_indx] for overline_epsilon_c_diss_hat_chunk in weak_form_chunks.overline_epsilon_c_diss_hat_chunks]
                for nu_chunk_indx in range(len(mp.nu_chunks_list)):
                    overline_epsilon_c_diss_hat___nu_chunk = [overline_epsilon_c_diss_hat_chunk[nu_chunk_indx] for overline_epsilon_c_diss_hat_chunk in overline_epsilon_c_diss_hat___meshpoint_chunk]
                    plt.plot(deformation.t_chunks, overline_epsilon_c_diss_hat___nu_chunk, linestyle='-', color=mp.nu_chunks_color_list[nu_chunk_indx], alpha=1, linewidth=2.5, label=mp.nu_chunks_label_list[nu_chunk_indx])
                plt.legend(loc='best')
                plt.grid(True, alpha=0.25)
                save_current_figure(self.savedir, r'$t$', 30, r'$\overline{\hat{\varepsilon}_c^{diss}}$', 30, "fenics-weak-form-uniaxial-rate-independent-t-vs-overline_epsilon_c_diss_hat"+"_"+gp.meshpoints_name_list[meshpoint_indx])
            
            for meshpoint_indx in range(len(gp.meshpoints)):
                fig = plt.figure()
                overline_epsilon_c_diss_hat___meshpoint_chunk = [overline_epsilon_c_diss_hat_chunk[meshpoint_indx] for overline_epsilon_c_diss_hat_chunk in weak_form_chunks.overline_epsilon_c_diss_hat_chunks]
                for nu_chunk_indx in range(len(mp.nu_chunks_list)):
                    overline_epsilon_c_diss_hat___nu_chunk = [overline_epsilon_c_diss_hat_chunk[nu_chunk_indx] for overline_epsilon_c_diss_hat_chunk in overline_epsilon_c_diss_hat___meshpoint_chunk]
                    plt.plot(deformation.u_2_chunks, overline_epsilon_c_diss_hat___nu_chunk, linestyle='-', color=mp.nu_chunks_color_list[nu_chunk_indx], alpha=1, linewidth=2.5, label=mp.nu_chunks_label_list[nu_chunk_indx])
                plt.legend(loc='best')
                plt.grid(True, alpha=0.25)
                save_current_figure(self.savedir, r'$u_2$', 30, r'$\overline{\hat{\varepsilon}_c^{diss}}$', 30, "fenics-weak-form-uniaxial-rate-independent-u_2-vs-overline_epsilon_c_diss_hat"+"_"+gp.meshpoints_name_list[meshpoint_indx])
        
        # overline_Epsilon_c_diss_hat
        if ppp.save_overline_Epsilon_c_diss_hat_chunks:
            fig = plt.figure()
            for meshpoint_indx in range(len(gp.meshpoints)):
                overline_Epsilon_c_diss_hat___meshpoint_chunk = [overline_Epsilon_c_diss_hat_chunk[meshpoint_indx] for overline_Epsilon_c_diss_hat_chunk in weak_form_chunks.overline_Epsilon_c_diss_hat_chunks]
                plt.plot(deformation.t_chunks, overline_Epsilon_c_diss_hat___meshpoint_chunk, linestyle='-', color=gp.meshpoints_color_list[meshpoint_indx], alpha=1, linewidth=2.5, label=gp.meshpoints_label_list[meshpoint_indx])
            plt.legend(loc='best')
            plt.grid(True, alpha=0.25)
            save_current_figure(self.savedir, r'$t$', 30, r'$\overline{\hat{E}_c^{diss}}$', 30, "fenics-weak-form-uniaxial-rate-independent-t-vs-overline_Epsilon_c_diss_hat")

            fig = plt.figure()
            for meshpoint_indx in range(len(gp.meshpoints)):
                overline_Epsilon_c_diss_hat___meshpoint_chunk = [overline_Epsilon_c_diss_hat_chunk[meshpoint_indx] for overline_Epsilon_c_diss_hat_chunk in weak_form_chunks.overline_Epsilon_c_diss_hat_chunks]
                plt.plot(deformation.u_2_chunks, overline_Epsilon_c_diss_hat___meshpoint_chunk, linestyle='-', color=gp.meshpoints_color_list[meshpoint_indx], alpha=1, linewidth=2.5, label=gp.meshpoints_label_list[meshpoint_indx])
            plt.legend(loc='best')
            plt.grid(True, alpha=0.25)
            save_current_figure(self.savedir, r'$u_2$', 30, r'$\overline{\hat{E}_c^{diss}}$', 30, "fenics-weak-form-uniaxial-rate-independent-u_2-vs-overline_Epsilon_c_diss_hat")
        
        # sigma
        if ppp.save_sigma_chunks:
            fig = plt.figure()
            for meshpoint_indx in range(len(gp.meshpoints)):
                sigma_22___meshpoint_chunk = [sigma_22_chunk[meshpoint_indx] for sigma_22_chunk in weak_form_chunks.sigma_22_chunks]
                plt.plot(deformation.t_chunks, sigma_22___meshpoint_chunk, linestyle='-', color=gp.meshpoints_color_list[meshpoint_indx], alpha=1, linewidth=2.5, label=gp.meshpoints_label_list[meshpoint_indx])
            plt.legend(loc='best')
            plt.grid(True, alpha=0.25)
            save_current_figure(self.savedir, r'$t$', 30, r'$\sigma_{22}$', 30, "fenics-weak-form-uniaxial-rate-independent-t-vs-sigma_22")

            fig = plt.figure()
            for meshpoint_indx in range(len(gp.meshpoints)):
                sigma_22___meshpoint_chunk = [sigma_22_chunk[meshpoint_indx] for sigma_22_chunk in weak_form_chunks.sigma_22_chunks]
                plt.plot(deformation.u_2_chunks, sigma_22___meshpoint_chunk, linestyle='-', color=gp.meshpoints_color_list[meshpoint_indx], alpha=1, linewidth=2.5, label=gp.meshpoints_label_list[meshpoint_indx])
            plt.legend(loc='best')
            plt.grid(True, alpha=0.25)
            save_current_figure(self.savedir, r'$u_2$', 30, r'$\sigma_{22}$', 30, "fenics-weak-form-uniaxial-rate-independent-u_2-vs-sigma_22")

            fig = plt.figure()
            for meshpoint_indx in range(len(gp.meshpoints)):
                sigma_22_penalty_term___meshpoint_chunk = [sigma_22_penalty_term_chunk[meshpoint_indx] for sigma_22_penalty_term_chunk in weak_form_chunks.sigma_22_penalty_term_chunks]
                plt.plot(deformation.t_chunks, sigma_22_penalty_term___meshpoint_chunk, linestyle='-', color=gp.meshpoints_color_list[meshpoint_indx], alpha=1, linewidth=2.5, label=gp.meshpoints_label_list[meshpoint_indx])
            plt.legend(loc='best')
            plt.grid(True, alpha=0.25)
            save_current_figure(self.savedir, r'$t$', 30, r'$(\sigma_{22})_{penalty}$', 30, "fenics-weak-form-uniaxial-rate-independent-t-vs-sigma_22_penalty_term")

            fig = plt.figure()
            for meshpoint_indx in range(len(gp.meshpoints)):
                sigma_22_penalty_term___meshpoint_chunk = [sigma_22_penalty_term_chunk[meshpoint_indx] for sigma_22_penalty_term_chunk in weak_form_chunks.sigma_22_penalty_term_chunks]
                plt.plot(deformation.u_2_chunks, sigma_22_penalty_term___meshpoint_chunk, linestyle='-', color=gp.meshpoints_color_list[meshpoint_indx], alpha=1, linewidth=2.5, label=gp.meshpoints_label_list[meshpoint_indx])
            plt.legend(loc='best')
            plt.grid(True, alpha=0.25)
            save_current_figure(self.savedir, r'$u_2$', 30, r'$(\sigma_{22})_{penalty}$', 30, "fenics-weak-form-uniaxial-rate-independent-u_2-vs-sigma_22_penalty_term")

            fig = plt.figure()
            for meshpoint_indx in range(len(gp.meshpoints)):
                sigma_22_less_penalty_term___meshpoint_chunk = [sigma_22_less_penalty_term_chunk[meshpoint_indx] for sigma_22_less_penalty_term_chunk in weak_form_chunks.sigma_22_less_penalty_term_chunks]
                plt.plot(deformation.t_chunks, sigma_22_less_penalty_term___meshpoint_chunk, linestyle='-', color=gp.meshpoints_color_list[meshpoint_indx], alpha=1, linewidth=2.5, label=gp.meshpoints_label_list[meshpoint_indx])
            plt.legend(loc='best')
            plt.grid(True, alpha=0.25)
            save_current_figure(self.savedir, r'$t$', 30, r'$\sigma_{22} - (\sigma_{22})_{penalty}$', 30, "fenics-weak-form-uniaxial-rate-independent-t-vs-sigma_22_less_penalty_term")

            fig = plt.figure()
            for meshpoint_indx in range(len(gp.meshpoints)):
                sigma_22_less_penalty_term___meshpoint_chunk = [sigma_22_less_penalty_term_chunk[meshpoint_indx] for sigma_22_less_penalty_term_chunk in weak_form_chunks.sigma_22_less_penalty_term_chunks]
                plt.plot(deformation.u_2_chunks, sigma_22_less_penalty_term___meshpoint_chunk, linestyle='-', color=gp.meshpoints_color_list[meshpoint_indx], alpha=1, linewidth=2.5, label=gp.meshpoints_label_list[meshpoint_indx])
            plt.legend(loc='best')
            plt.grid(True, alpha=0.25)
            save_current_figure(self.savedir, r'$u_2$', 30, r'$\sigma_{22} - (\sigma_{22})_{penalty}$', 30, "fenics-weak-form-uniaxial-rate-independent-u_2-vs-sigma_22_less_penalty_term")

        # F
        if ppp.save_F_chunks:
            fig = plt.figure()
            for meshpoint_indx in range(len(gp.meshpoints)):
                F_22___meshpoint_chunk = [F_22_chunk[meshpoint_indx] for F_22_chunk in weak_form_chunks.F_22_chunks]
                plt.plot(deformation.t_chunks, F_22___meshpoint_chunk, linestyle='-', color=gp.meshpoints_color_list[meshpoint_indx], alpha=1, linewidth=2.5, label=gp.meshpoints_label_list[meshpoint_indx])
            plt.legend(loc='best')
            plt.grid(True, alpha=0.25)
            save_current_figure(self.savedir, r'$t$', 30, r'$F_{22}$', 30, "fenics-weak-form-uniaxial-rate-independent-t-vs-F_22")

            fig = plt.figure()
            for meshpoint_indx in range(len(gp.meshpoints)):
                F_22___meshpoint_chunk = [F_22_chunk[meshpoint_indx] for F_22_chunk in weak_form_chunks.F_22_chunks]
                plt.plot(deformation.u_2_chunks, F_22___meshpoint_chunk, linestyle='-', color=gp.meshpoints_color_list[meshpoint_indx], alpha=1, linewidth=2.5, label=gp.meshpoints_label_list[meshpoint_indx])
            plt.legend(loc='best')
            plt.grid(True, alpha=0.25)
            save_current_figure(self.savedir, r'$u_2$', 30, r'$F_{22}$', 30, "fenics-weak-form-uniaxial-rate-independent-u_2-vs-F_22")

if __name__ == '__main__':

    L, H = 1.0, 1.5
    x_notch_point = 0.5
    r_notch = 0.02
    notch_fine_mesh_layer_level_num = 1
    fine_mesh_elem_size = 0.01
    coarse_mesh_elem_size = 0.01 # 0.01 # 0.25
    l_nl = coarse_mesh_elem_size # 10*r_notch # 1.25*r_notch # 0.02 = 2*coarse_mesh_elem_size
    problem = TwoDimensionalPlaneStrainNearlyIncompressibleNonaffineEightChainModelEqualStrainRateIndependentNotchedCrack(L, H, x_notch_point, r_notch, notch_fine_mesh_layer_level_num, fine_mesh_elem_size, coarse_mesh_elem_size, l_nl)
    problem.solve_fenics_weak_form_deformation()