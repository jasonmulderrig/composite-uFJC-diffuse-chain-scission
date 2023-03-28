# Import necessary libraries
from __future__ import division
from dolfin import *
from .composite_ufjc_network import CompositeuFJCNetwork
import numpy as np
from types import SimpleNamespace
from copy import deepcopy
import sys

class TwoDimensionalPlaneStrainIncompressibleNonaffineEightChainModelEqualStrainRateIndependentCompositeuFJCNetwork:

    def __init__(self, parameters, strong_form_initialize_sigma_chunks,
                 lr_cg_deformation_gradient_func,
                 strong_form_calculate_sigma_func,
                 strong_form_store_calculated_sigma_chunks,
                 weak_form_initialize_deformation_sigma_chunks,
                 weak_form_store_calculated_sigma_chunks,
                 weak_form_store_calculated_deformation_chunks):
        
        # Retain specified functions for deformation-specific calculations
        self.strong_form_initialize_sigma_chunks = (
            strong_form_initialize_sigma_chunks
        )
        self.lr_cg_deformation_gradient_func = (
            lr_cg_deformation_gradient_func
        )
        self.strong_form_calculate_sigma_func  = (
            strong_form_calculate_sigma_func
        )
        self.strong_form_store_calculated_sigma_chunks = (
            strong_form_store_calculated_sigma_chunks
        )
        self.weak_form_initialize_deformation_sigma_chunks = (
            weak_form_initialize_deformation_sigma_chunks
        )
        self.weak_form_store_calculated_sigma_chunks = (
            weak_form_store_calculated_sigma_chunks
        )
        self.weak_form_store_calculated_deformation_chunks = (
            weak_form_store_calculated_deformation_chunks
        )

        CompositeuFJCNetwork.__init__(self, parameters)

class TwoDimensionalPlaneStrainIncompressibleNonaffineEightChainModelEqualStrainRateDependentCompositeuFJCNetwork:

    def __init__(self, parameters, strong_form_initialize_sigma_chunks,
                 lr_cg_deformation_gradient_func,
                 strong_form_calculate_sigma_func,
                 strong_form_store_calculated_sigma_chunks,
                 weak_form_initialize_deformation_sigma_chunks,
                 weak_form_store_calculated_sigma_chunks,
                 weak_form_store_calculated_deformation_chunks):
        
        # Retain specified functions for deformation-specific calculations
        self.strong_form_initialize_sigma_chunks = (
            strong_form_initialize_sigma_chunks
        )
        self.lr_cg_deformation_gradient_func = (
            lr_cg_deformation_gradient_func
        )
        self.strong_form_calculate_sigma_func  = (
            strong_form_calculate_sigma_func
        )
        self.strong_form_store_calculated_sigma_chunks = (
            strong_form_store_calculated_sigma_chunks
        )
        self.weak_form_initialize_deformation_sigma_chunks = (
            weak_form_initialize_deformation_sigma_chunks
        )
        self.weak_form_store_calculated_sigma_chunks = (
            weak_form_store_calculated_sigma_chunks
        )
        self.weak_form_store_calculated_deformation_chunks = (
            weak_form_store_calculated_deformation_chunks
        )

        CompositeuFJCNetwork.__init__(self, parameters)

class TwoDimensionalPlaneStrainIncompressibleNonaffineEightChainModelEqualForceRateIndependentCompositeuFJCNetwork:

    def __init__(self, parameters, strong_form_initialize_sigma_chunks,
                 lr_cg_deformation_gradient_func,
                 strong_form_calculate_sigma_func,
                 strong_form_store_calculated_sigma_chunks,
                 weak_form_initialize_deformation_sigma_chunks,
                 weak_form_store_calculated_sigma_chunks,
                 weak_form_store_calculated_deformation_chunks):
        
        # Retain specified functions for deformation-specific calculations
        self.strong_form_initialize_sigma_chunks = (
            strong_form_initialize_sigma_chunks
        )
        self.lr_cg_deformation_gradient_func = (
            lr_cg_deformation_gradient_func
        )
        self.strong_form_calculate_sigma_func  = (
            strong_form_calculate_sigma_func
        )
        self.strong_form_store_calculated_sigma_chunks = (
            strong_form_store_calculated_sigma_chunks
        )
        self.weak_form_initialize_deformation_sigma_chunks = (
            weak_form_initialize_deformation_sigma_chunks
        )
        self.weak_form_store_calculated_sigma_chunks = (
            weak_form_store_calculated_sigma_chunks
        )
        self.weak_form_store_calculated_deformation_chunks = (
            weak_form_store_calculated_deformation_chunks
        )

        CompositeuFJCNetwork.__init__(self, parameters)

class TwoDimensionalPlaneStrainIncompressibleNonaffineEightChainModelEqualForceRateDependentCompositeuFJCNetwork:

    def __init__(self, parameters, strong_form_initialize_sigma_chunks,
                 lr_cg_deformation_gradient_func,
                 strong_form_calculate_sigma_func,
                 strong_form_store_calculated_sigma_chunks,
                 weak_form_initialize_deformation_sigma_chunks,
                 weak_form_store_calculated_sigma_chunks,
                 weak_form_store_calculated_deformation_chunks):
        
        # Retain specified functions for deformation-specific calculations
        self.strong_form_initialize_sigma_chunks = (
            strong_form_initialize_sigma_chunks
        )
        self.lr_cg_deformation_gradient_func = (
            lr_cg_deformation_gradient_func
        )
        self.strong_form_calculate_sigma_func  = (
            strong_form_calculate_sigma_func
        )
        self.strong_form_store_calculated_sigma_chunks = (
            strong_form_store_calculated_sigma_chunks
        )
        self.weak_form_initialize_deformation_sigma_chunks = (
            weak_form_initialize_deformation_sigma_chunks
        )
        self.weak_form_store_calculated_sigma_chunks = (
            weak_form_store_calculated_sigma_chunks
        )
        self.weak_form_store_calculated_deformation_chunks = (
            weak_form_store_calculated_deformation_chunks
        )

        CompositeuFJCNetwork.__init__(self, parameters)

class TwoDimensionalPlaneStrainNearlyIncompressibleNonaffineEightChainModelEqualStrainRateIndependentCompositeuFJCNetwork:

    def __init__(self, parameters, strong_form_initialize_sigma_chunks,
                 lr_cg_deformation_gradient_func,
                 strong_form_calculate_sigma_func,
                 strong_form_store_calculated_sigma_chunks,
                 weak_form_initialize_deformation_sigma_chunks,
                 weak_form_store_calculated_sigma_chunks,
                 weak_form_store_calculated_deformation_chunks):
        
        # Retain specified functions for deformation-specific calculations
        self.strong_form_initialize_sigma_chunks = (
            strong_form_initialize_sigma_chunks
        )
        self.lr_cg_deformation_gradient_func = (
            lr_cg_deformation_gradient_func
        )
        self.strong_form_calculate_sigma_func  = (
            strong_form_calculate_sigma_func
        )
        self.strong_form_store_calculated_sigma_chunks = (
            strong_form_store_calculated_sigma_chunks
        )
        self.weak_form_initialize_deformation_sigma_chunks = (
            weak_form_initialize_deformation_sigma_chunks
        )
        self.weak_form_store_calculated_sigma_chunks = (
            weak_form_store_calculated_sigma_chunks
        )
        self.weak_form_store_calculated_deformation_chunks = (
            weak_form_store_calculated_deformation_chunks
        )

        CompositeuFJCNetwork.__init__(self, parameters)
    
    def homogeneous_strong_form_initialization(self):
        results = SimpleNamespace()
        chunks  = SimpleNamespace()

        # initialize lists to zeros - necessary for irreversibility
        chunks = self.strong_form_initialize_sigma_chunks(chunks)
        
        # lmbda_c
        chunks.lmbda_c_chunks = []
        results.lmbda_c_val   = 0.
        # lmbda_c_eq
        chunks.lmbda_c_eq_chunks     = []
        chunks.lmbda_c_eq_chunks_val = [0. for nu_chunk_indx in range(len(self.nu_chunks_indx_list))]
        results.lmbda_c_eq_val       = [0. for nu_indx in range(self.nu_num)]
        # lmbda_nu
        chunks.lmbda_nu_chunks     = []
        chunks.lmbda_nu_chunks_val = [0. for nu_chunk_indx in range(len(self.nu_chunks_indx_list))]
        results.lmbda_nu_val       = [0. for nu_indx in range(self.nu_num)]
        # lmbda_nu_max
        chunks.lmbda_nu_max_chunks     = []
        chunks.lmbda_nu_max_chunks_val = [0. for nu_chunk_indx in range(len(self.nu_chunks_indx_list))]
        results.lmbda_nu_max_val       = [0. for nu_indx in range(self.nu_num)]
        # upsilon_c
        chunks.upsilon_c_chunks     = []
        chunks.upsilon_c_chunks_val = [0. for nu_chunk_indx in range(len(self.nu_chunks_indx_list))]
        results.upsilon_c_val       = [0. for nu_indx in range(self.nu_num)]
        # Upsilon_c
        chunks.Upsilon_c_chunks     = []
        chunks.Upsilon_c_chunks_val = [0. for nu_chunk_indx in range(len(self.nu_chunks_indx_list))]
        results.Upsilon_c_val       = [0. for nu_indx in range(self.nu_num)]
        # d_c
        chunks.d_c_chunks     = []
        chunks.d_c_chunks_val = [0. for nu_chunk_indx in range(len(self.nu_chunks_indx_list))]
        results.d_c_val       = [0. for nu_indx in range(self.nu_num)]
        # D_c
        chunks.D_c_chunks = []
        results.D_c_val   = 0.
        # epsilon_cnu_diss_hat
        chunks.epsilon_cnu_diss_hat_chunks     = []
        chunks.epsilon_cnu_diss_hat_chunks_val = [0. for nu_chunk_indx in range(len(self.nu_chunks_indx_list))]
        results.epsilon_cnu_diss_hat_val       = [0. for nu_indx in range(self.nu_num)]
        # Epsilon_cnu_diss_hat
        chunks.Epsilon_cnu_diss_hat_chunks = []
        results.Epsilon_cnu_diss_hat_val   = 0.
        # epsilon_c_diss_hat
        chunks.epsilon_c_diss_hat_chunks     = []
        chunks.epsilon_c_diss_hat_chunks_val = [0. for nu_chunk_indx in range(len(self.nu_chunks_indx_list))]
        results.epsilon_c_diss_hat_val       = [0. for nu_indx in range(self.nu_num)]
        # Epsilon_c_diss_hat
        chunks.Epsilon_c_diss_hat_chunks = []
        results.Epsilon_c_diss_hat_val   = 0.
        # overline_epsilon_cnu_diss_hat
        chunks.overline_epsilon_cnu_diss_hat_chunks     = []
        chunks.overline_epsilon_cnu_diss_hat_chunks_val = [0. for nu_chunk_indx in range(len(self.nu_chunks_indx_list))]
        results.overline_epsilon_cnu_diss_hat_val       = [0. for nu_indx in range(self.nu_num)]
        # overline_Epsilon_cnu_diss_hat
        chunks.overline_Epsilon_cnu_diss_hat_chunks = []
        results.overline_Epsilon_cnu_diss_hat_val   = 0.
        # overline_epsilon_c_diss_hat
        chunks.overline_epsilon_c_diss_hat_chunks     = []
        chunks.overline_epsilon_c_diss_hat_chunks_val = [0. for nu_chunk_indx in range(len(self.nu_chunks_indx_list))]
        results.overline_epsilon_c_diss_hat_val       = [0. for nu_indx in range(self.nu_num)]
        # overline_Epsilon_c_diss_hat
        chunks.overline_Epsilon_c_diss_hat_chunks = []
        results.overline_Epsilon_c_diss_hat_val   = 0.
        # sigma_hyp_val
        results.sigma_hyp_val = 0.

        return results, chunks

    def homogeneous_strong_form_solve_step(self, deformation, results):
        F_val, C_val, b_val = self.lr_cg_deformation_gradient_func(deformation)

        Upsilon_c_val                     = 0
        D_c_val                           = 0
        Epsilon_cnu_diss_hat_val          = 0
        Epsilon_c_diss_hat_val            = 0
        overline_Epsilon_cnu_diss_hat_val = 0
        overline_Epsilon_c_diss_hat_val   = 0
        sigma_hyp_val                     = 0
        
        lmbda_c_val = np.sqrt(np.trace(C_val)/3.)

        zeta_nu_char = self.composite_ufjc_list[0].zeta_nu_char
        
        for nu_indx in range(self.nu_num):
            nu_val              = self.nu_list[nu_indx]
            P_nu___nu_val       = self.P_nu_list[nu_indx]
            A_nu___nu_val       = self.composite_ufjc_list[nu_indx].A_nu
            lmbda_c_eq___nu_val = lmbda_c_val*A_nu___nu_val
            lmbda_nu___nu_val   = self.composite_ufjc_list[nu_indx].lmbda_nu_func(lmbda_c_eq___nu_val)
            xi_c___nu_val       = self.composite_ufjc_list[nu_indx].xi_c_func(lmbda_nu___nu_val, lmbda_c_eq___nu_val)
            # impose irreversibility
            lmbda_nu_max___nu_val                  = max([results.lmbda_nu_max_val[nu_indx], lmbda_nu___nu_val])
            upsilon_c___nu_val                     = (1.-self.k_cond_val)*self.composite_ufjc_list[nu_indx].p_c_sur_hat_func(lmbda_nu_max___nu_val) + self.k_cond_val
            d_c___nu_val                           = 1. - upsilon_c___nu_val
            epsilon_cnu_diss_hat___nu_val          = self.composite_ufjc_list[nu_indx].epsilon_cnu_diss_hat_equiv_func(lmbda_nu_max___nu_val)
            epsilon_c_diss_hat___nu_val            = epsilon_cnu_diss_hat___nu_val*nu_val
            overline_epsilon_cnu_diss_hat___nu_val = epsilon_cnu_diss_hat___nu_val/zeta_nu_char
            overline_epsilon_c_diss_hat___nu_val   = epsilon_c_diss_hat___nu_val/zeta_nu_char

            results.lmbda_c_eq_val[nu_indx]                    = lmbda_c_eq___nu_val
            results.lmbda_nu_val[nu_indx]                      = lmbda_nu___nu_val
            results.lmbda_nu_max_val[nu_indx]                  = lmbda_nu_max___nu_val
            results.upsilon_c_val[nu_indx]                     = upsilon_c___nu_val
            results.d_c_val[nu_indx]                           = d_c___nu_val
            results.epsilon_cnu_diss_hat_val[nu_indx]          = epsilon_cnu_diss_hat___nu_val
            results.epsilon_c_diss_hat_val[nu_indx]            = epsilon_c_diss_hat___nu_val
            results.overline_epsilon_cnu_diss_hat_val[nu_indx] = overline_epsilon_cnu_diss_hat___nu_val
            results.overline_epsilon_c_diss_hat_val[nu_indx]   = overline_epsilon_c_diss_hat___nu_val

            Upsilon_c_val                     += P_nu___nu_val*upsilon_c___nu_val
            D_c_val                           += P_nu___nu_val*d_c___nu_val
            Epsilon_cnu_diss_hat_val          += P_nu___nu_val*epsilon_cnu_diss_hat___nu_val
            Epsilon_c_diss_hat_val            += P_nu___nu_val*epsilon_c_diss_hat___nu_val
            overline_Epsilon_cnu_diss_hat_val += P_nu___nu_val*overline_epsilon_cnu_diss_hat___nu_val
            overline_Epsilon_c_diss_hat_val   += P_nu___nu_val*overline_epsilon_c_diss_hat___nu_val
            sigma_hyp_val                     += upsilon_c___nu_val*P_nu___nu_val*nu_val*A_nu___nu_val*xi_c___nu_val/(3.*lmbda_c_val)
        
        Upsilon_c_val                     = Upsilon_c_val/self.P_nu_sum
        D_c_val                           = D_c_val/self.P_nu_sum
        Epsilon_cnu_diss_hat_val          = Epsilon_cnu_diss_hat_val/self.P_nu_sum
        Epsilon_c_diss_hat_val            = Epsilon_c_diss_hat_val/self.P_nu_sum
        overline_Epsilon_cnu_diss_hat_val = overline_Epsilon_cnu_diss_hat_val/self.P_nu_sum
        overline_Epsilon_c_diss_hat_val   = overline_Epsilon_c_diss_hat_val/self.P_nu_sum
        
        results.lmbda_c_val                       = lmbda_c_val
        results.Upsilon_c_val                     = Upsilon_c_val
        results.D_c_val                           = D_c_val
        results.Epsilon_cnu_diss_hat_val          = Epsilon_cnu_diss_hat_val
        results.Epsilon_c_diss_hat_val            = Epsilon_c_diss_hat_val
        results.overline_Epsilon_cnu_diss_hat_val = overline_Epsilon_cnu_diss_hat_val
        results.overline_Epsilon_c_diss_hat_val   = overline_Epsilon_c_diss_hat_val
        results.sigma_hyp_val                     = sigma_hyp_val

        return results
    
    def homogeneous_strong_form_chunk_post_processing(self, deformation, results, chunks):
        for nu_chunk_indx in range(len(self.nu_chunks_indx_list)):
            # first dimension is nu_chunk_val: list[nu_chunk_val]
            chunks.lmbda_c_eq_chunks_val[nu_chunk_indx]                    = results.lmbda_c_eq_val[self.nu_chunks_indx_list[nu_chunk_indx]]
            chunks.lmbda_nu_chunks_val[nu_chunk_indx]                      = results.lmbda_nu_val[self.nu_chunks_indx_list[nu_chunk_indx]]
            chunks.lmbda_nu_max_chunks_val[nu_chunk_indx]                  = results.lmbda_nu_max_val[self.nu_chunks_indx_list[nu_chunk_indx]]
            chunks.upsilon_c_chunks_val[nu_chunk_indx]                     = results.upsilon_c_val[self.nu_chunks_indx_list[nu_chunk_indx]]
            chunks.d_c_chunks_val[nu_chunk_indx]                           = results.d_c_val[self.nu_chunks_indx_list[nu_chunk_indx]]
            chunks.epsilon_cnu_diss_hat_chunks_val[nu_chunk_indx]          = results.epsilon_cnu_diss_hat_val[self.nu_chunks_indx_list[nu_chunk_indx]]
            chunks.epsilon_c_diss_hat_chunks_val[nu_chunk_indx]            = results.epsilon_c_diss_hat_val[self.nu_chunks_indx_list[nu_chunk_indx]]
            chunks.overline_epsilon_cnu_diss_hat_chunks_val[nu_chunk_indx] = results.overline_epsilon_cnu_diss_hat_val[self.nu_chunks_indx_list[nu_chunk_indx]]
            chunks.overline_epsilon_c_diss_hat_chunks_val[nu_chunk_indx]   = results.overline_epsilon_c_diss_hat_val[self.nu_chunks_indx_list[nu_chunk_indx]]

        sigma_val = self.strong_form_calculate_sigma_func(results.sigma_hyp_val, deformation)
        chunks    = self.strong_form_store_calculated_sigma_chunks(sigma_val, chunks)

        # first dimension is t_chunk_val: list[t_chunk_val]
        chunks.lmbda_c_chunks.append(results.lmbda_c_val)
        chunks.Upsilon_c_chunks.append(results.Upsilon_c_val)
        chunks.D_c_chunks.append(results.D_c_val)
        chunks.Epsilon_cnu_diss_hat_chunks.append(results.Epsilon_cnu_diss_hat_val)
        chunks.Epsilon_c_diss_hat_chunks.append(results.Epsilon_c_diss_hat_val)
        chunks.overline_Epsilon_cnu_diss_hat_chunks.append(results.overline_Epsilon_cnu_diss_hat_val)
        chunks.overline_Epsilon_c_diss_hat_chunks.append(results.overline_Epsilon_c_diss_hat_val)
        
        # first dimension is t_chunk_val, second dimension is nu_chunk_val: list[t_chunk_val][nu_chunk_val]
        chunks.lmbda_c_eq_chunks.append(deepcopy(chunks.lmbda_c_eq_chunks_val))
        chunks.lmbda_nu_chunks.append(deepcopy(chunks.lmbda_nu_chunks_val))
        chunks.lmbda_nu_max_chunks.append(deepcopy(chunks.lmbda_nu_max_chunks_val))
        chunks.upsilon_c_chunks.append(deepcopy(chunks.upsilon_c_chunks_val))
        chunks.d_c_chunks.append(deepcopy(chunks.d_c_chunks_val))
        chunks.epsilon_cnu_diss_hat_chunks.append(deepcopy(chunks.epsilon_cnu_diss_hat_chunks_val))
        chunks.epsilon_c_diss_hat_chunks.append(deepcopy(chunks.epsilon_c_diss_hat_chunks_val))
        chunks.overline_epsilon_cnu_diss_hat_chunks.append(deepcopy(chunks.overline_epsilon_cnu_diss_hat_chunks_val))
        chunks.overline_epsilon_c_diss_hat_chunks.append(deepcopy(chunks.overline_epsilon_c_diss_hat_chunks_val))

        return chunks

    def fenics_variational_formulation(self, parameters, fem):
        femp = parameters.fem
        mp   = parameters.material

        # Create function space for displacement
        fem.V_u = VectorFunctionSpace(fem.mesh, "CG", femp.u_degree)
        # Create function space for non-local chain stretch
        fem.V_lmbda_c_tilde = FunctionSpace(fem.mesh, "CG", femp.scalar_prmtr_degree)
        # Create function space for maximal non-local chain stretch
        # fem.V_lmbda_c_tilde_max = FunctionSpace(fem.mesh, "CG", femp.scalar_prmtr_degree)

        if fem.solver_algorithm == "alternate_minimization":
            # Define solution, trial, and test functions, respectively, for displacement
            fem.u   = Function(fem.V_u)
            fem.du  = TrialFunction(fem.V_u)
            fem.v_u = TestFunction(fem.V_u)

            # Define solution, trial, and test functions, respectively, for non-local chain stretch
            fem.lmbda_c_tilde   = Function(fem.V_lmbda_c_tilde)
            fem.dlmbda_c_tilde  = TrialFunction(fem.V_lmbda_c_tilde)
            fem.v_lmbda_c_tilde = TestFunction(fem.V_lmbda_c_tilde)

            # Define objects needed for calculations
            fem.I           = Identity(len(fem.u))
            fem.V_CG_scalar = FunctionSpace(fem.mesh, "CG", femp.scalar_prmtr_degree)
            fem.V_DG_scalar = FunctionSpace(fem.mesh, "DG", femp.scalar_prmtr_degree)
            fem.V_DG_tensor = TensorFunctionSpace(fem.mesh, "DG", 0)

            # Define body force and traction force
            fem.b = Constant((0.0, 0.0)) # Body force per unit volume
            fem.t = Constant((0.0, 0.0)) # Traction force on the boundary

            # Initialization
            fem.lmbda_c_tilde = interpolate(Expression("1.", degree=femp.scalar_prmtr_degree), fem.V_lmbda_c_tilde)
            fem.lmbda_c_tilde_prior = interpolate(Expression("1.", degree=femp.scalar_prmtr_degree), fem.V_lmbda_c_tilde)
            fem.lmbda_c_tilde_max = interpolate(Expression("1.", degree=femp.scalar_prmtr_degree), fem.V_lmbda_c_tilde)

            if fem.solver_bounded is True:
                # Initialization
                fem.lmbda_c_tilde_lb = interpolate(Expression("1.", degree=femp.scalar_prmtr_degree), fem.V_lmbda_c_tilde)
                # The upper-bound is technically (+)infinity
                fem.lmbda_c_tilde_ub = interpolate(Expression("10000.0", degree=femp.scalar_prmtr_degree), fem.V_lmbda_c_tilde)

            # Kinematics
            fem.F           = fem.I + grad(fem.u) # deformation gradient tensor
            fem.F_inv       = inv(fem.F) # inverse deformation gradient tensor
            fem.J           = det(fem.F) # volume ratio
            fem.C           = fem.F.T*fem.F # right Cauchy-Green tensor
            fem.I_C         = tr(fem.C)+1 # 2D plane strain form of the trace of right Cauchy-Green tensor, where F_33 = 1 always -- this is the case of plane strain
            fem.lmbda_c     = sqrt(fem.I_C/3.0)

            # Calculate the weak form for displacement
            fem.WF_u = inner(self.first_pk_stress_ufl_fenics_mesh_func(fem), grad(fem.v_u))*dx(metadata=femp.metadata) - dot(fem.b, fem.v_u)*dx(metadata=femp.metadata) - dot(fem.t, fem.v_u)*ds

            # Calculate the Gateaux derivative for displacement
            fem.Jac_u = derivative(fem.WF_u, fem.u, fem.du)

            # Calculate the weak form for non-local chain stretch
            fem.WF_lmbda_c_tilde = (fem.v_lmbda_c_tilde*fem.lmbda_c_tilde + mp.l_nl**2*dot(grad(fem.v_lmbda_c_tilde), grad(fem.lmbda_c_tilde)) - fem.v_lmbda_c_tilde*fem.lmbda_c)*dx(metadata=femp.metadata)

            # Calculate the Gateaux derivative for non-local chain stretch
            fem.Jac_lmbda_c_tilde = derivative(fem.WF_lmbda_c_tilde, fem.lmbda_c_tilde, fem.dlmbda_c_tilde)
        
        elif fem.solver_algorithm == "monolithic":
            # Create UFL element from the displacement function space
            fem.V_u_ufl_elem = fem.V_u.ufl_element()
            # Create UFL element from the non-local chain stretch function space
            fem.V_lmbda_c_tilde_ufl_elem = fem.V_lmbda_c_tilde.ufl_element()
            # Create UFL mixed element
            fem.mixed_ufl_elem = MixedElement([fem.V_u_ufl_elem, fem.V_lmbda_c_tilde_ufl_elem])
            # Define function space for the UFL mixed element
            fem.V = FunctionSpace(fem.mesh, fem.mixed_ufl_elem)

            # Define solution, trial, and test functions for the UFL mixed element
            fem.mixed_space = Function(fem.V)
            fem.dmixed_space = TrialFunction(fem.V)
            fem.v_mixed_space = TestFunction(fem.V)

            # Split the mixed function space and the mixed test function space for displacement and non-local chain stretch
            (fem.u, fem.lmbda_c_tilde) = split(fem.mixed_space)
            (fem.v_u, fem.v_lmbda_c_tilde) = split(fem.v_mixed_space)

            # Define objects needed for calculations
            fem.I        = Identity(len(fem.u))
            fem.V_CG_scalar = FunctionSpace(fem.mesh, "CG", femp.scalar_prmtr_degree)
            fem.V_DG_scalar = FunctionSpace(fem.mesh, "DG", femp.scalar_prmtr_degree)
            fem.V_DG_tensor = TensorFunctionSpace(fem.mesh, "DG", 0)

            # Define body force and traction force
            fem.b = Constant((0.0, 0.0)) # Body force per unit volume
            fem.t = Constant((0.0, 0.0)) # Traction force on the boundary

            # Initialization
            fem.lmbda_c_tilde_max = interpolate(Expression("0.", degree=femp.scalar_prmtr_degree), fem.V_lmbda_c_tilde)
            if femp.u_degree != femp.scalar_prmtr_degree:
                sys.exit("The displacement element degreee is not equal to the non-local chain stretch element degree")
            else:
                mixed_space_degree = femp.scalar_prmtr_degree
                # Initial conditions class
                class InitialConditions(UserExpression):
                    def eval(self, vals, x):
                        vals[0] = 0.0 # u_x
                        vals[1] = 0.0 # u_y
                        vals[2] = 1.0 # tilde_lambda_c
                    def value_shape(self):
                        return (3,)
                
                ics = InitialConditions(degree=mixed_space_degree)
                fem.mixed_space.interpolate(ics)

            # Kinematics
            fem.F           = fem.I + grad(fem.u) # deformation gradient tensor
            fem.F_inv       = inv(fem.F) # inverse deformation gradient tensor
            fem.J           = det(fem.F) # volume ratio
            fem.C           = fem.F.T*fem.F # right Cauchy-Green tensor
            fem.I_C         = tr(fem.C)+1 # 2D plane strain form of the trace of right Cauchy-Green tensor, where F_33 = 1 always -- this is the case of plane strain
            fem.lmbda_c     = sqrt(fem.I_C/3.0)

            # Calculate the weak form for displacement
            fem.WF_u = inner(self.first_pk_stress_ufl_fenics_mesh_func(fem), grad(fem.v_u))*dx(metadata=femp.metadata) - dot(fem.b, fem.v_u)*dx(metadata=femp.metadata) - dot(fem.t, fem.v_u)*ds

            # Calculate the weak form for non-local chain stretch
            fem.WF_lmbda_c_tilde = (fem.v_lmbda_c_tilde*fem.lmbda_c_tilde + mp.l_nl**2*dot(grad(fem.v_lmbda_c_tilde), grad(fem.lmbda_c_tilde)) - fem.v_lmbda_c_tilde*fem.lmbda_c)*dx(metadata=femp.metadata)

            # Calculate the overall weak form
            fem.WF = fem.WF_u + fem.WF_lmbda_c_tilde

            # Calculate the Gateaux derivative
            fem.Jac = derivative(fem.WF, fem.mixed_space, fem.dmixed_space)

        return fem
    
    def setup_u_solver(self, fem):
        """
        Setup the weak form solver for displacement
        """
        fem.problem_u = NonlinearVariationalProblem(fem.WF_u, fem.u, fem.bc_u, J=fem.Jac_u)
        fem.solver_u = NonlinearVariationalSolver(fem.problem_u)

        fem.solver_u.parameters.update(fem.dict_solver_u_parameters)
        info(fem.solver_u.parameters, True)

        return fem
    
    def setup_bounded_lmbda_c_tilde_solver(self, fem):
        """
        Setup the weak form solver for nonlocal chain stretch
        """
        fem.problem_bounded_lmbda_c_tilde = NonlinearVariationalProblem(fem.WF_lmbda_c_tilde, fem.lmbda_c_tilde, fem.bc_lmbda_c_tilde, J=fem.Jac_lmbda_c_tilde)
        # impose non-local chain stretch bounds
        fem.problem_bounded_lmbda_c_tilde.set_bounds(fem.lmbda_c_tilde_lb.vector(), fem.lmbda_c_tilde_ub.vector())

        fem.solver_bounded_lmbda_c_tilde = NonlinearVariationalSolver(fem.problem_bounded_lmbda_c_tilde)

        fem.solver_bounded_lmbda_c_tilde.parameters.update(fem.dict_solver_bounded_lmbda_c_tilde_parameters)
        info(fem.solver_bounded_lmbda_c_tilde.parameters, True)

        return fem
    
    def setup_unbounded_lmbda_c_tilde_solver(self, fem):
        """
        Setup the weak form solver for nonlocal chain stretch
        """
        fem.problem_unbounded_lmbda_c_tilde = NonlinearVariationalProblem(fem.WF_lmbda_c_tilde, fem.lmbda_c_tilde, fem.bc_lmbda_c_tilde, J=fem.Jac_lmbda_c_tilde)
        fem.solver_unbounded_lmbda_c_tilde = NonlinearVariationalSolver(fem.problem_unbounded_lmbda_c_tilde)

        fem.solver_unbounded_lmbda_c_tilde.parameters.update(fem.dict_solver_unbounded_lmbda_c_tilde_parameters)
        info(fem.solver_unbounded_lmbda_c_tilde.parameters, True)

        return fem
    
    def setup_bounded_monolithic_solver(self, fem):
        """
        Setup the monolithic weak form solver
        """
        fem.problem_bounded_monolithic = NonlinearVariationalProblem(fem.WF, fem.mixed_space, fem.bc_monolithic, J=fem.Jac)
        fem.solver_bounded_monolithic = NonlinearVariationalSolver(fem.problem_bounded_monolithic)

        fem.solver_bounded_monolithic.parameters.update(fem.dict_solver_bounded_monolithic_parameters)
        info(fem.solver_bounded_monolithic.parameters, True)

        # Need to figure out how to implement the bounds here!!!

        return fem
    
    def setup_unbounded_monolithic_solver(self, fem):
        """
        Setup the monolithic weak form solver
        """
        fem.problem_unbounded_monolithic = NonlinearVariationalProblem(fem.WF, fem.mixed_space, fem.bc_monolithic, J=fem.Jac)
        fem.solver_unbounded_monolithic = NonlinearVariationalSolver(fem.problem_unbounded_monolithic)

        fem.solver_unbounded_monolithic.parameters.update(fem.dict_solver_unbounded_monolithic_parameters)
        info(fem.solver_unbounded_monolithic.parameters, True)

        (fem.u, fem.lmbda_c_tilde) = fem.mixed_space.split()

        return fem
    
    def fenics_weak_form_solver_setup(self, fem):
        """
        Setup the weak form solver
        """
        if fem.solver_algorithm == "alternate_minimization":
            fem = self.setup_u_solver(fem)
            if fem.solver_bounded is True:
                fem = self.setup_bounded_lmbda_c_tilde_solver(fem)
            else:
                fem = self.setup_unbounded_lmbda_c_tilde_solver(fem)
        elif fem.solver_algorithm == "monolithic":
            if fem.solver_bounded is True:
                fem = self.setup_bounded_monolithic_solver(fem)
            else:
                fem = self.setup_unbounded_monolithic_solver(fem)

        return fem
    
    def solve_u(self, fem):
        """
        Solve the displacement problem
        """
        print("Displacement problem")
        (iter, converged) = fem.solver_u.solve()

        return fem
    
    def solve_bounded_lmbda_c_tilde(self, fem):
        """
        Solve the non-local chain stretch problem
        """
        print("Non-local chain stretch problem")
        (iter, converged) = fem.solver_bounded_lmbda_c_tilde.solve()

        return fem
    
    def solve_unbounded_lmbda_c_tilde(self, fem):
        """
        Solve the non-local chain stretch problem
        """
        print("Non-local chain stretch problem")
        (iter, converged) = fem.solver_unbounded_lmbda_c_tilde.solve()

        return fem
    
    def solve_bounded_monolithic(self, fem):
        """
        Solve the weak form
        """
        print("Displacement and non-local chain stretch monolithic problem")
        (iter, converged) = fem.solver_bounded_monolithic.solve()

        return fem
    
    def solve_unbounded_monolithic(self, fem):
        """
        Solve the weak form
        """
        print("Displacement and non-local chain stretch monolithic problem")
        (iter, converged) = fem.solver_unbounded_monolithic.solve()

        return fem
    
    def fenics_weak_form_solve_step(self, fem):
        """
        Solve the weak form
        """
        if fem.solver_algorithm == "alternate_minimization" and fem.solver_bounded is True:
            itrtn = 1
            error_lmbda_c_tilde = 1.

            while itrtn < self.itrtn_max_lmbda_c_tilde_val and error_lmbda_c_tilde > self.tol_lmbda_c_tilde_val:
                error_lmbda_c_tilde = 0.
                # solve for the displacement while holding non-local chain stretch fixed
                fem = self.solve_u(fem)
                # solve for the non-local chain stretch while holding displacement fixed
                fem = self.solve_bounded_lmbda_c_tilde(fem)
                # calculate the L-infinity error norm for the non-local chain stretch
                lmbda_c_tilde_diff = fem.lmbda_c_tilde.vector() - fem.lmbda_c_tilde_prior.vector()
                error_lmbda_c_tilde = lmbda_c_tilde_diff.norm('linf')
                # monitor the results
                print("Alternate minimization scheme: Iteration # {0:3d}; error = {1:>14.8f}".format(itrtn, error_lmbda_c_tilde))
                # update prior non-local chain stretch
                fem.lmbda_c_tilde_prior.assign(fem.lmbda_c_tilde)
                # update iteration
                itrtn += 1
        
        elif fem.solver_algorithm == "alternate_minimization" and fem.solver_bounded is False: pass
        elif fem.solver_algorithm == "monolithic" and fem.solver_bounded is True: pass
        elif fem.solver_algorithm == "monolithic" and fem.solver_bounded is False:
            fem = self.solve_unbounded_monolithic(fem)
        
        # update maximal non-local chain stretch to account for network irreversibility
        fem.lmbda_c_tilde_max = project(conditional(gt(fem.lmbda_c_tilde, fem.lmbda_c_tilde_max), fem.lmbda_c_tilde, fem.lmbda_c_tilde_max), fem.V_lmbda_c_tilde)

        return fem

    def fenics_weak_form_initialization(self, fem, parameters):
        gp      = parameters.geometry
        chunks  = SimpleNamespace()

        # initialize lists to zeros - necessary for irreversibility
        chunks = self.weak_form_initialize_deformation_sigma_chunks(gp.meshpoints, chunks)
        
        # lmbda_c
        chunks.lmbda_c_chunks     = []
        chunks.lmbda_c_chunks_val = [0. for meshpoint_indx in range(len(gp.meshpoints))]
        # lmbda_c_eq
        chunks.lmbda_c_eq_chunks     = []
        chunks.lmbda_c_eq_chunks_val = [[0. for nu_chunk_indx in range(len(self.nu_chunks_indx_list))] for meshpoint_indx in range(len(gp.meshpoints))]
        # lmbda_nu
        chunks.lmbda_nu_chunks     = []
        chunks.lmbda_nu_chunks_val = [[0. for nu_chunk_indx in range(len(self.nu_chunks_indx_list))] for meshpoint_indx in range(len(gp.meshpoints))]
        # lmbda_c_tilde
        chunks.lmbda_c_tilde_chunks     = []
        chunks.lmbda_c_tilde_chunks_val = [0. for meshpoint_indx in range(len(gp.meshpoints))]
        # lmbda_c_eq_tilde
        chunks.lmbda_c_eq_tilde_chunks     = []
        chunks.lmbda_c_eq_tilde_chunks_val = [[0. for nu_chunk_indx in range(len(self.nu_chunks_indx_list))] for meshpoint_indx in range(len(gp.meshpoints))]
        # lmbda_nu_tilde
        chunks.lmbda_nu_tilde_chunks     = []
        chunks.lmbda_nu_tilde_chunks_val = [[0. for nu_chunk_indx in range(len(self.nu_chunks_indx_list))] for meshpoint_indx in range(len(gp.meshpoints))]
        # lmbda_c_tilde_max
        chunks.lmbda_c_tilde_max_chunks     = []
        chunks.lmbda_c_tilde_max_chunks_val = [0. for meshpoint_indx in range(len(gp.meshpoints))]
        # lmbda_c_eq_tilde_max
        chunks.lmbda_c_eq_tilde_max_chunks     = []
        chunks.lmbda_c_eq_tilde_max_chunks_val = [[0. for nu_chunk_indx in range(len(self.nu_chunks_indx_list))] for meshpoint_indx in range(len(gp.meshpoints))]
        # lmbda_nu_tilde_max
        chunks.lmbda_nu_tilde_max_chunks     = []
        chunks.lmbda_nu_tilde_max_chunks_val = [[0. for nu_chunk_indx in range(len(self.nu_chunks_indx_list))] for meshpoint_indx in range(len(gp.meshpoints))]
        # upsilon_c
        chunks.upsilon_c_chunks     = []
        chunks.upsilon_c_chunks_val = [[0. for nu_chunk_indx in range(len(self.nu_chunks_indx_list))] for meshpoint_indx in range(len(gp.meshpoints))]
        # Upsilon_c
        chunks.Upsilon_c_chunks     = []
        chunks.Upsilon_c_chunks_val = [0. for meshpoint_indx in range(len(gp.meshpoints))]
        # d_c
        chunks.d_c_chunks     = []
        chunks.d_c_chunks_val = [[0. for nu_chunk_indx in range(len(self.nu_chunks_indx_list))] for meshpoint_indx in range(len(gp.meshpoints))]
        # D_c
        chunks.D_c_chunks     = []
        chunks.D_c_chunks_val = [0. for meshpoint_indx in range(len(gp.meshpoints))]
        # epsilon_cnu_diss_hat
        chunks.epsilon_cnu_diss_hat_chunks     = []
        chunks.epsilon_cnu_diss_hat_chunks_val = [[0. for nu_chunk_indx in range(len(self.nu_chunks_indx_list))] for meshpoint_indx in range(len(gp.meshpoints))]
        # Epsilon_cnu_diss_hat
        chunks.Epsilon_cnu_diss_hat_chunks     = []
        chunks.Epsilon_cnu_diss_hat_chunks_val = [0. for meshpoint_indx in range(len(gp.meshpoints))]
        # epsilon_c_diss_hat
        chunks.epsilon_c_diss_hat_chunks     = []
        chunks.epsilon_c_diss_hat_chunks_val = [[0. for nu_chunk_indx in range(len(self.nu_chunks_indx_list))] for meshpoint_indx in range(len(gp.meshpoints))]
        # Epsilon_c_diss_hat
        chunks.Epsilon_c_diss_hat_chunks     = []
        chunks.Epsilon_c_diss_hat_chunks_val = [0. for meshpoint_indx in range(len(gp.meshpoints))]
        # overline_epsilon_cnu_diss_hat
        chunks.overline_epsilon_cnu_diss_hat_chunks     = []
        chunks.overline_epsilon_cnu_diss_hat_chunks_val = [[0. for nu_chunk_indx in range(len(self.nu_chunks_indx_list))] for meshpoint_indx in range(len(gp.meshpoints))]
        # overline_Epsilon_cnu_diss_hat
        chunks.overline_Epsilon_cnu_diss_hat_chunks     = []
        chunks.overline_Epsilon_cnu_diss_hat_chunks_val = [0. for meshpoint_indx in range(len(gp.meshpoints))]
        # overline_epsilon_c_diss_hat
        chunks.overline_epsilon_c_diss_hat_chunks     = []
        chunks.overline_epsilon_c_diss_hat_chunks_val = [[0. for nu_chunk_indx in range(len(self.nu_chunks_indx_list))] for meshpoint_indx in range(len(gp.meshpoints))]
        # overline_Epsilon_c_diss_hat
        chunks.overline_Epsilon_c_diss_hat_chunks     = []
        chunks.overline_Epsilon_c_diss_hat_chunks_val = [0. for meshpoint_indx in range(len(gp.meshpoints))]

        return fem, chunks

    def fenics_weak_form_chunk_post_processing(self, deformation, chunks, fem, file_results, parameters):
        """
        Post-processing at the end of each time iteration chunk
        """
        gp   = parameters.geometry
        femp = parameters.fem
        ppp  = parameters.post_processing

        file_results.parameters["rewrite_function_mesh"] = ppp.rewrite_function_mesh
        file_results.parameters["flush_output"]          = ppp.flush_output
        file_results.parameters["functions_share_mesh"]  = ppp.functions_share_mesh

        # u
        if ppp.save_u:
            fem.u.rename("Displacement", "u")
            file_results.write(fem.u, deformation.t_val)
        
        # lmbda_c
        if ppp.save_lmbda_c_mesh:
            lmbda_c_val = project(fem.lmbda_c, fem.V_DG_scalar)
            lmbda_c_val.rename("Chain stretch", "lmbda_c_val")
            file_results.write(lmbda_c_val, deformation.t_val)
        
        if ppp.save_lmbda_c_chunks:
            lmbda_c_val = project(fem.lmbda_c, fem.V_DG_scalar)
            for meshpoint_indx in range(len(gp.meshpoints)):
                chunks.lmbda_c_chunks_val[meshpoint_indx] = lmbda_c_val(gp.meshpoints[meshpoint_indx])
            chunks.lmbda_c_chunks.append(deepcopy(chunks.lmbda_c_chunks_val))
        
        # lmbda_c_eq
        if ppp.save_lmbda_c_eq_mesh:
            for nu_chunk_indx in range(len(self.nu_chunks_indx_list)):
                nu_indx             = self.nu_chunks_indx_list[nu_chunk_indx]
                lmbda_c_eq___nu_val = project(self.lmbda_c_eq_ufl_fenics_mesh_func(nu_indx, fem), fem.V_DG_scalar)

                nu_str        = str(self.nu_list[nu_indx])
                name_str      = "Equilibrium chain stretch nu = "+nu_str
                parameter_str = "lmbda_c_eq___nu_"+nu_str+"_val"

                lmbda_c_eq___nu_val.rename(name_str, parameter_str)
                file_results.write(lmbda_c_eq___nu_val, deformation.t_val)
        
        if ppp.save_lmbda_c_eq_chunks:
            for meshpoint_indx in range(len(gp.meshpoints)):
                for nu_chunk_indx in range(len(self.nu_chunks_indx_list)):
                    nu_indx             = self.nu_chunks_indx_list[nu_chunk_indx]
                    lmbda_c_eq___nu_val = project(self.lmbda_c_eq_ufl_fenics_mesh_func(nu_indx, fem), fem.V_DG_scalar)
                    chunks.lmbda_c_eq_chunks_val[meshpoint_indx][nu_chunk_indx] = lmbda_c_eq___nu_val(gp.meshpoints[meshpoint_indx])
            chunks.lmbda_c_eq_chunks.append(deepcopy(chunks.lmbda_c_eq_chunks_val))
        
        # lmbda_nu
        if ppp.save_lmbda_nu_mesh:
            for nu_chunk_indx in range(len(self.nu_chunks_indx_list)):
                nu_indx            = self.nu_chunks_indx_list[nu_chunk_indx]
                lmbda_nu___nu_val  = project(self.lmbda_nu_ufl_fenics_mesh_func(nu_indx, fem), fem.V_DG_scalar)

                nu_str        = str(self.nu_list[nu_indx])
                name_str      = "Segment stretch nu = "+nu_str
                parameter_str = "lmbda_nu___nu_"+nu_str+"_val"

                lmbda_nu___nu_val.rename(name_str, parameter_str)
                file_results.write(lmbda_nu___nu_val, deformation.t_val)
        
        if ppp.save_lmbda_nu_chunks:
            for meshpoint_indx in range(len(gp.meshpoints)):
                for nu_chunk_indx in range(len(self.nu_chunks_indx_list)):
                    nu_indx           = self.nu_chunks_indx_list[nu_chunk_indx]
                    lmbda_nu___nu_val = project(self.lmbda_nu_ufl_fenics_mesh_func(nu_indx, fem), fem.V_DG_scalar)
                    chunks.lmbda_nu_chunks_val[meshpoint_indx][nu_chunk_indx] = lmbda_nu___nu_val(gp.meshpoints[meshpoint_indx])
            chunks.lmbda_nu_chunks.append(deepcopy(chunks.lmbda_nu_chunks_val))
        
        # lmbda_c_tilde
        if ppp.save_lmbda_c_tilde:
            fem.lmbda_c_tilde.rename("Non-local chain stretch", "lmbda_c_tilde")
            file_results.write(fem.lmbda_c_tilde, deformation.t_val)
        
        if ppp.save_lmbda_c_tilde_chunks:
            for meshpoint_indx in range(len(gp.meshpoints)):
                chunks.lmbda_c_tilde_chunks_val[meshpoint_indx] = fem.lmbda_c_tilde(gp.meshpoints[meshpoint_indx])
            chunks.lmbda_c_tilde_chunks.append(deepcopy(chunks.lmbda_c_tilde_chunks_val))
        
        # lmbda_c_eq_tilde
        if ppp.save_lmbda_c_eq_tilde_mesh:
            for nu_chunk_indx in range(len(self.nu_chunks_indx_list)):
                nu_indx             = self.nu_chunks_indx_list[nu_chunk_indx]
                lmbda_c_eq_tilde___nu_val = project(self.lmbda_c_eq_tilde_ufl_fenics_mesh_func(nu_indx, fem), fem.V_DG_scalar)

                nu_str        = str(self.nu_list[nu_indx])
                name_str      = "Non-local equilibrium chain stretch nu = "+nu_str
                parameter_str = "lmbda_c_eq_tilde___nu_"+nu_str+"_val"

                lmbda_c_eq_tilde___nu_val.rename(name_str, parameter_str)
                file_results.write(lmbda_c_eq_tilde___nu_val, deformation.t_val)
        
        if ppp.save_lmbda_c_eq_tilde_chunks:
            for meshpoint_indx in range(len(gp.meshpoints)):
                for nu_chunk_indx in range(len(self.nu_chunks_indx_list)):
                    nu_indx             = self.nu_chunks_indx_list[nu_chunk_indx]
                    lmbda_c_eq_tilde___nu_val = project(self.lmbda_c_eq_tilde_ufl_fenics_mesh_func(nu_indx, fem), fem.V_DG_scalar)
                    chunks.lmbda_c_eq_tilde_chunks_val[meshpoint_indx][nu_chunk_indx] = lmbda_c_eq_tilde___nu_val(gp.meshpoints[meshpoint_indx])
            chunks.lmbda_c_eq_tilde_chunks.append(deepcopy(chunks.lmbda_c_eq_tilde_chunks_val))
        
        # lmbda_nu_tilde
        if ppp.save_lmbda_nu_tilde_mesh:
            for nu_chunk_indx in range(len(self.nu_chunks_indx_list)):
                nu_indx            = self.nu_chunks_indx_list[nu_chunk_indx]
                lmbda_nu_tilde___nu_val  = project(self.lmbda_nu_tilde_ufl_fenics_mesh_func(nu_indx, fem), fem.V_DG_scalar)

                nu_str        = str(self.nu_list[nu_indx])
                name_str      = "Non-local segment stretch nu = "+nu_str
                parameter_str = "lmbda_nu_tilde___nu_"+nu_str+"_val"

                lmbda_nu_tilde___nu_val.rename(name_str, parameter_str)
                file_results.write(lmbda_nu_tilde___nu_val, deformation.t_val)
        
        if ppp.save_lmbda_nu_tilde_chunks:
            for meshpoint_indx in range(len(gp.meshpoints)):
                for nu_chunk_indx in range(len(self.nu_chunks_indx_list)):
                    nu_indx           = self.nu_chunks_indx_list[nu_chunk_indx]
                    lmbda_nu_tilde___nu_val = project(self.lmbda_nu_tilde_ufl_fenics_mesh_func(nu_indx, fem), fem.V_DG_scalar)
                    chunks.lmbda_nu_tilde_chunks_val[meshpoint_indx][nu_chunk_indx] = lmbda_nu_tilde___nu_val(gp.meshpoints[meshpoint_indx])
            chunks.lmbda_nu_tilde_chunks.append(deepcopy(chunks.lmbda_nu_tilde_chunks_val))
        
        # lmbda_c_tilde_max
        if ppp.save_lmbda_c_tilde_max:
            fem.lmbda_c_tilde_max.rename("Maximal non-local chain stretch", "lmbda_c_tilde_max")
            file_results.write(fem.lmbda_c_tilde_max, deformation.t_val)
        
        if ppp.save_lmbda_c_tilde_max_chunks:
            for meshpoint_indx in range(len(gp.meshpoints)):
                chunks.lmbda_c_tilde_max_chunks_val[meshpoint_indx] = fem.lmbda_c_tilde_max(gp.meshpoints[meshpoint_indx])
            chunks.lmbda_c_tilde_max_chunks.append(deepcopy(chunks.lmbda_c_tilde_max_chunks_val))
        
        # lmbda_c_eq_tilde_max
        if ppp.save_lmbda_c_eq_tilde_max_mesh:
            for nu_chunk_indx in range(len(self.nu_chunks_indx_list)):
                nu_indx             = self.nu_chunks_indx_list[nu_chunk_indx]
                lmbda_c_eq_tilde_max___nu_val = project(self.lmbda_c_eq_tilde_max_ufl_fenics_mesh_func(nu_indx, fem), fem.V_DG_scalar)

                nu_str        = str(self.nu_list[nu_indx])
                name_str      = "Maximal non-local equilibrium chain stretch nu = "+nu_str
                parameter_str = "lmbda_c_eq_tilde_max___nu_"+nu_str+"_val"

                lmbda_c_eq_tilde_max___nu_val.rename(name_str, parameter_str)
                file_results.write(lmbda_c_eq_tilde_max___nu_val, deformation.t_val)
        
        if ppp.save_lmbda_c_eq_tilde_max_chunks:
            for meshpoint_indx in range(len(gp.meshpoints)):
                for nu_chunk_indx in range(len(self.nu_chunks_indx_list)):
                    nu_indx             = self.nu_chunks_indx_list[nu_chunk_indx]
                    lmbda_c_eq_tilde_max___nu_val = project(self.lmbda_c_eq_tilde_max_ufl_fenics_mesh_func(nu_indx, fem), fem.V_DG_scalar)
                    chunks.lmbda_c_eq_tilde_max_chunks_val[meshpoint_indx][nu_chunk_indx] = lmbda_c_eq_tilde_max___nu_val(gp.meshpoints[meshpoint_indx])
            chunks.lmbda_c_eq_tilde_max_chunks.append(deepcopy(chunks.lmbda_c_eq_tilde_max_chunks_val))
        
        # lmbda_nu_tilde_max
        if ppp.save_lmbda_nu_tilde_max_mesh:
            for nu_chunk_indx in range(len(self.nu_chunks_indx_list)):
                nu_indx            = self.nu_chunks_indx_list[nu_chunk_indx]
                lmbda_nu_tilde_max___nu_val  = project(self.lmbda_nu_tilde_max_ufl_fenics_mesh_func(nu_indx, fem), fem.V_DG_scalar)

                nu_str        = str(self.nu_list[nu_indx])
                name_str      = "Maximal non-local segment stretch nu = "+nu_str
                parameter_str = "lmbda_nu_tilde_max___nu_"+nu_str+"_val"

                lmbda_nu_tilde_max___nu_val.rename(name_str, parameter_str)
                file_results.write(lmbda_nu_tilde_max___nu_val, deformation.t_val)
        
        if ppp.save_lmbda_nu_tilde_max_chunks:
            for meshpoint_indx in range(len(gp.meshpoints)):
                for nu_chunk_indx in range(len(self.nu_chunks_indx_list)):
                    nu_indx           = self.nu_chunks_indx_list[nu_chunk_indx]
                    lmbda_nu_tilde_max___nu_val = project(self.lmbda_nu_tilde_max_ufl_fenics_mesh_func(nu_indx, fem), fem.V_DG_scalar)
                    chunks.lmbda_nu_tilde_max_chunks_val[meshpoint_indx][nu_chunk_indx] = lmbda_nu_tilde_max___nu_val(gp.meshpoints[meshpoint_indx])
            chunks.lmbda_nu_tilde_max_chunks.append(deepcopy(chunks.lmbda_nu_tilde_max_chunks_val))

        # upsilon_c
        if ppp.save_upsilon_c_mesh:
            for nu_chunk_indx in range(len(self.nu_chunks_indx_list)):
                nu_indx            = self.nu_chunks_indx_list[nu_chunk_indx]
                upsilon_c___nu_val = project(self.upsilon_c_ufl_fenics_mesh_func(nu_indx, fem), fem.V_DG_scalar)

                nu_str        = str(self.nu_list[nu_indx])
                name_str      = "Chain survival nu = "+nu_str
                parameter_str = "upsilon_c___nu_"+nu_str+"_val"

                upsilon_c___nu_val.rename(name_str, parameter_str)
                file_results.write(upsilon_c___nu_val, deformation.t_val)
        
        if ppp.save_upsilon_c_chunks:
            for meshpoint_indx in range(len(gp.meshpoints)):
                for nu_chunk_indx in range(len(self.nu_chunks_indx_list)):
                    nu_indx            = self.nu_chunks_indx_list[nu_chunk_indx]
                    upsilon_c___nu_val = project(self.upsilon_c_ufl_fenics_mesh_func(nu_indx, fem), fem.V_DG_scalar)
                    chunks.upsilon_c_chunks_val[meshpoint_indx][nu_chunk_indx] = upsilon_c___nu_val(gp.meshpoints[meshpoint_indx])
            chunks.upsilon_c_chunks.append(deepcopy(chunks.upsilon_c_chunks_val))

        # Upsilon_c
        if ppp.save_Upsilon_c_mesh:
            Upsilon_c_val = project(self.Upsilon_c_ufl_fenics_mesh_func(fem), fem.V_DG_scalar)
            Upsilon_c_val.rename("Average chain survival", "Upsilon_c_val")
            file_results.write(Upsilon_c_val, deformation.t_val)
        
        if ppp.save_Upsilon_c_chunks:
            Upsilon_c_val = project(self.Upsilon_c_ufl_fenics_mesh_func(fem), fem.V_DG_scalar)
            for meshpoint_indx in range(len(gp.meshpoints)):
                chunks.Upsilon_c_chunks_val[meshpoint_indx] = Upsilon_c_val(gp.meshpoints[meshpoint_indx])
            chunks.Upsilon_c_chunks.append(deepcopy(chunks.Upsilon_c_chunks_val))
        
        # d_c
        if ppp.save_d_c_mesh:
            for nu_chunk_indx in range(len(self.nu_chunks_indx_list)):
                nu_indx      = self.nu_chunks_indx_list[nu_chunk_indx]
                d_c___nu_val = project(self.d_c_ufl_fenics_mesh_func(nu_indx, fem), fem.V_DG_scalar)
                
                nu_str        = str(self.nu_list[nu_indx])
                name_str      = "Chain damage nu = "+nu_str
                parameter_str = "d_c___nu_"+nu_str+"_val"

                d_c___nu_val.rename(name_str, parameter_str)
                file_results.write(d_c___nu_val, deformation.t_val)
        
        if ppp.save_d_c_chunks:
            for meshpoint_indx in range(len(gp.meshpoints)):
                for nu_chunk_indx in range(len(self.nu_chunks_indx_list)):
                    nu_indx      = self.nu_chunks_indx_list[nu_chunk_indx]
                    d_c___nu_val = project(self.d_c_ufl_fenics_mesh_func(nu_indx, fem), fem.V_DG_scalar)
                    chunks.d_c_chunks_val[meshpoint_indx][nu_chunk_indx] = d_c___nu_val(gp.meshpoints[meshpoint_indx])
            chunks.d_c_chunks.append(deepcopy(chunks.d_c_chunks_val))
        
        # D_c
        if ppp.save_D_c_mesh:
            D_c_val = project(self.D_c_ufl_fenics_mesh_func(fem), fem.V_DG_scalar)
            D_c_val.rename("Average chain damage", "D_c_val")
            file_results.write(D_c_val, deformation.t_val)
        
        if ppp.save_D_c_chunks:
            D_c_val = project(self.D_c_ufl_fenics_mesh_func(fem), fem.V_DG_scalar)
            for meshpoint_indx in range(len(gp.meshpoints)):
                chunks.D_c_chunks_val[meshpoint_indx] = D_c_val(gp.meshpoints[meshpoint_indx])
            chunks.D_c_chunks.append(deepcopy(chunks.D_c_chunks_val))
        
        # epsilon_cnu_diss_hat
        if ppp.save_epsilon_cnu_diss_hat_mesh:
            for nu_chunk_indx in range(len(self.nu_chunks_indx_list)):
                nu_indx = self.nu_chunks_indx_list[nu_chunk_indx]
                epsilon_cnu_diss_hat___nu_val = project(self.epsilon_cnu_diss_hat_ufl_fenics_mesh_func(nu_indx, fem), fem.V_DG_scalar)

                nu_str        = str(self.nu_list[nu_indx])
                name_str      = "Per segment nondimensional dissipated chain scission energy nu = "+nu_str
                parameter_str = "epsilon_cnu_diss_hat___nu_"+nu_str+"_val"

                epsilon_cnu_diss_hat___nu_val.rename(name_str, parameter_str)
                file_results.write(epsilon_cnu_diss_hat___nu_val, deformation.t_val)
        
        if ppp.save_epsilon_cnu_diss_hat_chunks:
            for meshpoint_indx in range(len(gp.meshpoints)):
                for nu_chunk_indx in range(len(self.nu_chunks_indx_list)):
                    nu_indx               = self.nu_chunks_indx_list[nu_chunk_indx]
                    epsilon_cnu_diss_hat___nu_val = project(self.epsilon_cnu_diss_hat_ufl_fenics_mesh_func(nu_indx, fem), fem.V_DG_scalar)
                    chunks.epsilon_cnu_diss_hat_chunks_val[meshpoint_indx][nu_chunk_indx] = epsilon_cnu_diss_hat___nu_val(gp.meshpoints[meshpoint_indx])
            chunks.epsilon_cnu_diss_hat_chunks.append(deepcopy(chunks.epsilon_cnu_diss_hat_chunks_val))
        
        # Epsilon_cnu_diss_hat
        if ppp.save_Epsilon_cnu_diss_hat_mesh:
            Epsilon_cnu_diss_hat_val = project(self.Epsilon_cnu_diss_hat_ufl_fenics_mesh_func(fem), fem.V_DG_scalar)
            Epsilon_cnu_diss_hat_val.rename("Average per segment nondimensional dissipated chain scission energy", "Epsilon_cnu_diss_hat_val")
            file_results.write(Epsilon_cnu_diss_hat_val, deformation.t_val)
        
        if ppp.save_Epsilon_cnu_diss_hat_chunks:
            Epsilon_cnu_diss_hat_val = project(self.Epsilon_cnu_diss_hat_ufl_fenics_mesh_func(fem), fem.V_DG_scalar)
            for meshpoint_indx in range(len(gp.meshpoints)):
                chunks.Epsilon_cnu_diss_hat_chunks_val[meshpoint_indx] = Epsilon_cnu_diss_hat_val(gp.meshpoints[meshpoint_indx])
            chunks.Epsilon_cnu_diss_hat_chunks.append(deepcopy(chunks.Epsilon_cnu_diss_hat_chunks_val))
        
        # epsilon_c_diss_hat
        if ppp.save_epsilon_c_diss_hat_mesh:
            for nu_chunk_indx in range(len(self.nu_chunks_indx_list)):
                nu_indx = self.nu_chunks_indx_list[nu_chunk_indx]
                epsilon_c_diss_hat___nu_val = project(self.epsilon_c_diss_hat_ufl_fenics_mesh_func(nu_indx, fem), fem.V_DG_scalar)

                nu_str        = str(self.nu_list[nu_indx])
                name_str      = "Nondimensional dissipated chain scission energy nu = "+nu_str
                parameter_str = "epsilon_c_diss_hat___nu_"+nu_str+"_val"

                epsilon_c_diss_hat___nu_val.rename(name_str, parameter_str)
                file_results.write(epsilon_c_diss_hat___nu_val, deformation.t_val)
        
        if ppp.save_epsilon_c_diss_hat_chunks:
            for meshpoint_indx in range(len(gp.meshpoints)):
                for nu_chunk_indx in range(len(self.nu_chunks_indx_list)):
                    nu_indx               = self.nu_chunks_indx_list[nu_chunk_indx]
                    epsilon_c_diss_hat___nu_val = project(self.epsilon_c_diss_hat_ufl_fenics_mesh_func(nu_indx, fem), fem.V_DG_scalar)
                    chunks.epsilon_c_diss_hat_chunks_val[meshpoint_indx][nu_chunk_indx] = epsilon_c_diss_hat___nu_val(gp.meshpoints[meshpoint_indx])
            chunks.epsilon_c_diss_hat_chunks.append(deepcopy(chunks.epsilon_c_diss_hat_chunks_val))
        
        # Epsilon_c_diss_hat
        if ppp.save_Epsilon_c_diss_hat_mesh:
            Epsilon_c_diss_hat_val = project(self.Epsilon_c_diss_hat_ufl_fenics_mesh_func(fem), fem.V_DG_scalar)
            Epsilon_c_diss_hat_val.rename("Average nondimensional dissipated chain scission energy", "Epsilon_c_diss_hat_val")
            file_results.write(Epsilon_c_diss_hat_val, deformation.t_val)
        
        if ppp.save_Epsilon_c_diss_hat_chunks:
            Epsilon_c_diss_hat_val = project(self.Epsilon_c_diss_hat_ufl_fenics_mesh_func(fem), fem.V_DG_scalar)
            for meshpoint_indx in range(len(gp.meshpoints)):
                chunks.Epsilon_c_diss_hat_chunks_val[meshpoint_indx] = Epsilon_c_diss_hat_val(gp.meshpoints[meshpoint_indx])
            chunks.Epsilon_c_diss_hat_chunks.append(deepcopy(chunks.Epsilon_c_diss_hat_chunks_val))
        
        # overline_epsilon_cnu_diss_hat
        if ppp.save_overline_epsilon_cnu_diss_hat_mesh:
            for nu_chunk_indx in range(len(self.nu_chunks_indx_list)):
                nu_indx = self.nu_chunks_indx_list[nu_chunk_indx]
                overline_epsilon_cnu_diss_hat___nu_val = project(self.overline_epsilon_cnu_diss_hat_ufl_fenics_mesh_func(nu_indx, fem), fem.V_DG_scalar)

                nu_str        = str(self.nu_list[nu_indx])
                name_str      = "Per segment nondimensional scaled dissipated chain scission energy nu = "+nu_str
                parameter_str = "overline_epsilon_cnu_diss_hat___nu_"+nu_str+"_val"

                overline_epsilon_cnu_diss_hat___nu_val.rename(name_str, parameter_str)
                file_results.write(overline_epsilon_cnu_diss_hat___nu_val, deformation.t_val)
        
        if ppp.save_overline_epsilon_cnu_diss_hat_chunks:
            for meshpoint_indx in range(len(gp.meshpoints)):
                for nu_chunk_indx in range(len(self.nu_chunks_indx_list)):
                    nu_indx               = self.nu_chunks_indx_list[nu_chunk_indx]
                    overline_epsilon_cnu_diss_hat___nu_val = project(self.overline_epsilon_cnu_diss_hat_ufl_fenics_mesh_func(nu_indx, fem), fem.V_DG_scalar)
                    chunks.overline_epsilon_cnu_diss_hat_chunks_val[meshpoint_indx][nu_chunk_indx] = overline_epsilon_cnu_diss_hat___nu_val(gp.meshpoints[meshpoint_indx])
            chunks.overline_epsilon_cnu_diss_hat_chunks.append(deepcopy(chunks.overline_epsilon_cnu_diss_hat_chunks_val))
        
        # overline_Epsilon_cnu_diss_hat
        if ppp.save_overline_Epsilon_cnu_diss_hat_mesh:
            overline_Epsilon_cnu_diss_hat_val = project(self.overline_Epsilon_cnu_diss_hat_ufl_fenics_mesh_func(fem), fem.V_DG_scalar)
            overline_Epsilon_cnu_diss_hat_val.rename("Average per segment nondimensional scaled dissipated chain scission energy", "overline_Epsilon_cnu_diss_hat_val")
            file_results.write(overline_Epsilon_cnu_diss_hat_val, deformation.t_val)
        
        if ppp.save_overline_Epsilon_cnu_diss_hat_chunks:
            overline_Epsilon_cnu_diss_hat_val = project(self.overline_Epsilon_cnu_diss_hat_ufl_fenics_mesh_func(fem), fem.V_DG_scalar)
            for meshpoint_indx in range(len(gp.meshpoints)):
                chunks.overline_Epsilon_cnu_diss_hat_chunks_val[meshpoint_indx] = overline_Epsilon_cnu_diss_hat_val(gp.meshpoints[meshpoint_indx])
            chunks.overline_Epsilon_cnu_diss_hat_chunks.append(deepcopy(chunks.overline_Epsilon_cnu_diss_hat_chunks_val))
        
        # overline_epsilon_c_diss_hat
        if ppp.save_overline_epsilon_c_diss_hat_mesh:
            for nu_chunk_indx in range(len(self.nu_chunks_indx_list)):
                nu_indx = self.nu_chunks_indx_list[nu_chunk_indx]
                overline_epsilon_c_diss_hat___nu_val = project(self.overline_epsilon_c_diss_hat_ufl_fenics_mesh_func(nu_indx, fem), fem.V_DG_scalar)

                nu_str        = str(self.nu_list[nu_indx])
                name_str      = "Nondimensional scaled dissipated chain scission energy nu = "+nu_str
                parameter_str = "overline_epsilon_c_diss_hat___nu_"+nu_str+"_val"

                overline_epsilon_c_diss_hat___nu_val.rename(name_str, parameter_str)
                file_results.write(overline_epsilon_c_diss_hat___nu_val, deformation.t_val)
        
        if ppp.save_overline_epsilon_c_diss_hat_chunks:
            for meshpoint_indx in range(len(gp.meshpoints)):
                for nu_chunk_indx in range(len(self.nu_chunks_indx_list)):
                    nu_indx               = self.nu_chunks_indx_list[nu_chunk_indx]
                    overline_epsilon_c_diss_hat___nu_val = project(self.overline_epsilon_c_diss_hat_ufl_fenics_mesh_func(nu_indx, fem), fem.V_DG_scalar)
                    chunks.overline_epsilon_c_diss_hat_chunks_val[meshpoint_indx][nu_chunk_indx] = overline_epsilon_c_diss_hat___nu_val(gp.meshpoints[meshpoint_indx])
            chunks.overline_epsilon_c_diss_hat_chunks.append(deepcopy(chunks.overline_epsilon_c_diss_hat_chunks_val))
        
        # overline_Epsilon_c_diss_hat
        if ppp.save_overline_Epsilon_c_diss_hat_mesh:
            overline_Epsilon_c_diss_hat_val = project(self.overline_Epsilon_c_diss_hat_ufl_fenics_mesh_func(fem), fem.V_DG_scalar)
            overline_Epsilon_c_diss_hat_val.rename("Average nondimensional scaled dissipated chain scission energy", "overline_Epsilon_c_diss_hat_val")
            file_results.write(overline_Epsilon_c_diss_hat_val, deformation.t_val)
        
        if ppp.save_overline_Epsilon_c_diss_hat_chunks:
            overline_Epsilon_c_diss_hat_val = project(self.overline_Epsilon_c_diss_hat_ufl_fenics_mesh_func(fem), fem.V_DG_scalar)
            for meshpoint_indx in range(len(gp.meshpoints)):
                chunks.overline_Epsilon_c_diss_hat_chunks_val[meshpoint_indx] = overline_Epsilon_c_diss_hat_val(gp.meshpoints[meshpoint_indx])
            chunks.overline_Epsilon_c_diss_hat_chunks.append(deepcopy(chunks.overline_Epsilon_c_diss_hat_chunks_val))
        
        # sigma
        if ppp.save_sigma_mesh:
            sigma_val = project(self.first_pk_stress_ufl_fenics_mesh_func(fem)/fem.J*fem.F.T, fem.V_DG_tensor)
            sigma_val.rename("Normalized Cauchy stress", "sigma_val")
            file_results.write(sigma_val, deformation.t_val)
        
        if ppp.save_sigma_chunks:
            sigma_val = project(self.first_pk_stress_ufl_fenics_mesh_func(fem)/fem.J*fem.F.T, fem.V_DG_tensor)
            chunks    = self.weak_form_store_calculated_sigma_chunks(sigma_val, femp.two_dim_tensor2vector_indx_dict, gp.meshpoints, chunks)
        
        # F
        if ppp.save_F_mesh:
            F_val = project(fem.F, fem.V_DG_tensor)
            F_val.rename("Deformation gradient", "F_val")
            file_results.write(F_val, deformation.t_val)
        
        if ppp.save_F_chunks:
            F_val  = project(fem.F, fem.V_DG_tensor)
            chunks = self.weak_form_store_calculated_deformation_chunks(F_val, femp.two_dim_tensor2vector_indx_dict, gp.meshpoints, chunks)
        
        return chunks
    
    def first_pk_stress_ufl_fenics_mesh_func(self, fem):
        second_pk_stress_val = Constant(0.0)*fem.I
        for nu_indx in range(self.nu_num):
            # determine equilibrium chain stretch and segement stretch
            nu_val              = self.nu_list[nu_indx]
            P_nu___nu_val       = self.P_nu_list[nu_indx]
            A_nu___nu_val       = self.A_nu_list[nu_indx]
            lmbda_c_eq___nu_val = fem.lmbda_c*A_nu___nu_val
            lmbda_nu___nu_val   = self.composite_ufjc_ufl_fenics_list[nu_indx].lmbda_nu_ufl_fenics_func(lmbda_c_eq___nu_val)
            xi_c___nu_val       = self.composite_ufjc_ufl_fenics_list[nu_indx].xi_c_ufl_fenics_func(lmbda_nu___nu_val, lmbda_c_eq___nu_val)
            # determine chain damage
            upsilon_c___nu_val = self.upsilon_c_ufl_fenics_mesh_func(nu_indx, fem)
            # determine stress response
            second_pk_stress_val += upsilon_c___nu_val*P_nu___nu_val*nu_val*A_nu___nu_val*xi_c___nu_val/(3.*fem.lmbda_c)*fem.F
        second_pk_stress_val += self.Upsilon_c_ufl_fenics_mesh_func(fem)**2*self.K_G*(fem.J-1)*fem.J*fem.F_inv.T
        return second_pk_stress_val
    
    def lmbda_c_eq_tilde_ufl_fenics_mesh_func(self, nu_indx, fem):
        A_nu___nu_val       = self.A_nu_list[nu_indx]
        lmbda_c_eq_tilde___nu_val = fem.lmbda_c_tilde*A_nu___nu_val
        return lmbda_c_eq_tilde___nu_val
    
    def lmbda_nu_tilde_ufl_fenics_mesh_func(self, nu_indx, fem):
        lmbda_c_eq_tilde___nu_val = self.lmbda_c_eq_tilde_ufl_fenics_mesh_func(nu_indx, fem)
        lmbda_nu_tilde___nu_val = self.composite_ufjc_ufl_fenics_list[nu_indx].lmbda_nu_ufl_fenics_func(lmbda_c_eq_tilde___nu_val)
        return lmbda_nu_tilde___nu_val
    
    def lmbda_c_eq_tilde_max_ufl_fenics_mesh_func(self, nu_indx, fem):
        A_nu___nu_val       = self.A_nu_list[nu_indx]
        lmbda_c_eq_tilde_max___nu_val = fem.lmbda_c_tilde_max*A_nu___nu_val
        return lmbda_c_eq_tilde_max___nu_val
    
    def lmbda_nu_tilde_max_ufl_fenics_mesh_func(self, nu_indx, fem):
        lmbda_c_eq_tilde_max___nu_val = self.lmbda_c_eq_tilde_max_ufl_fenics_mesh_func(nu_indx, fem)
        lmbda_nu_tilde_max___nu_val = self.composite_ufjc_ufl_fenics_list[nu_indx].lmbda_nu_ufl_fenics_func(lmbda_c_eq_tilde_max___nu_val)
        return lmbda_nu_tilde_max___nu_val
    
    def upsilon_c_ufl_fenics_mesh_func(self, nu_indx, fem):
        lmbda_nu_tilde_max___nu_val = self.lmbda_nu_tilde_max_ufl_fenics_mesh_func(nu_indx, fem)
        upsilon_c___nu_val      = (1.-self.k_cond_val)*self.composite_ufjc_ufl_fenics_list[nu_indx].p_c_sur_hat_ufl_fenics_func(lmbda_nu_tilde_max___nu_val) + self.k_cond_val
        return upsilon_c___nu_val
    
    def d_c_ufl_fenics_mesh_func(self, nu_indx, fem):
        return 1. - self.upsilon_c_ufl_fenics_mesh_func(nu_indx, fem)
    
    def Upsilon_c_ufl_fenics_mesh_func(self, fem):
        Upsilon_c_val = Constant(0.0)
        for nu_indx in range(self.nu_num):
            P_nu___nu_val      = self.P_nu_list[nu_indx]
            upsilon_c___nu_val = self.upsilon_c_ufl_fenics_mesh_func(nu_indx, fem)
            Upsilon_c_val      += P_nu___nu_val*upsilon_c___nu_val
        Upsilon_c_val = Upsilon_c_val/self.P_nu_sum
        return Upsilon_c_val

    def D_c_ufl_fenics_mesh_func(self, fem):
        D_c_val = Constant(0.0)
        for nu_indx in range(self.nu_num):
            P_nu___nu_val = self.P_nu_list[nu_indx]
            d_c___nu_val  = self.d_c_ufl_fenics_mesh_func(nu_indx, fem)
            D_c_val       += P_nu___nu_val*d_c___nu_val
        D_c_val = D_c_val/self.P_nu_sum
        return D_c_val
    
    def lmbda_c_eq_ufl_fenics_mesh_func(self, nu_indx, fem):
        A_nu___nu_val       = self.A_nu_list[nu_indx]
        lmbda_c_eq___nu_val = fem.lmbda_c*A_nu___nu_val
        return lmbda_c_eq___nu_val

    def lmbda_nu_ufl_fenics_mesh_func(self, nu_indx, fem):
        lmbda_c_eq___nu_val = self.lmbda_c_eq_ufl_fenics_mesh_func(nu_indx, fem)
        lmbda_nu___nu_val   = self.composite_ufjc_ufl_fenics_list[nu_indx].lmbda_nu_ufl_fenics_func(lmbda_c_eq___nu_val)
        return lmbda_nu___nu_val
    
    def epsilon_cnu_diss_hat_ufl_fenics_mesh_func(self, nu_indx, fem):
        lmbda_nu_tilde___nu_val = self.lmbda_nu_tilde_ufl_fenics_mesh_func(nu_indx, fem)
        epsilon_cnu_diss_hat___nu_val = self.composite_ufjc_ufl_fenics_list[nu_indx].epsilon_cnu_diss_hat_equiv_ufl_fenics_func(lmbda_nu_tilde___nu_val)
        return epsilon_cnu_diss_hat___nu_val
    
    def epsilon_c_diss_hat_ufl_fenics_mesh_func(self, nu_indx, fem):
        nu_val = self.nu_list[nu_indx]
        epsilon_cnu_diss_hat___nu_val = self.epsilon_cnu_diss_hat_ufl_fenics_mesh_func(nu_indx, fem)
        epsilon_c_diss_hat___nu_val = nu_val * epsilon_cnu_diss_hat___nu_val
        return epsilon_c_diss_hat___nu_val
    
    def overline_epsilon_cnu_diss_hat_ufl_fenics_mesh_func(self, nu_indx, fem):
        epsilon_cnu_diss_hat___nu_val = self.epsilon_cnu_diss_hat_ufl_fenics_mesh_func(nu_indx, fem)
        overline_epsilon_cnu_diss_hat___nu_val = epsilon_cnu_diss_hat___nu_val / self.zeta_nu_char
        return overline_epsilon_cnu_diss_hat___nu_val
    
    def overline_epsilon_c_diss_hat_ufl_fenics_mesh_func(self, nu_indx, fem):
        epsilon_c_diss_hat___nu_val = self.epsilon_c_diss_hat_ufl_fenics_mesh_func(nu_indx, fem)
        overline_epsilon_c_diss_hat___nu_val = epsilon_c_diss_hat___nu_val / self.zeta_nu_char
        return overline_epsilon_c_diss_hat___nu_val
    
    def Epsilon_cnu_diss_hat_ufl_fenics_mesh_func(self, fem):
        Epsilon_cnu_diss_hat_val = Constant(0.0)
        for nu_indx in range(self.nu_num):
            P_nu___nu_val = self.P_nu_list[nu_indx]
            epsilon_cnu_diss_hat___nu_val = self.epsilon_cnu_diss_hat_ufl_fenics_mesh_func(nu_indx, fem)
            Epsilon_cnu_diss_hat_val      += P_nu___nu_val*epsilon_cnu_diss_hat___nu_val
        Epsilon_cnu_diss_hat_val = Epsilon_cnu_diss_hat_val/self.P_nu_sum
        return Epsilon_cnu_diss_hat_val
    
    def Epsilon_c_diss_hat_ufl_fenics_mesh_func(self, fem):
        Epsilon_c_diss_hat_val = Constant(0.0)
        for nu_indx in range(self.nu_num):
            P_nu___nu_val = self.P_nu_list[nu_indx]
            epsilon_c_diss_hat___nu_val = self.epsilon_c_diss_hat_ufl_fenics_mesh_func(nu_indx, fem)
            Epsilon_c_diss_hat_val      += P_nu___nu_val*epsilon_c_diss_hat___nu_val
        Epsilon_c_diss_hat_val = Epsilon_c_diss_hat_val/self.P_nu_sum
        return Epsilon_c_diss_hat_val
    
    def overline_Epsilon_cnu_diss_hat_ufl_fenics_mesh_func(self, fem):
        overline_Epsilon_cnu_diss_hat_val = Constant(0.0)
        for nu_indx in range(self.nu_num):
            P_nu___nu_val = self.P_nu_list[nu_indx]
            overline_epsilon_cnu_diss_hat___nu_val = self.overline_epsilon_cnu_diss_hat_ufl_fenics_mesh_func(nu_indx, fem)
            overline_Epsilon_cnu_diss_hat_val      += P_nu___nu_val*overline_epsilon_cnu_diss_hat___nu_val
        overline_Epsilon_cnu_diss_hat_val = overline_Epsilon_cnu_diss_hat_val/self.P_nu_sum
        return overline_Epsilon_cnu_diss_hat_val
    
    def overline_Epsilon_c_diss_hat_ufl_fenics_mesh_func(self, fem):
        overline_Epsilon_c_diss_hat_val = Constant(0.0)
        for nu_indx in range(self.nu_num):
            P_nu___nu_val = self.P_nu_list[nu_indx]
            overline_epsilon_c_diss_hat___nu_val = self.overline_epsilon_c_diss_hat_ufl_fenics_mesh_func(nu_indx, fem)
            overline_Epsilon_c_diss_hat_val      += P_nu___nu_val*overline_epsilon_c_diss_hat___nu_val
        overline_Epsilon_c_diss_hat_val = overline_Epsilon_c_diss_hat_val/self.P_nu_sum
        return overline_Epsilon_c_diss_hat_val

class TwoDimensionalPlaneStrainNearlyIncompressibleNonaffineEightChainModelEqualStrainRateDependentCompositeuFJCNetwork:

    def __init__(self, parameters, strong_form_initialize_sigma_chunks,
                 lr_cg_deformation_gradient_func,
                 strong_form_calculate_sigma_func,
                 strong_form_store_calculated_sigma_chunks,
                 weak_form_initialize_deformation_sigma_chunks,
                 weak_form_store_calculated_sigma_chunks,
                 weak_form_store_calculated_deformation_chunks):
        
        # Retain specified functions for deformation-specific calculations
        self.strong_form_initialize_sigma_chunks = (
            strong_form_initialize_sigma_chunks
        )
        self.lr_cg_deformation_gradient_func = (
            lr_cg_deformation_gradient_func
        )
        self.strong_form_calculate_sigma_func  = (
            strong_form_calculate_sigma_func
        )
        self.strong_form_store_calculated_sigma_chunks = (
            strong_form_store_calculated_sigma_chunks
        )
        self.weak_form_initialize_deformation_sigma_chunks = (
            weak_form_initialize_deformation_sigma_chunks
        )
        self.weak_form_store_calculated_sigma_chunks = (
            weak_form_store_calculated_sigma_chunks
        )
        self.weak_form_store_calculated_deformation_chunks = (
            weak_form_store_calculated_deformation_chunks
        )

        CompositeuFJCNetwork.__init__(self, parameters)

class TwoDimensionalPlaneStrainNearlyIncompressibleNonaffineEightChainModelEqualForceRateIndependentCompositeuFJCNetwork:

    def __init__(self, parameters, strong_form_initialize_sigma_chunks,
                 lr_cg_deformation_gradient_func,
                 strong_form_calculate_sigma_func,
                 strong_form_store_calculated_sigma_chunks,
                 weak_form_initialize_deformation_sigma_chunks,
                 weak_form_store_calculated_sigma_chunks,
                 weak_form_store_calculated_deformation_chunks):
        
        # Retain specified functions for deformation-specific calculations
        self.strong_form_initialize_sigma_chunks = (
            strong_form_initialize_sigma_chunks
        )
        self.lr_cg_deformation_gradient_func = (
            lr_cg_deformation_gradient_func
        )
        self.strong_form_calculate_sigma_func  = (
            strong_form_calculate_sigma_func
        )
        self.strong_form_store_calculated_sigma_chunks = (
            strong_form_store_calculated_sigma_chunks
        )
        self.weak_form_initialize_deformation_sigma_chunks = (
            weak_form_initialize_deformation_sigma_chunks
        )
        self.weak_form_store_calculated_sigma_chunks = (
            weak_form_store_calculated_sigma_chunks
        )
        self.weak_form_store_calculated_deformation_chunks = (
            weak_form_store_calculated_deformation_chunks
        )

        CompositeuFJCNetwork.__init__(self, parameters)

class TwoDimensionalPlaneStrainNearlyIncompressibleNonaffineEightChainModelEqualForceRateDependentCompositeuFJCNetwork:

    def __init__(self, parameters, strong_form_initialize_sigma_chunks,
                 lr_cg_deformation_gradient_func,
                 strong_form_calculate_sigma_func,
                 strong_form_store_calculated_sigma_chunks,
                 weak_form_initialize_deformation_sigma_chunks,
                 weak_form_store_calculated_sigma_chunks,
                 weak_form_store_calculated_deformation_chunks):
        
        # Retain specified functions for deformation-specific calculations
        self.strong_form_initialize_sigma_chunks = (
            strong_form_initialize_sigma_chunks
        )
        self.lr_cg_deformation_gradient_func = (
            lr_cg_deformation_gradient_func
        )
        self.strong_form_calculate_sigma_func  = (
            strong_form_calculate_sigma_func
        )
        self.strong_form_store_calculated_sigma_chunks = (
            strong_form_store_calculated_sigma_chunks
        )
        self.weak_form_initialize_deformation_sigma_chunks = (
            weak_form_initialize_deformation_sigma_chunks
        )
        self.weak_form_store_calculated_sigma_chunks = (
            weak_form_store_calculated_sigma_chunks
        )
        self.weak_form_store_calculated_deformation_chunks = (
            weak_form_store_calculated_deformation_chunks
        )

        CompositeuFJCNetwork.__init__(self, parameters)

class TwoDimensionalPlaneStrainCompressibleNonaffineEightChainModelEqualStrainRateIndependentCompositeuFJCNetwork:

    def __init__(self, parameters, strong_form_initialize_sigma_chunks,
                 lr_cg_deformation_gradient_func,
                 strong_form_calculate_sigma_func,
                 strong_form_store_calculated_sigma_chunks,
                 weak_form_initialize_deformation_sigma_chunks,
                 weak_form_store_calculated_sigma_chunks,
                 weak_form_store_calculated_deformation_chunks):
        
        # Retain specified functions for deformation-specific calculations
        self.strong_form_initialize_sigma_chunks = (
            strong_form_initialize_sigma_chunks
        )
        self.lr_cg_deformation_gradient_func = (
            lr_cg_deformation_gradient_func
        )
        self.strong_form_calculate_sigma_func  = (
            strong_form_calculate_sigma_func
        )
        self.strong_form_store_calculated_sigma_chunks = (
            strong_form_store_calculated_sigma_chunks
        )
        self.weak_form_initialize_deformation_sigma_chunks = (
            weak_form_initialize_deformation_sigma_chunks
        )
        self.weak_form_store_calculated_sigma_chunks = (
            weak_form_store_calculated_sigma_chunks
        )
        self.weak_form_store_calculated_deformation_chunks = (
            weak_form_store_calculated_deformation_chunks
        )

        CompositeuFJCNetwork.__init__(self, parameters)

class TwoDimensionalPlaneStrainCompressibleNonaffineEightChainModelEqualStrainRateDependentCompositeuFJCNetwork:

    def __init__(self, parameters, strong_form_initialize_sigma_chunks,
                 lr_cg_deformation_gradient_func,
                 strong_form_calculate_sigma_func,
                 strong_form_store_calculated_sigma_chunks,
                 weak_form_initialize_deformation_sigma_chunks,
                 weak_form_store_calculated_sigma_chunks,
                 weak_form_store_calculated_deformation_chunks):
        
        # Retain specified functions for deformation-specific calculations
        self.strong_form_initialize_sigma_chunks = (
            strong_form_initialize_sigma_chunks
        )
        self.lr_cg_deformation_gradient_func = (
            lr_cg_deformation_gradient_func
        )
        self.strong_form_calculate_sigma_func  = (
            strong_form_calculate_sigma_func
        )
        self.strong_form_store_calculated_sigma_chunks = (
            strong_form_store_calculated_sigma_chunks
        )
        self.weak_form_initialize_deformation_sigma_chunks = (
            weak_form_initialize_deformation_sigma_chunks
        )
        self.weak_form_store_calculated_sigma_chunks = (
            weak_form_store_calculated_sigma_chunks
        )
        self.weak_form_store_calculated_deformation_chunks = (
            weak_form_store_calculated_deformation_chunks
        )

        CompositeuFJCNetwork.__init__(self, parameters)

class TwoDimensionalPlaneStrainCompressibleNonaffineEightChainModelEqualForceRateIndependentCompositeuFJCNetwork:

    def __init__(self, parameters, strong_form_initialize_sigma_chunks,
                 lr_cg_deformation_gradient_func,
                 strong_form_calculate_sigma_func,
                 strong_form_store_calculated_sigma_chunks,
                 weak_form_initialize_deformation_sigma_chunks,
                 weak_form_store_calculated_sigma_chunks,
                 weak_form_store_calculated_deformation_chunks):
        
        # Retain specified functions for deformation-specific calculations
        self.strong_form_initialize_sigma_chunks = (
            strong_form_initialize_sigma_chunks
        )
        self.lr_cg_deformation_gradient_func = (
            lr_cg_deformation_gradient_func
        )
        self.strong_form_calculate_sigma_func  = (
            strong_form_calculate_sigma_func
        )
        self.strong_form_store_calculated_sigma_chunks = (
            strong_form_store_calculated_sigma_chunks
        )
        self.weak_form_initialize_deformation_sigma_chunks = (
            weak_form_initialize_deformation_sigma_chunks
        )
        self.weak_form_store_calculated_sigma_chunks = (
            weak_form_store_calculated_sigma_chunks
        )
        self.weak_form_store_calculated_deformation_chunks = (
            weak_form_store_calculated_deformation_chunks
        )

        CompositeuFJCNetwork.__init__(self, parameters)

class TwoDimensionalPlaneStrainCompressibleNonaffineEightChainModelEqualForceRateDependentCompositeuFJCNetwork:

    def __init__(self, parameters, strong_form_initialize_sigma_chunks,
                 lr_cg_deformation_gradient_func,
                 strong_form_calculate_sigma_func,
                 strong_form_store_calculated_sigma_chunks,
                 weak_form_initialize_deformation_sigma_chunks,
                 weak_form_store_calculated_sigma_chunks,
                 weak_form_store_calculated_deformation_chunks):
        
        # Retain specified functions for deformation-specific calculations
        self.strong_form_initialize_sigma_chunks = (
            strong_form_initialize_sigma_chunks
        )
        self.lr_cg_deformation_gradient_func = (
            lr_cg_deformation_gradient_func
        )
        self.strong_form_calculate_sigma_func  = (
            strong_form_calculate_sigma_func
        )
        self.strong_form_store_calculated_sigma_chunks = (
            strong_form_store_calculated_sigma_chunks
        )
        self.weak_form_initialize_deformation_sigma_chunks = (
            weak_form_initialize_deformation_sigma_chunks
        )
        self.weak_form_store_calculated_sigma_chunks = (
            weak_form_store_calculated_sigma_chunks
        )
        self.weak_form_store_calculated_deformation_chunks = (
            weak_form_store_calculated_deformation_chunks
        )

        CompositeuFJCNetwork.__init__(self, parameters)

class TwoDimensionalGeneralizedPlaneStrainIncompressibleNonaffineEightChainModelEqualStrainRateIndependentCompositeuFJCNetwork:

    def __init__(self, parameters, strong_form_initialize_sigma_chunks,
                 lr_cg_deformation_gradient_func,
                 strong_form_calculate_sigma_func,
                 strong_form_store_calculated_sigma_chunks,
                 weak_form_initialize_deformation_sigma_chunks,
                 weak_form_store_calculated_sigma_chunks,
                 weak_form_store_calculated_deformation_chunks):
        
        # Retain specified functions for deformation-specific calculations
        self.strong_form_initialize_sigma_chunks = (
            strong_form_initialize_sigma_chunks
        )
        self.lr_cg_deformation_gradient_func = (
            lr_cg_deformation_gradient_func
        )
        self.strong_form_calculate_sigma_func  = (
            strong_form_calculate_sigma_func
        )
        self.strong_form_store_calculated_sigma_chunks = (
            strong_form_store_calculated_sigma_chunks
        )
        self.weak_form_initialize_deformation_sigma_chunks = (
            weak_form_initialize_deformation_sigma_chunks
        )
        self.weak_form_store_calculated_sigma_chunks = (
            weak_form_store_calculated_sigma_chunks
        )
        self.weak_form_store_calculated_deformation_chunks = (
            weak_form_store_calculated_deformation_chunks
        )

        CompositeuFJCNetwork.__init__(self, parameters)

class TwoDimensionalGeneralizedPlaneStrainIncompressibleNonaffineEightChainModelEqualStrainRateDependentCompositeuFJCNetwork:

    def __init__(self, parameters, strong_form_initialize_sigma_chunks,
                 lr_cg_deformation_gradient_func,
                 strong_form_calculate_sigma_func,
                 strong_form_store_calculated_sigma_chunks,
                 weak_form_initialize_deformation_sigma_chunks,
                 weak_form_store_calculated_sigma_chunks,
                 weak_form_store_calculated_deformation_chunks):
        
        # Retain specified functions for deformation-specific calculations
        self.strong_form_initialize_sigma_chunks = (
            strong_form_initialize_sigma_chunks
        )
        self.lr_cg_deformation_gradient_func = (
            lr_cg_deformation_gradient_func
        )
        self.strong_form_calculate_sigma_func  = (
            strong_form_calculate_sigma_func
        )
        self.strong_form_store_calculated_sigma_chunks = (
            strong_form_store_calculated_sigma_chunks
        )
        self.weak_form_initialize_deformation_sigma_chunks = (
            weak_form_initialize_deformation_sigma_chunks
        )
        self.weak_form_store_calculated_sigma_chunks = (
            weak_form_store_calculated_sigma_chunks
        )
        self.weak_form_store_calculated_deformation_chunks = (
            weak_form_store_calculated_deformation_chunks
        )

        CompositeuFJCNetwork.__init__(self, parameters)

class TwoDimensionalGeneralizedPlaneStrainIncompressibleNonaffineEightChainModelEqualForceRateIndependentCompositeuFJCNetwork:

    def __init__(self, parameters, strong_form_initialize_sigma_chunks,
                 lr_cg_deformation_gradient_func,
                 strong_form_calculate_sigma_func,
                 strong_form_store_calculated_sigma_chunks,
                 weak_form_initialize_deformation_sigma_chunks,
                 weak_form_store_calculated_sigma_chunks,
                 weak_form_store_calculated_deformation_chunks):
        
        # Retain specified functions for deformation-specific calculations
        self.strong_form_initialize_sigma_chunks = (
            strong_form_initialize_sigma_chunks
        )
        self.lr_cg_deformation_gradient_func = (
            lr_cg_deformation_gradient_func
        )
        self.strong_form_calculate_sigma_func  = (
            strong_form_calculate_sigma_func
        )
        self.strong_form_store_calculated_sigma_chunks = (
            strong_form_store_calculated_sigma_chunks
        )
        self.weak_form_initialize_deformation_sigma_chunks = (
            weak_form_initialize_deformation_sigma_chunks
        )
        self.weak_form_store_calculated_sigma_chunks = (
            weak_form_store_calculated_sigma_chunks
        )
        self.weak_form_store_calculated_deformation_chunks = (
            weak_form_store_calculated_deformation_chunks
        )

        CompositeuFJCNetwork.__init__(self, parameters)

class TwoDimensionalGeneralizedPlaneStrainIncompressibleNonaffineEightChainModelEqualForceRateDependentCompositeuFJCNetwork:

    def __init__(self, parameters, strong_form_initialize_sigma_chunks,
                 lr_cg_deformation_gradient_func,
                 strong_form_calculate_sigma_func,
                 strong_form_store_calculated_sigma_chunks,
                 weak_form_initialize_deformation_sigma_chunks,
                 weak_form_store_calculated_sigma_chunks,
                 weak_form_store_calculated_deformation_chunks):
        
        # Retain specified functions for deformation-specific calculations
        self.strong_form_initialize_sigma_chunks = (
            strong_form_initialize_sigma_chunks
        )
        self.lr_cg_deformation_gradient_func = (
            lr_cg_deformation_gradient_func
        )
        self.strong_form_calculate_sigma_func  = (
            strong_form_calculate_sigma_func
        )
        self.strong_form_store_calculated_sigma_chunks = (
            strong_form_store_calculated_sigma_chunks
        )
        self.weak_form_initialize_deformation_sigma_chunks = (
            weak_form_initialize_deformation_sigma_chunks
        )
        self.weak_form_store_calculated_sigma_chunks = (
            weak_form_store_calculated_sigma_chunks
        )
        self.weak_form_store_calculated_deformation_chunks = (
            weak_form_store_calculated_deformation_chunks
        )

        CompositeuFJCNetwork.__init__(self, parameters)

class TwoDimensionalGeneralizedPlaneStrainNearlyIncompressibleNonaffineEightChainModelEqualStrainRateIndependentCompositeuFJCNetwork:

    def __init__(self, parameters, strong_form_initialize_sigma_chunks,
                 lr_cg_deformation_gradient_func,
                 strong_form_calculate_sigma_func,
                 strong_form_store_calculated_sigma_chunks,
                 weak_form_initialize_deformation_sigma_chunks,
                 weak_form_store_calculated_sigma_chunks,
                 weak_form_store_calculated_deformation_chunks):
        
        # Retain specified functions for deformation-specific calculations
        self.strong_form_initialize_sigma_chunks = (
            strong_form_initialize_sigma_chunks
        )
        self.lr_cg_deformation_gradient_func = (
            lr_cg_deformation_gradient_func
        )
        self.strong_form_calculate_sigma_func  = (
            strong_form_calculate_sigma_func
        )
        self.strong_form_store_calculated_sigma_chunks = (
            strong_form_store_calculated_sigma_chunks
        )
        self.weak_form_initialize_deformation_sigma_chunks = (
            weak_form_initialize_deformation_sigma_chunks
        )
        self.weak_form_store_calculated_sigma_chunks = (
            weak_form_store_calculated_sigma_chunks
        )
        self.weak_form_store_calculated_deformation_chunks = (
            weak_form_store_calculated_deformation_chunks
        )

        CompositeuFJCNetwork.__init__(self, parameters)

class TwoDimensionalGeneralizedPlaneStrainNearlyIncompressibleNonaffineEightChainModelEqualStrainRateDependentCompositeuFJCNetwork:

    def __init__(self, parameters, strong_form_initialize_sigma_chunks,
                 lr_cg_deformation_gradient_func,
                 strong_form_calculate_sigma_func,
                 strong_form_store_calculated_sigma_chunks,
                 weak_form_initialize_deformation_sigma_chunks,
                 weak_form_store_calculated_sigma_chunks,
                 weak_form_store_calculated_deformation_chunks):
        
        # Retain specified functions for deformation-specific calculations
        self.strong_form_initialize_sigma_chunks = (
            strong_form_initialize_sigma_chunks
        )
        self.lr_cg_deformation_gradient_func = (
            lr_cg_deformation_gradient_func
        )
        self.strong_form_calculate_sigma_func  = (
            strong_form_calculate_sigma_func
        )
        self.strong_form_store_calculated_sigma_chunks = (
            strong_form_store_calculated_sigma_chunks
        )
        self.weak_form_initialize_deformation_sigma_chunks = (
            weak_form_initialize_deformation_sigma_chunks
        )
        self.weak_form_store_calculated_sigma_chunks = (
            weak_form_store_calculated_sigma_chunks
        )
        self.weak_form_store_calculated_deformation_chunks = (
            weak_form_store_calculated_deformation_chunks
        )

        CompositeuFJCNetwork.__init__(self, parameters)

class TwoDimensionalGeneralizedPlaneStrainNearlyIncompressibleNonaffineEightChainModelEqualForceRateIndependentCompositeuFJCNetwork:

    def __init__(self, parameters, strong_form_initialize_sigma_chunks,
                 lr_cg_deformation_gradient_func,
                 strong_form_calculate_sigma_func,
                 strong_form_store_calculated_sigma_chunks,
                 weak_form_initialize_deformation_sigma_chunks,
                 weak_form_store_calculated_sigma_chunks,
                 weak_form_store_calculated_deformation_chunks):
        
        # Retain specified functions for deformation-specific calculations
        self.strong_form_initialize_sigma_chunks = (
            strong_form_initialize_sigma_chunks
        )
        self.lr_cg_deformation_gradient_func = (
            lr_cg_deformation_gradient_func
        )
        self.strong_form_calculate_sigma_func  = (
            strong_form_calculate_sigma_func
        )
        self.strong_form_store_calculated_sigma_chunks = (
            strong_form_store_calculated_sigma_chunks
        )
        self.weak_form_initialize_deformation_sigma_chunks = (
            weak_form_initialize_deformation_sigma_chunks
        )
        self.weak_form_store_calculated_sigma_chunks = (
            weak_form_store_calculated_sigma_chunks
        )
        self.weak_form_store_calculated_deformation_chunks = (
            weak_form_store_calculated_deformation_chunks
        )

        CompositeuFJCNetwork.__init__(self, parameters)

class TwoDimensionalGeneralizedPlaneStrainNearlyIncompressibleNonaffineEightChainModelEqualForceRateDependentCompositeuFJCNetwork:

    def __init__(self, parameters, strong_form_initialize_sigma_chunks,
                 lr_cg_deformation_gradient_func,
                 strong_form_calculate_sigma_func,
                 strong_form_store_calculated_sigma_chunks,
                 weak_form_initialize_deformation_sigma_chunks,
                 weak_form_store_calculated_sigma_chunks,
                 weak_form_store_calculated_deformation_chunks):
        
        # Retain specified functions for deformation-specific calculations
        self.strong_form_initialize_sigma_chunks = (
            strong_form_initialize_sigma_chunks
        )
        self.lr_cg_deformation_gradient_func = (
            lr_cg_deformation_gradient_func
        )
        self.strong_form_calculate_sigma_func  = (
            strong_form_calculate_sigma_func
        )
        self.strong_form_store_calculated_sigma_chunks = (
            strong_form_store_calculated_sigma_chunks
        )
        self.weak_form_initialize_deformation_sigma_chunks = (
            weak_form_initialize_deformation_sigma_chunks
        )
        self.weak_form_store_calculated_sigma_chunks = (
            weak_form_store_calculated_sigma_chunks
        )
        self.weak_form_store_calculated_deformation_chunks = (
            weak_form_store_calculated_deformation_chunks
        )

        CompositeuFJCNetwork.__init__(self, parameters)

class TwoDimensionalGeneralizedPlaneStrainCompressibleNonaffineEightChainModelEqualStrainRateIndependentCompositeuFJCNetwork:

    def __init__(self, parameters, strong_form_initialize_sigma_chunks,
                 lr_cg_deformation_gradient_func,
                 strong_form_calculate_sigma_func,
                 strong_form_store_calculated_sigma_chunks,
                 weak_form_initialize_deformation_sigma_chunks,
                 weak_form_store_calculated_sigma_chunks,
                 weak_form_store_calculated_deformation_chunks):
        
        # Retain specified functions for deformation-specific calculations
        self.strong_form_initialize_sigma_chunks = (
            strong_form_initialize_sigma_chunks
        )
        self.lr_cg_deformation_gradient_func = (
            lr_cg_deformation_gradient_func
        )
        self.strong_form_calculate_sigma_func  = (
            strong_form_calculate_sigma_func
        )
        self.strong_form_store_calculated_sigma_chunks = (
            strong_form_store_calculated_sigma_chunks
        )
        self.weak_form_initialize_deformation_sigma_chunks = (
            weak_form_initialize_deformation_sigma_chunks
        )
        self.weak_form_store_calculated_sigma_chunks = (
            weak_form_store_calculated_sigma_chunks
        )
        self.weak_form_store_calculated_deformation_chunks = (
            weak_form_store_calculated_deformation_chunks
        )

        CompositeuFJCNetwork.__init__(self, parameters)

class TwoDimensionalGeneralizedPlaneStrainCompressibleNonaffineEightChainModelEqualStrainRateDependentCompositeuFJCNetwork:

    def __init__(self, parameters, strong_form_initialize_sigma_chunks,
                 lr_cg_deformation_gradient_func,
                 strong_form_calculate_sigma_func,
                 strong_form_store_calculated_sigma_chunks,
                 weak_form_initialize_deformation_sigma_chunks,
                 weak_form_store_calculated_sigma_chunks,
                 weak_form_store_calculated_deformation_chunks):
        
        # Retain specified functions for deformation-specific calculations
        self.strong_form_initialize_sigma_chunks = (
            strong_form_initialize_sigma_chunks
        )
        self.lr_cg_deformation_gradient_func = (
            lr_cg_deformation_gradient_func
        )
        self.strong_form_calculate_sigma_func  = (
            strong_form_calculate_sigma_func
        )
        self.strong_form_store_calculated_sigma_chunks = (
            strong_form_store_calculated_sigma_chunks
        )
        self.weak_form_initialize_deformation_sigma_chunks = (
            weak_form_initialize_deformation_sigma_chunks
        )
        self.weak_form_store_calculated_sigma_chunks = (
            weak_form_store_calculated_sigma_chunks
        )
        self.weak_form_store_calculated_deformation_chunks = (
            weak_form_store_calculated_deformation_chunks
        )

        CompositeuFJCNetwork.__init__(self, parameters)

class TwoDimensionalGeneralizedPlaneStrainCompressibleNonaffineEightChainModelEqualForceRateIndependentCompositeuFJCNetwork:

    def __init__(self, parameters, strong_form_initialize_sigma_chunks,
                 lr_cg_deformation_gradient_func,
                 strong_form_calculate_sigma_func,
                 strong_form_store_calculated_sigma_chunks,
                 weak_form_initialize_deformation_sigma_chunks,
                 weak_form_store_calculated_sigma_chunks,
                 weak_form_store_calculated_deformation_chunks):
        
        # Retain specified functions for deformation-specific calculations
        self.strong_form_initialize_sigma_chunks = (
            strong_form_initialize_sigma_chunks
        )
        self.lr_cg_deformation_gradient_func = (
            lr_cg_deformation_gradient_func
        )
        self.strong_form_calculate_sigma_func  = (
            strong_form_calculate_sigma_func
        )
        self.strong_form_store_calculated_sigma_chunks = (
            strong_form_store_calculated_sigma_chunks
        )
        self.weak_form_initialize_deformation_sigma_chunks = (
            weak_form_initialize_deformation_sigma_chunks
        )
        self.weak_form_store_calculated_sigma_chunks = (
            weak_form_store_calculated_sigma_chunks
        )
        self.weak_form_store_calculated_deformation_chunks = (
            weak_form_store_calculated_deformation_chunks
        )

        CompositeuFJCNetwork.__init__(self, parameters)

class TwoDimensionalGeneralizedPlaneStrainCompressibleNonaffineEightChainModelEqualForceRateDependentCompositeuFJCNetwork:

    def __init__(self, parameters, strong_form_initialize_sigma_chunks,
                 lr_cg_deformation_gradient_func,
                 strong_form_calculate_sigma_func,
                 strong_form_store_calculated_sigma_chunks,
                 weak_form_initialize_deformation_sigma_chunks,
                 weak_form_store_calculated_sigma_chunks,
                 weak_form_store_calculated_deformation_chunks):
        
        # Retain specified functions for deformation-specific calculations
        self.strong_form_initialize_sigma_chunks = (
            strong_form_initialize_sigma_chunks
        )
        self.lr_cg_deformation_gradient_func = (
            lr_cg_deformation_gradient_func
        )
        self.strong_form_calculate_sigma_func  = (
            strong_form_calculate_sigma_func
        )
        self.strong_form_store_calculated_sigma_chunks = (
            strong_form_store_calculated_sigma_chunks
        )
        self.weak_form_initialize_deformation_sigma_chunks = (
            weak_form_initialize_deformation_sigma_chunks
        )
        self.weak_form_store_calculated_sigma_chunks = (
            weak_form_store_calculated_sigma_chunks
        )
        self.weak_form_store_calculated_deformation_chunks = (
            weak_form_store_calculated_deformation_chunks
        )

        CompositeuFJCNetwork.__init__(self, parameters)

class TwoDimensionalPlaneStressIncompressibleNonaffineEightChainModelEqualStrainRateIndependentCompositeuFJCNetwork:

    def __init__(self, parameters, strong_form_initialize_sigma_chunks,
                 lr_cg_deformation_gradient_func,
                 strong_form_calculate_sigma_func,
                 strong_form_store_calculated_sigma_chunks,
                 weak_form_initialize_deformation_sigma_chunks,
                 weak_form_store_calculated_sigma_chunks,
                 weak_form_store_calculated_deformation_chunks):
        
        # Retain specified functions for deformation-specific calculations
        self.strong_form_initialize_sigma_chunks = (
            strong_form_initialize_sigma_chunks
        )
        self.lr_cg_deformation_gradient_func = (
            lr_cg_deformation_gradient_func
        )
        self.strong_form_calculate_sigma_func  = (
            strong_form_calculate_sigma_func
        )
        self.strong_form_store_calculated_sigma_chunks = (
            strong_form_store_calculated_sigma_chunks
        )
        self.weak_form_initialize_deformation_sigma_chunks = (
            weak_form_initialize_deformation_sigma_chunks
        )
        self.weak_form_store_calculated_sigma_chunks = (
            weak_form_store_calculated_sigma_chunks
        )
        self.weak_form_store_calculated_deformation_chunks = (
            weak_form_store_calculated_deformation_chunks
        )

        CompositeuFJCNetwork.__init__(self, parameters)

class TwoDimensionalPlaneStressIncompressibleNonaffineEightChainModelEqualStrainRateDependentCompositeuFJCNetwork:

    def __init__(self, parameters, strong_form_initialize_sigma_chunks,
                 lr_cg_deformation_gradient_func,
                 strong_form_calculate_sigma_func,
                 strong_form_store_calculated_sigma_chunks,
                 weak_form_initialize_deformation_sigma_chunks,
                 weak_form_store_calculated_sigma_chunks,
                 weak_form_store_calculated_deformation_chunks):
        
        # Retain specified functions for deformation-specific calculations
        self.strong_form_initialize_sigma_chunks = (
            strong_form_initialize_sigma_chunks
        )
        self.lr_cg_deformation_gradient_func = (
            lr_cg_deformation_gradient_func
        )
        self.strong_form_calculate_sigma_func  = (
            strong_form_calculate_sigma_func
        )
        self.strong_form_store_calculated_sigma_chunks = (
            strong_form_store_calculated_sigma_chunks
        )
        self.weak_form_initialize_deformation_sigma_chunks = (
            weak_form_initialize_deformation_sigma_chunks
        )
        self.weak_form_store_calculated_sigma_chunks = (
            weak_form_store_calculated_sigma_chunks
        )
        self.weak_form_store_calculated_deformation_chunks = (
            weak_form_store_calculated_deformation_chunks
        )

        CompositeuFJCNetwork.__init__(self, parameters)

class TwoDimensionalPlaneStressIncompressibleNonaffineEightChainModelEqualForceRateIndependentCompositeuFJCNetwork:

    def __init__(self, parameters, strong_form_initialize_sigma_chunks,
                 lr_cg_deformation_gradient_func,
                 strong_form_calculate_sigma_func,
                 strong_form_store_calculated_sigma_chunks,
                 weak_form_initialize_deformation_sigma_chunks,
                 weak_form_store_calculated_sigma_chunks,
                 weak_form_store_calculated_deformation_chunks):
        
        # Retain specified functions for deformation-specific calculations
        self.strong_form_initialize_sigma_chunks = (
            strong_form_initialize_sigma_chunks
        )
        self.lr_cg_deformation_gradient_func = (
            lr_cg_deformation_gradient_func
        )
        self.strong_form_calculate_sigma_func  = (
            strong_form_calculate_sigma_func
        )
        self.strong_form_store_calculated_sigma_chunks = (
            strong_form_store_calculated_sigma_chunks
        )
        self.weak_form_initialize_deformation_sigma_chunks = (
            weak_form_initialize_deformation_sigma_chunks
        )
        self.weak_form_store_calculated_sigma_chunks = (
            weak_form_store_calculated_sigma_chunks
        )
        self.weak_form_store_calculated_deformation_chunks = (
            weak_form_store_calculated_deformation_chunks
        )

        CompositeuFJCNetwork.__init__(self, parameters)

class TwoDimensionalPlaneStressIncompressibleNonaffineEightChainModelEqualForceRateDependentCompositeuFJCNetwork:

    def __init__(self, parameters, strong_form_initialize_sigma_chunks,
                 lr_cg_deformation_gradient_func,
                 strong_form_calculate_sigma_func,
                 strong_form_store_calculated_sigma_chunks,
                 weak_form_initialize_deformation_sigma_chunks,
                 weak_form_store_calculated_sigma_chunks,
                 weak_form_store_calculated_deformation_chunks):
        
        # Retain specified functions for deformation-specific calculations
        self.strong_form_initialize_sigma_chunks = (
            strong_form_initialize_sigma_chunks
        )
        self.lr_cg_deformation_gradient_func = (
            lr_cg_deformation_gradient_func
        )
        self.strong_form_calculate_sigma_func  = (
            strong_form_calculate_sigma_func
        )
        self.strong_form_store_calculated_sigma_chunks = (
            strong_form_store_calculated_sigma_chunks
        )
        self.weak_form_initialize_deformation_sigma_chunks = (
            weak_form_initialize_deformation_sigma_chunks
        )
        self.weak_form_store_calculated_sigma_chunks = (
            weak_form_store_calculated_sigma_chunks
        )
        self.weak_form_store_calculated_deformation_chunks = (
            weak_form_store_calculated_deformation_chunks
        )

        CompositeuFJCNetwork.__init__(self, parameters)

class TwoDimensionalPlaneStressNearlyIncompressibleNonaffineEightChainModelEqualStrainRateIndependentCompositeuFJCNetwork:

    def __init__(self, parameters, strong_form_initialize_sigma_chunks,
                 lr_cg_deformation_gradient_func,
                 strong_form_calculate_sigma_func,
                 strong_form_store_calculated_sigma_chunks,
                 weak_form_initialize_deformation_sigma_chunks,
                 weak_form_store_calculated_sigma_chunks,
                 weak_form_store_calculated_deformation_chunks):
        
        # Retain specified functions for deformation-specific calculations
        self.strong_form_initialize_sigma_chunks = (
            strong_form_initialize_sigma_chunks
        )
        self.lr_cg_deformation_gradient_func = (
            lr_cg_deformation_gradient_func
        )
        self.strong_form_calculate_sigma_func  = (
            strong_form_calculate_sigma_func
        )
        self.strong_form_store_calculated_sigma_chunks = (
            strong_form_store_calculated_sigma_chunks
        )
        self.weak_form_initialize_deformation_sigma_chunks = (
            weak_form_initialize_deformation_sigma_chunks
        )
        self.weak_form_store_calculated_sigma_chunks = (
            weak_form_store_calculated_sigma_chunks
        )
        self.weak_form_store_calculated_deformation_chunks = (
            weak_form_store_calculated_deformation_chunks
        )

        CompositeuFJCNetwork.__init__(self, parameters)

class TwoDimensionalPlaneStressNearlyIncompressibleNonaffineEightChainModelEqualStrainRateDependentCompositeuFJCNetwork:

    def __init__(self, parameters, strong_form_initialize_sigma_chunks,
                 lr_cg_deformation_gradient_func,
                 strong_form_calculate_sigma_func,
                 strong_form_store_calculated_sigma_chunks,
                 weak_form_initialize_deformation_sigma_chunks,
                 weak_form_store_calculated_sigma_chunks,
                 weak_form_store_calculated_deformation_chunks):
        
        # Retain specified functions for deformation-specific calculations
        self.strong_form_initialize_sigma_chunks = (
            strong_form_initialize_sigma_chunks
        )
        self.lr_cg_deformation_gradient_func = (
            lr_cg_deformation_gradient_func
        )
        self.strong_form_calculate_sigma_func  = (
            strong_form_calculate_sigma_func
        )
        self.strong_form_store_calculated_sigma_chunks = (
            strong_form_store_calculated_sigma_chunks
        )
        self.weak_form_initialize_deformation_sigma_chunks = (
            weak_form_initialize_deformation_sigma_chunks
        )
        self.weak_form_store_calculated_sigma_chunks = (
            weak_form_store_calculated_sigma_chunks
        )
        self.weak_form_store_calculated_deformation_chunks = (
            weak_form_store_calculated_deformation_chunks
        )

        CompositeuFJCNetwork.__init__(self, parameters)

class TwoDimensionalPlaneStressNearlyIncompressibleNonaffineEightChainModelEqualForceRateIndependentCompositeuFJCNetwork:

    def __init__(self, parameters, strong_form_initialize_sigma_chunks,
                 lr_cg_deformation_gradient_func,
                 strong_form_calculate_sigma_func,
                 strong_form_store_calculated_sigma_chunks,
                 weak_form_initialize_deformation_sigma_chunks,
                 weak_form_store_calculated_sigma_chunks,
                 weak_form_store_calculated_deformation_chunks):
        
        # Retain specified functions for deformation-specific calculations
        self.strong_form_initialize_sigma_chunks = (
            strong_form_initialize_sigma_chunks
        )
        self.lr_cg_deformation_gradient_func = (
            lr_cg_deformation_gradient_func
        )
        self.strong_form_calculate_sigma_func  = (
            strong_form_calculate_sigma_func
        )
        self.strong_form_store_calculated_sigma_chunks = (
            strong_form_store_calculated_sigma_chunks
        )
        self.weak_form_initialize_deformation_sigma_chunks = (
            weak_form_initialize_deformation_sigma_chunks
        )
        self.weak_form_store_calculated_sigma_chunks = (
            weak_form_store_calculated_sigma_chunks
        )
        self.weak_form_store_calculated_deformation_chunks = (
            weak_form_store_calculated_deformation_chunks
        )

        CompositeuFJCNetwork.__init__(self, parameters)

class TwoDimensionalPlaneStressNearlyIncompressibleNonaffineEightChainModelEqualForceRateDependentCompositeuFJCNetwork:

    def __init__(self, parameters, strong_form_initialize_sigma_chunks,
                 lr_cg_deformation_gradient_func,
                 strong_form_calculate_sigma_func,
                 strong_form_store_calculated_sigma_chunks,
                 weak_form_initialize_deformation_sigma_chunks,
                 weak_form_store_calculated_sigma_chunks,
                 weak_form_store_calculated_deformation_chunks):
        
        # Retain specified functions for deformation-specific calculations
        self.strong_form_initialize_sigma_chunks = (
            strong_form_initialize_sigma_chunks
        )
        self.lr_cg_deformation_gradient_func = (
            lr_cg_deformation_gradient_func
        )
        self.strong_form_calculate_sigma_func  = (
            strong_form_calculate_sigma_func
        )
        self.strong_form_store_calculated_sigma_chunks = (
            strong_form_store_calculated_sigma_chunks
        )
        self.weak_form_initialize_deformation_sigma_chunks = (
            weak_form_initialize_deformation_sigma_chunks
        )
        self.weak_form_store_calculated_sigma_chunks = (
            weak_form_store_calculated_sigma_chunks
        )
        self.weak_form_store_calculated_deformation_chunks = (
            weak_form_store_calculated_deformation_chunks
        )

        CompositeuFJCNetwork.__init__(self, parameters)

class TwoDimensionalPlaneStressCompressibleNonaffineEightChainModelEqualStrainRateIndependentCompositeuFJCNetwork:

    def __init__(self, parameters, strong_form_initialize_sigma_chunks,
                 lr_cg_deformation_gradient_func,
                 strong_form_calculate_sigma_func,
                 strong_form_store_calculated_sigma_chunks,
                 weak_form_initialize_deformation_sigma_chunks,
                 weak_form_store_calculated_sigma_chunks,
                 weak_form_store_calculated_deformation_chunks):
        
        # Retain specified functions for deformation-specific calculations
        self.strong_form_initialize_sigma_chunks = (
            strong_form_initialize_sigma_chunks
        )
        self.lr_cg_deformation_gradient_func = (
            lr_cg_deformation_gradient_func
        )
        self.strong_form_calculate_sigma_func  = (
            strong_form_calculate_sigma_func
        )
        self.strong_form_store_calculated_sigma_chunks = (
            strong_form_store_calculated_sigma_chunks
        )
        self.weak_form_initialize_deformation_sigma_chunks = (
            weak_form_initialize_deformation_sigma_chunks
        )
        self.weak_form_store_calculated_sigma_chunks = (
            weak_form_store_calculated_sigma_chunks
        )
        self.weak_form_store_calculated_deformation_chunks = (
            weak_form_store_calculated_deformation_chunks
        )

        CompositeuFJCNetwork.__init__(self, parameters)

class TwoDimensionalPlaneStressCompressibleNonaffineEightChainModelEqualStrainRateDependentCompositeuFJCNetwork:

    def __init__(self, parameters, strong_form_initialize_sigma_chunks,
                 lr_cg_deformation_gradient_func,
                 strong_form_calculate_sigma_func,
                 strong_form_store_calculated_sigma_chunks,
                 weak_form_initialize_deformation_sigma_chunks,
                 weak_form_store_calculated_sigma_chunks,
                 weak_form_store_calculated_deformation_chunks):
        
        # Retain specified functions for deformation-specific calculations
        self.strong_form_initialize_sigma_chunks = (
            strong_form_initialize_sigma_chunks
        )
        self.lr_cg_deformation_gradient_func = (
            lr_cg_deformation_gradient_func
        )
        self.strong_form_calculate_sigma_func  = (
            strong_form_calculate_sigma_func
        )
        self.strong_form_store_calculated_sigma_chunks = (
            strong_form_store_calculated_sigma_chunks
        )
        self.weak_form_initialize_deformation_sigma_chunks = (
            weak_form_initialize_deformation_sigma_chunks
        )
        self.weak_form_store_calculated_sigma_chunks = (
            weak_form_store_calculated_sigma_chunks
        )
        self.weak_form_store_calculated_deformation_chunks = (
            weak_form_store_calculated_deformation_chunks
        )

        CompositeuFJCNetwork.__init__(self, parameters)

class TwoDimensionalPlaneStressCompressibleNonaffineEightChainModelEqualForceRateIndependentCompositeuFJCNetwork:

    def __init__(self, parameters, strong_form_initialize_sigma_chunks,
                 lr_cg_deformation_gradient_func,
                 strong_form_calculate_sigma_func,
                 strong_form_store_calculated_sigma_chunks,
                 weak_form_initialize_deformation_sigma_chunks,
                 weak_form_store_calculated_sigma_chunks,
                 weak_form_store_calculated_deformation_chunks):
        
        # Retain specified functions for deformation-specific calculations
        self.strong_form_initialize_sigma_chunks = (
            strong_form_initialize_sigma_chunks
        )
        self.lr_cg_deformation_gradient_func = (
            lr_cg_deformation_gradient_func
        )
        self.strong_form_calculate_sigma_func  = (
            strong_form_calculate_sigma_func
        )
        self.strong_form_store_calculated_sigma_chunks = (
            strong_form_store_calculated_sigma_chunks
        )
        self.weak_form_initialize_deformation_sigma_chunks = (
            weak_form_initialize_deformation_sigma_chunks
        )
        self.weak_form_store_calculated_sigma_chunks = (
            weak_form_store_calculated_sigma_chunks
        )
        self.weak_form_store_calculated_deformation_chunks = (
            weak_form_store_calculated_deformation_chunks
        )

        CompositeuFJCNetwork.__init__(self, parameters)

class TwoDimensionalPlaneStressCompressibleNonaffineEightChainModelEqualForceRateDependentCompositeuFJCNetwork:

    def __init__(self, parameters, strong_form_initialize_sigma_chunks,
                 lr_cg_deformation_gradient_func,
                 strong_form_calculate_sigma_func,
                 strong_form_store_calculated_sigma_chunks,
                 weak_form_initialize_deformation_sigma_chunks,
                 weak_form_store_calculated_sigma_chunks,
                 weak_form_store_calculated_deformation_chunks):
        
        # Retain specified functions for deformation-specific calculations
        self.strong_form_initialize_sigma_chunks = (
            strong_form_initialize_sigma_chunks
        )
        self.lr_cg_deformation_gradient_func = (
            lr_cg_deformation_gradient_func
        )
        self.strong_form_calculate_sigma_func  = (
            strong_form_calculate_sigma_func
        )
        self.strong_form_store_calculated_sigma_chunks = (
            strong_form_store_calculated_sigma_chunks
        )
        self.weak_form_initialize_deformation_sigma_chunks = (
            weak_form_initialize_deformation_sigma_chunks
        )
        self.weak_form_store_calculated_sigma_chunks = (
            weak_form_store_calculated_sigma_chunks
        )
        self.weak_form_store_calculated_deformation_chunks = (
            weak_form_store_calculated_deformation_chunks
        )

        CompositeuFJCNetwork.__init__(self, parameters)