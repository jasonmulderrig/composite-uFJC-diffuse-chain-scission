# Import necessary libraries
from __future__ import division
from dolfin import *
from composite_ufjc_scission import (
    RateIndependentScissionCompositeuFJC,
    RateDependentScissionCompositeuFJC,
    RateIndependentSmoothstepScissionCompositeuFJC,
    RateDependentSmoothstepScissionCompositeuFJC,
    RateIndependentSigmoidScissionCompositeuFJC,
    RateDependentSigmoidScissionCompositeuFJC
)
from composite_ufjc_scission_ufl_fenics import (
    RateIndependentScissionCompositeuFJCUFLFEniCS,
    RateDependentScissionCompositeuFJCUFLFEniCS,
    RateIndependentSmoothstepScissionCompositeuFJCUFLFEniCS,
    RateDependentSmoothstepScissionCompositeuFJCUFLFEniCS,
    RateIndependentSigmoidScissionCompositeuFJCUFLFEniCS,
    RateDependentSigmoidScissionCompositeuFJCUFLFEniCS
)
# from .microsphere_quadrature import MicrosphereQuadratureScheme
# from .microcircle_quadrature import MicrocircleQuadratureScheme
import sys
import numpy as np


class CompositeuFJCNetwork(object):
    
    def __init__(self, parameters):

        def equal_strain_composite_ufjc_network(material_parameters):
        
            mp = material_parameters

            if mp.rate_dependence == 'rate_independent' and mp.scission_model == 'analytical':
                composite_ufjc_list = [
                    RateIndependentScissionCompositeuFJC(nu = mp.nu_list[nu_indx],
                                                        nu_b = mp.nu_b,
                                                        zeta_b_char = mp.zeta_b_char,
                                                        kappa_b = mp.kappa_b,
                                                        zeta_nu_char = mp.zeta_nu_char,
                                                        kappa_nu = mp.kappa_nu)
                    for nu_indx in range(len(mp.nu_list))
                ]
                composite_ufjc_ufl_fenics_list = [
                    RateIndependentScissionCompositeuFJCUFLFEniCS(nu = mp.nu_list[nu_indx],
                                                        nu_b = mp.nu_b,
                                                        zeta_b_char = mp.zeta_b_char,
                                                        kappa_b = mp.kappa_b,
                                                        zeta_nu_char = mp.zeta_nu_char,
                                                        kappa_nu = mp.kappa_nu)
                    for nu_indx in range(len(mp.nu_list))
                ]
            elif mp.rate_dependence == 'rate_dependent' and mp.scission_model == 'analytical':
                composite_ufjc_list = [
                    RateDependentScissionCompositeuFJC(omega_0 = mp.omega_0,
                                                        nu = mp.nu_list[nu_indx],
                                                        nu_b = mp.nu_b,
                                                        zeta_b_char = mp.zeta_b_char,
                                                        kappa_b = mp.kappa_b,
                                                        zeta_nu_char = mp.zeta_nu_char,
                                                        kappa_nu = mp.kappa_nu)
                    for nu_indx in range(len(mp.nu_list))
                ]
                composite_ufjc_ufl_fenics_list = [
                    RateDependentScissionCompositeuFJCUFLFEniCS(omega_0 = mp.omega_0,
                                                        nu = mp.nu_list[nu_indx],
                                                        nu_b = mp.nu_b,
                                                        zeta_b_char = mp.zeta_b_char,
                                                        kappa_b = mp.kappa_b,
                                                        zeta_nu_char = mp.zeta_nu_char,
                                                        kappa_nu = mp.kappa_nu)
                    for nu_indx in range(len(mp.nu_list))
                ]
            elif mp.rate_dependence == 'rate_independent' and mp.scission_model == 'smoothstep':
                composite_ufjc_list = [
                    RateIndependentSmoothstepScissionCompositeuFJC(nu = mp.nu_list[nu_indx],
                                                        nu_b = mp.nu_b,
                                                        zeta_b_char = mp.zeta_b_char,
                                                        kappa_b = mp.kappa_b,
                                                        zeta_nu_char = mp.zeta_nu_char,
                                                        kappa_nu = mp.kappa_nu,
                                                        lmbda_nu_crit_min = mp.lmbda_nu_crit_min,
                                                        lmbda_nu_crit_max = mp.lmbda_nu_crit_max)
                    for nu_indx in range(len(mp.nu_list))
                ]
                composite_ufjc_ufl_fenics_list = [
                    RateIndependentSmoothstepScissionCompositeuFJCUFLFEniCS(nu = mp.nu_list[nu_indx],
                                                        nu_b = mp.nu_b,
                                                        zeta_b_char = mp.zeta_b_char,
                                                        kappa_b = mp.kappa_b,
                                                        zeta_nu_char = mp.zeta_nu_char,
                                                        kappa_nu = mp.kappa_nu,
                                                        lmbda_nu_crit_min = mp.lmbda_nu_crit_min,
                                                        lmbda_nu_crit_max = mp.lmbda_nu_crit_max)
                    for nu_indx in range(len(mp.nu_list))
                ]
            elif mp.rate_dependence == 'rate_dependent' and mp.scission_model == 'smoothstep':
                composite_ufjc_list = [
                    RateDependentSmoothstepScissionCompositeuFJC(omega_0 = mp.omega_0,
                                                        nu = mp.nu_list[nu_indx],
                                                        nu_b = mp.nu_b,
                                                        zeta_b_char = mp.zeta_b_char,
                                                        kappa_b = mp.kappa_b,
                                                        zeta_nu_char = mp.zeta_nu_char,
                                                        kappa_nu = mp.kappa_nu,
                                                        lmbda_nu_crit_min = mp.lmbda_nu_crit_min,
                                                        lmbda_nu_crit_max = mp.lmbda_nu_crit_max)
                    for nu_indx in range(len(mp.nu_list))
                ]
                composite_ufjc_ufl_fenics_list = [
                    RateDependentSmoothstepScissionCompositeuFJCUFLFEniCS(omega_0 = mp.omega_0,
                                                        nu = mp.nu_list[nu_indx],
                                                        nu_b = mp.nu_b,
                                                        zeta_b_char = mp.zeta_b_char,
                                                        kappa_b = mp.kappa_b,
                                                        zeta_nu_char = mp.zeta_nu_char,
                                                        kappa_nu = mp.kappa_nu,
                                                        lmbda_nu_crit_min = mp.lmbda_nu_crit_min,
                                                        lmbda_nu_crit_max = mp.lmbda_nu_crit_max)
                    for nu_indx in range(len(mp.nu_list))
                ]
            elif mp.rate_dependence == 'rate_independent' and mp.scission_model == 'sigmoid':
                composite_ufjc_list = [
                    RateIndependentSigmoidScissionCompositeuFJC(nu = mp.nu_list[nu_indx],
                                                        nu_b = mp.nu_b,
                                                        zeta_b_char = mp.zeta_b_char,
                                                        kappa_b = mp.kappa_b,
                                                        zeta_nu_char = mp.zeta_nu_char,
                                                        kappa_nu = mp.kappa_nu,
                                                        tau = mp.tau,
                                                        lmbda_nu_check = mp.lmbda_nu_check)
                    for nu_indx in range(len(mp.nu_list))
                ]
                composite_ufjc_ufl_fenics_list = [
                    RateIndependentSigmoidScissionCompositeuFJCUFLFEniCS(nu = mp.nu_list[nu_indx],
                                                        nu_b = mp.nu_b,
                                                        zeta_b_char = mp.zeta_b_char,
                                                        kappa_b = mp.kappa_b,
                                                        zeta_nu_char = mp.zeta_nu_char,
                                                        kappa_nu = mp.kappa_nu,
                                                        tau = mp.tau,
                                                        lmbda_nu_check = mp.lmbda_nu_check)
                    for nu_indx in range(len(mp.nu_list))
                ]
            elif mp.rate_dependence == 'rate_dependent' and mp.scission_model == 'sigmoid':
                composite_ufjc_list = [
                    RateDependentSigmoidScissionCompositeuFJC(omega_0 = mp.omega_0,
                                                        nu = mp.nu_list[nu_indx],
                                                        nu_b = mp.nu_b,
                                                        zeta_b_char = mp.zeta_b_char,
                                                        kappa_b = mp.kappa_b,
                                                        zeta_nu_char = mp.zeta_nu_char,
                                                        kappa_nu = mp.kappa_nu,
                                                        tau = mp.tau,
                                                        lmbda_nu_check = mp.lmbda_nu_check)
                    for nu_indx in range(len(mp.nu_list))
                ]
                composite_ufjc_ufl_fenics_list = [
                    RateDependentSigmoidScissionCompositeuFJCUFLFEniCS(omega_0 = mp.omega_0,
                                                        nu = mp.nu_list[nu_indx],
                                                        nu_b = mp.nu_b,
                                                        zeta_b_char = mp.zeta_b_char,
                                                        kappa_b = mp.kappa_b,
                                                        zeta_nu_char = mp.zeta_nu_char,
                                                        kappa_nu = mp.kappa_nu,
                                                        tau = mp.tau,
                                                        lmbda_nu_check = mp.lmbda_nu_check)
                    for nu_indx in range(len(mp.nu_list))
                ]

            # Separate out specified parameters
            nu_list                   = mp.nu_list
            nu_min                    = min(nu_list)
            nu_max                    = max(nu_list)
            nu_num                    = len(nu_list)
            P_nu_list                 = [P_nu(mp, nu_list[nu_indx]) for nu_indx in range(len(nu_list))]
            P_nu_sum                  = np.sum(P_nu_list)

            cond_val                  = composite_ufjc_ufl_fenics_list[0].cond_val
            zeta_nu_char              = composite_ufjc_ufl_fenics_list[0].zeta_nu_char
            kappa_nu                  = composite_ufjc_ufl_fenics_list[0].kappa_nu
            lmbda_nu_ref              = composite_ufjc_ufl_fenics_list[0].lmbda_nu_ref
            lmbda_c_eq_ref            = composite_ufjc_ufl_fenics_list[0].lmbda_c_eq_ref
            lmbda_nu_crit             = composite_ufjc_ufl_fenics_list[0].lmbda_nu_crit
            lmbda_c_eq_crit           = composite_ufjc_ufl_fenics_list[0].lmbda_c_eq_crit
            xi_c_crit                 = composite_ufjc_ufl_fenics_list[0].xi_c_crit
            lmbda_nu_pade2berg_crit   = composite_ufjc_ufl_fenics_list[0].lmbda_nu_pade2berg_crit
            lmbda_c_eq_pade2berg_crit = composite_ufjc_ufl_fenics_list[0].lmbda_c_eq_pade2berg_crit
            A_nu_list                 = [composite_ufjc_ufl_fenics_list[nu_indx].A_nu for nu_indx in range(len(nu_list))]
            Lambda_nu_ref_list        = [composite_ufjc_ufl_fenics_list[nu_indx].Lambda_nu_ref for nu_indx in range(len(nu_list))]
            
            # Retain specified parameters
            self.composite_ufjc_list            = composite_ufjc_list
            self.composite_ufjc_ufl_fenics_list = composite_ufjc_ufl_fenics_list
            self.nu_list                        = nu_list
            self.nu_min                         = nu_min
            self.nu_max                         = nu_max
            self.nu_num                         = nu_num
            self.P_nu_list                      = P_nu_list
            self.P_nu_sum                       = P_nu_sum

            self.cond_val                       = cond_val
            self.zeta_nu_char                   = zeta_nu_char
            self.kappa_nu                       = kappa_nu
            self.lmbda_nu_ref                   = lmbda_nu_ref
            self.lmbda_c_eq_ref                 = lmbda_c_eq_ref
            self.lmbda_nu_crit                  = lmbda_nu_crit
            self.lmbda_c_eq_crit                = lmbda_c_eq_crit
            self.xi_c_crit                      = xi_c_crit
            self.lmbda_nu_pade2berg_crit        = lmbda_nu_pade2berg_crit
            self.lmbda_c_eq_pade2berg_crit      = lmbda_c_eq_pade2berg_crit
            self.A_nu_list                      = A_nu_list
            self.Lambda_nu_ref_list             = Lambda_nu_ref_list

        def equal_force_composite_ufjc_network(material_parameters):
            sys.exit("The equal force chain-level load sharing implementation has not been finalized yet. For now, please choose equal strain chain-level load sharing.")

        def P_nu(material_parameters, nu):
            
            mp = material_parameters

            if mp.nu_distribution == "itskov":
                return (1/(mp.Delta_nu+1))*(1+(1/mp.Delta_nu))**(mp.nu_min-nu)

        # Check the correctness of the specified parameters
        if hasattr(parameters, "material") == False or hasattr(parameters, "deformation") == False:
            sys.exit("Need to specify either material parameters, deformation parameters, or both in order to define the composite uFJC network")
        
        mp = parameters.material
        dp = parameters.deformation

        # Retain specified parameters
        self.scission_model                     = getattr(mp, "scission_model")
        self.network_model                      = getattr(mp, "network_model")
        self.physical_dimension                 = getattr(mp, "physical_dimension")
        self.incompressibility_assumption       = getattr(mp, "incompressibility_assumption")
        self.macro2micro_deformation_assumption = getattr(mp, "macro2micro_deformation_assumption")
        self.micro2macro_homogenization_scheme  = getattr(mp, "micro2macro_homogenization_scheme")
        self.chain_level_load_sharing           = getattr(mp, "chain_level_load_sharing")
        self.rate_dependence                    = getattr(mp, "rate_dependence")
        self.two_dimensional_formulation        = getattr(mp, "two_dimensional_formulation")
        self.microcircle_quadrature_order       = getattr(mp, "microcircle_quadrature_order")
        self.microsphere_quadrature_order       = getattr(mp, "microsphere_quadrature_order")

        self.omega_0 = getattr(mp, "omega_0")

        self.nu_chunks_indx_list    = getattr(mp, "nu_chunks_indx_list")
        self.point_chunks_indx_list = getattr(mp, "point_chunks_indx_list")

        self.deformation_type        = getattr(dp, "deformation_type")
        self.K_G                     = getattr(dp, "K_G")
        self.lmbda_damping_init      = getattr(dp, "lmbda_damping_init")
        self.min_lmbda_damping_val   = getattr(dp, "min_lmbda_damping_val")
        self.iter_max_Gamma_val_NR   = getattr(dp, "iter_max_Gamma_val_NR")
        self.tol_Gamma_val_NR        = getattr(dp, "tol_Gamma_val_NR")
        self.iter_max_lmbda_c_val_NR = getattr(dp, "iter_max_lmbda_c_val_NR")
        self.tol_lmbda_c_val_NR      = getattr(dp, "tol_lmbda_c_val_NR")
        self.iter_max_stag_NR        = getattr(dp, "iter_max_stag_NR")
        self.tol_lmbda_c_val_stag_NR = getattr(dp, "tol_lmbda_c_val_stag_NR")
        self.tol_Gamma_val_stag_NR   = getattr(dp, "tol_Gamma_val_stag_NR")
        self.epsilon                 = getattr(dp, "epsilon")
        self.max_J_val_cond          = getattr(dp, "max_J_val_cond")
        self.iter_max_d_c_val        = getattr(dp, "iter_max_d_c_val")
        self.tol_d_c_val             = getattr(dp, "tol_d_c_val")
        self.k_cond_val              = getattr(dp, "k_cond_val")


        if self.scission_model != "analytical" and self.scission_model != "smoothstep" and self.scission_model != "sigmoid":
            sys.exit("Error: Need to specify the scission model. Either the analytical, smoothstep, or sigmoid scission models can be used.")
        
        if self.network_model != "statistical_mechanics_model":
            sys.exit("Error: This composite uFJC material class strictly corresponds to a statistical mechanics model.")
        
        if self.physical_dimension != 2 and self.physical_dimension != 3:
            sys.exit("Error: Need to specify either a 2D or a 3D problem.")
        
        if self.incompressibility_assumption != "incompressible" and self.incompressibility_assumption != "nearly_incompressible" and self.incompressibility_assumption != "compressible":
            sys.exit("Error: Need to specify a proper incompressibility assumption for the material. The material is either incompressible, nearly incompressible, or compressible.")

        if self.macro2micro_deformation_assumption != 'affine' and self.macro2micro_deformation_assumption != 'nonaffine':
            sys.exit('Error: Need to specify the macro-to-micro deformation assumption in the network. Either affine deformation or non-affine deformation can be used.')
        
        if self.micro2macro_homogenization_scheme != 'eight_chain_model' and self.micro2macro_homogenization_scheme != 'full_network_microcircle_model' and self.micro2macro_homogenization_scheme != 'full_network_microsphere_model':
            sys.exit('Error: Need to specify the micro-to-macro homogenization scheme in the network. Either the eight chain model, the full network microcircle micro-to-macro homogenization scheme, or the full network microsphere micro-to-macro homogenization scheme can be used.')
        
        if self.chain_level_load_sharing != 'equal_strain' and self.chain_level_load_sharing != 'equal_force':
            sys.exit('Error: Need to specify the load sharing assumption that the network/chains in the composite uFJC network obey. Either the equal strain chain level load sharing assumption or the equal force chain level load sharing assumption can be used.')
        
        if self.rate_dependence != 'rate_dependent' and self.rate_dependence != 'rate_independent':
            sys.exit('Error: Need to specify the network/chain dependence on the rate of applied deformation. Either rate-dependent or rate-independent deformation can be used.')
        
        if self.rate_dependence == 'rate_dependent' and self.omega_0 is None:
            sys.exit('Error: Need to specify the microscopic frequency of segments in the network for rate-dependent network deformation.')
        
        if self.macro2micro_deformation_assumption == 'affine' and self.micro2macro_homogenization_scheme == 'eight_chain_model':
            sys.exit('Error: The eight chain micro-to-macro homogenization scheme technically exhibits the non-affine macro-to-micro deformation assumption.')
        
        if self.macro2micro_deformation_assumption == 'nonaffine' and (self.micro2macro_homogenization_scheme == 'full_network_microsphere_model' or self.micro2macro_homogenization_scheme == 'full_network_microcircle_model') and self.chain_level_load_sharing == 'equal_strain':
            sys.exit('Error: In the non-affine macro-to-micro deformation assumption utilizing either the full network microsphere micro-to-macro homogenization scheme or the full network microcircle micro-to-macro homogenization scheme, the composite uFJCs are required to obey the equal force load sharing assumption.')
        
        if self.physical_dimension == 2:
            if self.two_dimensional_formulation != "plane_strain" and self.two_dimensional_formulation != "generalized_plane_strain" and self.two_dimensional_formulation != "plane_stress":
                sys.exit("Error: Need to specify a proper two-dimensional formulation. Either plane strain, generalized plane strain, or plane stress can be used.")
            
            if self.micro2macro_homogenization_scheme == 'full_network_microsphere_model':
                sys.exit("Error: For a 2D problem, the full network microsphere micro-to-macro homogenization scheme cannot be used. Either the eight chain model or the full network microcircle micro-to-macro homogenization scheme can be used for 2D problems.")
            
            # # Specify full network microcircle quadrature scheme, if necessary
            # elif self.micro2macro_homogenization_scheme == 'full_network_microcircle_model':
            #     if self.microcircle_quadrature_order is None:
            #         sys.exit('Error: Need to specify microcircle quadrature order number in order to utilize the full network microcircle micro-to-macro homogenization scheme.')
            #     else:
            #         self.microcircle = MicrocircleQuadratureScheme(self.microcircle_quadrature_order)
        
        elif self.physical_dimension == 3:
            if self.micro2macro_homogenization_scheme == 'full_network_microcircle_model':
                sys.exit("Error: For a 2D problem, the full network microcircle micro-to-macro homogenization scheme cannot be used. Either the eight chain model or the full network microsphere micro-to-macro homogenization scheme can be used for 2D problems.")
            
            # # Specify full network microsphere quadrature scheme, if necessary
            # elif self.micro2macro_homogenization_scheme == 'full_network_microsphere_model':
            #     if self.microsphere_quadrature_order is None:
            #         sys.exit('Error: Need to specify microsphere quadrature order number in order to utilize the full network microsphere micro-to-macro homogenization scheme.')
            #     else:
            #         self.microsphere = MicrosphereQuadratureScheme(self.microsphere_quadrature_order)
        
        # Specify chain-level load sharing and chain composition
        if self.chain_level_load_sharing == 'equal_strain':
            equal_strain_composite_ufjc_network(mp)
        elif self.chain_level_load_sharing == 'equal_force':
            equal_force_composite_ufjc_network(mp)