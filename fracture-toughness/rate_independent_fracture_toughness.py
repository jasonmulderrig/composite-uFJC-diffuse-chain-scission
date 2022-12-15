"""The single-chain AFM tensile test curve fit validation 
characterization module for composite uFJCs that undergo scission
"""

# import external modules
from __future__ import division
from composite_ufjc_scission import (CompositeuFJCScissionCharacterizer,
RateIndependentScissionCompositeuFJC,
latex_formatting_figure,
save_current_figure,
save_current_figure_no_labels
)
import numpy as np
from math import floor, log10
from scipy import constants
import matplotlib.pyplot as plt


class AFMChainTensileTestCurveFitCharacterizer(
        CompositeuFJCScissionCharacterizer):
    """The characterization class assessing the single-chain AFM tensile
    test curve fit validation for composite uFJCs that undergo scission.
    It inherits all attributes and methods from the
    ``CompositeuFJCScissionCharacterizer`` class.
    """
    def __init__(self, paper_authors, chain, T):
        """Initializes the ``AFMChainTensileTestCurveFitCharacterizer``
        class by initializing and inheriting all attributes and methods
        from the ``CompositeuFJCScissionCharacterizer`` class.
        """
        self.paper_authors = paper_authors
        self.chain = chain
        self.T = T

        CompositeuFJCScissionCharacterizer.__init__(self)
    
    def set_user_parameters(self):
        """Set user-defined parameters"""
        k_B  = constants.value(u'Boltzmann constant') # J/K
        N_A  = constants.value(u'Avogadro constant') # 1/mol
        beta = 1. / (k_B*self.T) # 1/J

        p = self.parameters

        p.characterizer.chain_data_directory = (
            "./AFM_chain_tensile_test_curve_fit_results/"
        )

        p.characterizer.paper_authors2polymer_type_dict = {
            "al-maawali-et-al": "pdms",
            "hugel-et-al": "pva"
        }
        p.characterizer.paper_authors2polymer_type_label_dict = {
            "al-maawali-et-al": r'$\textrm{PDMS single chain data}$',
            "hugel-et-al": r'$\textrm{PVA single chain data}$'
        }
        p.characterizer.polymer_type_label2chain_backbone_bond_type_dict = {
            "pdms": "si-o",
            "pva": "c-c"
        }
        p.characterizer.paper_authors_chain2xlim_chain_mechanical_response_plot = {
            "al-maawali-et-al": {
                "chain-a": [0, 70]
            },
            "hugel-et-al": {
                "chain-a": [0, 1200],
            }
        }
        p.characterizer.paper_authors_chain2ylim_chain_mechanical_response_plot = {
            "al-maawali-et-al": {
                "chain-a": [-0.1, 2]
            },
            "hugel-et-al": {
                "chain-a": [-0.1, 1.5],
            }
        }
        # from DFT simulations on H_3C-CH_2-CH_3 (c-c) 
        # and H_3Si-O-CH_3 (si-o) by Beyer, J Chem. Phys., 2000
        p.characterizer.chain_backbone_bond_type2f_c_max_dict = {
            "c-c": 6.92,
            "si-o": 5.20
        } # nN
        # from DFT simulations on H_3C-CH_2-CH_3 (c-c) 
        # and H_3Si-O-CH_3 (si-o) by Beyer, J Chem. Phys., 2000
        p.characterizer.chain_backbone_bond_type2typcl_AFM_exprmt_f_c_max_dict = {
            "c-c": 3.81,
            "si-o": 3.14
        } # nN
        # (c-c) from the CRC Handbook of Chemistry and Physics for
        # CH_3-C_2H_5 citing Luo, Y.R., Comprehensive Handbook of
        # Chemical Bond Energies, CRC Press, 2007
        # (si-o) from Schwaderer et al., Langmuir, 2008, citing Holleman
        # and Wilberg, Inorganic Chemistry, 2001
        chain_backbone_bond_type2epsilon_b_char_dict = {
            "c-c": 370.3,
            "si-o": 444
        } # kJ/mol
        p.characterizer.chain_backbone_bond_type2zeta_b_char_dict = {
            chain_backbone_bond_type_key: epsilon_b_char_val/N_A*1000*beta
            for chain_backbone_bond_type_key, epsilon_b_char_val
            in chain_backbone_bond_type2epsilon_b_char_dict.items()
        } # (kJ/mol -> kJ -> J)*1/J
        # (c-c) from the CRC Handbook of Chemistry and Physics for the
        # C-C bond in C#-H_2C-CH_2-C#
        # (si-o) from the CRC Handbook of Chemistry and Physics for the
        # Si-O bond in X_3-Si-O-C#
        chain_backbone_bond_type2l_b_eq_dict = {
            "c-c": 1.524,
            "si-o": 1.645
        } # Angstroms
        p.characterizer.chain_backbone_bond_type2l_b_eq_dict = {
            chain_backbone_bond_type_key: l_b_eq_val/10
            for chain_backbone_bond_type_key, l_b_eq_val
            in chain_backbone_bond_type2l_b_eq_dict.items()
        } # Angstroms -> nm

        p.characterizer.f_c_num_steps = 100001

        f_c_dot_list = [1e1, 1e5, 1e9] # nN/sec
        f_c_dot_exponent_list = [
            int(floor(log10(abs(f_c_dot_list[i]))))
            for i in range(len(f_c_dot_list))
        ]
        f_c_dot_label_list = [
            r'$\dot{f}_c='+'10^{0:d}'.format(f_c_dot_exponent_list[i])+'~nN/sec$'
            for i in range(len(f_c_dot_list))
        ]
        f_c_dot_color_list = ['orange', 'purple', 'green']

        p.characterizer.f_c_dot_list          = f_c_dot_list
        p.characterizer.f_c_dot_exponent_list = f_c_dot_exponent_list
        p.characterizer.f_c_dot_label_list    = f_c_dot_label_list
        p.characterizer.f_c_dot_color_list    = f_c_dot_color_list

        # nu = 1 -> nu = 3125
        nu_list = [i for i in range(1, 5**5+1)]

        p.characterizer.nu_list = nu_list

    def prefix(self):
        """Set characterization prefix"""
        return "rate_independent_fracture_toughness"
    
    def characterization(self):
        """Define characterization routine"""
        k_B     = constants.value(u'Boltzmann constant') # J/K
        h       = constants.value(u'Planck constant') # J/Hz
        hbar    = h / (2*np.pi) # J*sec
        beta    = 1. / (k_B*self.T) # 1/J

        beta = beta / (1e9*1e9) # 1/J = 1/(N*m) -> 1/(nN*m) -> 1/(nN*nm)

        cp = self.parameters.characterizer

        polymer_type = cp.paper_authors2polymer_type_dict[self.paper_authors]
        chain_backbone_bond_type = (
            cp.polymer_type_label2chain_backbone_bond_type_dict[polymer_type]
        )
        self.data_file_prefix = (
            self.paper_authors + '-' + polymer_type + '-'
            + chain_backbone_bond_type + '-' + self.chain
        )

        # unitless, unitless, unitless, nm, respectively
        nu = np.loadtxt(
            cp.chain_data_directory+self.data_file_prefix+'-composite-uFJC-curve-fit-intgr_nu'+'.txt')
        zeta_nu_char = np.loadtxt(
            cp.chain_data_directory+self.data_file_prefix+'-composite-uFJC-curve-fit-zeta_nu_char_intgr_nu'+'.txt')
        kappa_nu = np.loadtxt(
            cp.chain_data_directory+self.data_file_prefix+'-composite-uFJC-curve-fit-kappa_nu_intgr_nu'+'.txt')
        l_nu_eq = np.loadtxt(
            cp.chain_data_directory+self.data_file_prefix+'-composite-uFJC-curve-fit-l_nu_eq_intgr_nu'+'.txt')
        
        f_c_max = (
            cp.chain_backbone_bond_type2f_c_max_dict[chain_backbone_bond_type]
        ) # nN
        typcl_AFM_exprmt_f_c_max = (
            cp.chain_backbone_bond_type2typcl_AFM_exprmt_f_c_max_dict[chain_backbone_bond_type]
        ) # nN
        xi_c_max = f_c_max * beta * l_nu_eq
        typcl_AFM_exprmt_xi_c_max = typcl_AFM_exprmt_f_c_max * beta * l_nu_eq
        
        
        single_chain_list = [
            RateIndependentScissionCompositeuFJC(
                nu=cp.nu_list[nu_indx], zeta_nu_char=zeta_nu_char,
                kappa_nu=kappa_nu)
            for nu_indx in range(len(cp.nu_list))
        ]
        
        
        A_nu_list = [
            single_chain.A_nu
            for single_chain in single_chain_list
        ]
        
        inext_gaussian_A_nu_list = [
            1/np.sqrt(nu_val) for nu_val in cp.nu_list
        ]
        
        inext_gaussian_A_nu_err_list = [
            np.abs((inext_gaussian_A_nu_val-A_nu_val)/A_nu_val)*100
            for inext_gaussian_A_nu_val, A_nu_val
            in zip(inext_gaussian_A_nu_list, A_nu_list)
        ]
        
        
        epsilon_cnu_diss_hat_crit_list = [
            single_chain.epsilon_cnu_diss_hat_crit
            for single_chain in single_chain_list
        ]
        g_c_crit_list = [
            single_chain.g_c_crit
            for single_chain in single_chain_list
        ]
        overline_epsilon_cnu_diss_hat_crit_list = [
            epsilon_cnu_diss_hat_crit_chain_network_val/zeta_nu_char
            for epsilon_cnu_diss_hat_crit_chain_network_val
            in epsilon_cnu_diss_hat_crit_list
        ]
        overline_g_c_crit_list = [
            g_c_crit_chain_network_val/zeta_nu_char
            for g_c_crit_chain_network_val in g_c_crit_list
        ]
        
        
        LT_epsilon_cnu_diss_hat_crit_list = (
            [zeta_nu_char] * len(cp.nu_list)
        )
        LT_g_c_crit_list = [
            0.5 * A_nu_val * nu_val**2 * epsilon_cnu_diss_hat_crit_val
            for A_nu_val, nu_val, epsilon_cnu_diss_hat_crit_val
            in zip(
                A_nu_list, cp.nu_list,
                LT_epsilon_cnu_diss_hat_crit_list)
        ]
        LT_overline_epsilon_cnu_diss_hat_crit_list = (
            [1] * len(cp.nu_list)
        )
        LT_overline_g_c_crit_list = [
            0.5 * A_nu_val * nu_val**2 * overline_epsilon_cnu_diss_hat_crit_val
            for A_nu_val, nu_val, overline_epsilon_cnu_diss_hat_crit_val
            in zip(
                A_nu_list, cp.nu_list,
                LT_overline_epsilon_cnu_diss_hat_crit_list)
        ]
        
        
        LT_inext_gaussian_g_c_crit_list = [
            0.5 * A_nu_val * nu_val**2 * epsilon_cnu_diss_hat_crit_val
            for A_nu_val, nu_val, epsilon_cnu_diss_hat_crit_val
            in zip(
                inext_gaussian_A_nu_list,
                cp.nu_list,
                LT_epsilon_cnu_diss_hat_crit_list)
        ]
        LT_overline_inext_gaussian_g_c_crit_list = [
            0.5 * A_nu_val * nu_val**2 * overline_epsilon_cnu_diss_hat_crit_val
            for A_nu_val, nu_val, overline_epsilon_cnu_diss_hat_crit_val
            in zip(
                inext_gaussian_A_nu_list,
                cp.nu_list,
                LT_overline_epsilon_cnu_diss_hat_crit_list)
        ]
        
        
        single_chain = RateIndependentScissionCompositeuFJC(
            nu=nu, zeta_nu_char=zeta_nu_char, kappa_nu=kappa_nu)
        
        
        RC_xi_c_max_epsilon_cnu_diss_hat_crit_val = (
            single_chain.epsilon_cnu_sci_hat_func(
                single_chain.lmbda_nu_xi_c_hat_func(xi_c_max))
        )
        RC_xi_c_max_epsilon_cnu_diss_hat_crit_list = (
            [RC_xi_c_max_epsilon_cnu_diss_hat_crit_val]
            * len(cp.nu_list)
        )
        RC_xi_c_max_g_c_crit_list = [
            0.5 * A_nu_val * nu_val**2 * epsilon_cnu_diss_hat_crit_val
            for A_nu_val, nu_val, epsilon_cnu_diss_hat_crit_val
            in zip(
                A_nu_list, cp.nu_list,
                RC_xi_c_max_epsilon_cnu_diss_hat_crit_list)
        ]
        RC_xi_c_max_overline_epsilon_cnu_diss_hat_crit_list = [
            RC_xi_c_max_epsilon_cnu_diss_hat_crit_val/zeta_nu_char
            for RC_xi_c_max_epsilon_cnu_diss_hat_crit_val
            in RC_xi_c_max_epsilon_cnu_diss_hat_crit_list
        ]
        RC_xi_c_max_overline_g_c_crit_list = [
            0.5 * A_nu_val * nu_val**2 * overline_epsilon_cnu_diss_hat_crit_val
            for A_nu_val, nu_val, overline_epsilon_cnu_diss_hat_crit_val
            in zip(
                A_nu_list, cp.nu_list,
                RC_xi_c_max_overline_epsilon_cnu_diss_hat_crit_list)
        ]
        
        
        RC_typcl_AFM_exprmt_xi_c_max_epsilon_cnu_diss_hat_crit_val = (
            single_chain.epsilon_cnu_sci_hat_func(
                single_chain.lmbda_nu_xi_c_hat_func(typcl_AFM_exprmt_xi_c_max))
        )
        RC_typcl_AFM_exprmt_xi_c_max_epsilon_cnu_diss_hat_crit_list = (
            [RC_typcl_AFM_exprmt_xi_c_max_epsilon_cnu_diss_hat_crit_val]
            * len(cp.nu_list)
        )
        RC_typcl_AFM_exprmt_xi_c_max_g_c_crit_list = [
            0.5 * A_nu_val * nu_val**2 * epsilon_cnu_diss_hat_crit_val
            for A_nu_val, nu_val, epsilon_cnu_diss_hat_crit_val
            in zip(
                A_nu_list, cp.nu_list,
                RC_typcl_AFM_exprmt_xi_c_max_epsilon_cnu_diss_hat_crit_list)
        ]
        RC_typcl_AFM_exprmt_xi_c_max_overline_epsilon_cnu_diss_hat_crit_list = [
            RC_typcl_AFM_exprmt_xi_c_max_epsilon_cnu_diss_hat_crit_val/zeta_nu_char
            for RC_typcl_AFM_exprmt_xi_c_max_epsilon_cnu_diss_hat_crit_val
            in RC_typcl_AFM_exprmt_xi_c_max_epsilon_cnu_diss_hat_crit_list
        ]
        RC_typcl_AFM_exprmt_xi_c_max_overline_g_c_crit_list = [
            0.5 * A_nu_val * nu_val**2 * overline_epsilon_cnu_diss_hat_crit_val
            for A_nu_val, nu_val, overline_epsilon_cnu_diss_hat_crit_val
            in zip(
                A_nu_list, cp.nu_list,
                RC_typcl_AFM_exprmt_xi_c_max_overline_epsilon_cnu_diss_hat_crit_list)
        ]
        
        
        self.A_nu_list = A_nu_list
        self.inext_gaussian_A_nu_list = (
            inext_gaussian_A_nu_list
        )
        self.inext_gaussian_A_nu_err_list = (
            inext_gaussian_A_nu_err_list
        )
        
        self.epsilon_cnu_diss_hat_crit_list = (
            epsilon_cnu_diss_hat_crit_list
        )
        self.g_c_crit_list = g_c_crit_list
        self.overline_epsilon_cnu_diss_hat_crit_list = (
            overline_epsilon_cnu_diss_hat_crit_list
        )
        self.overline_g_c_crit_list = (
            overline_g_c_crit_list
        )
        
        self.LT_epsilon_cnu_diss_hat_crit_list = (
            LT_epsilon_cnu_diss_hat_crit_list
        )
        self.LT_g_c_crit_list = LT_g_c_crit_list
        self.LT_overline_epsilon_cnu_diss_hat_crit_list = (
            LT_overline_epsilon_cnu_diss_hat_crit_list
        )
        self.LT_overline_g_c_crit_list = (
            LT_overline_g_c_crit_list
        )
        
        self.LT_inext_gaussian_g_c_crit_list = (
            LT_inext_gaussian_g_c_crit_list
        )
        self.LT_overline_inext_gaussian_g_c_crit_list = (
            LT_overline_inext_gaussian_g_c_crit_list
        )
        
        self.RC_xi_c_max_epsilon_cnu_diss_hat_crit_list = (
            RC_xi_c_max_epsilon_cnu_diss_hat_crit_list
        )
        self.RC_xi_c_max_g_c_crit_list = (
            RC_xi_c_max_g_c_crit_list
        )
        self.RC_xi_c_max_overline_epsilon_cnu_diss_hat_crit_list = (
            RC_xi_c_max_overline_epsilon_cnu_diss_hat_crit_list
        )
        self.RC_xi_c_max_overline_g_c_crit_list = (
            RC_xi_c_max_overline_g_c_crit_list
        )
        
        self.RC_typcl_AFM_exprmt_xi_c_max_epsilon_cnu_diss_hat_crit_list = (
            RC_typcl_AFM_exprmt_xi_c_max_epsilon_cnu_diss_hat_crit_list
        )
        self.RC_typcl_AFM_exprmt_xi_c_max_g_c_crit_list = (
            RC_typcl_AFM_exprmt_xi_c_max_g_c_crit_list
        )
        self.RC_typcl_AFM_exprmt_xi_c_max_overline_epsilon_cnu_diss_hat_crit_list = (
            RC_typcl_AFM_exprmt_xi_c_max_overline_epsilon_cnu_diss_hat_crit_list
        )
        self.RC_typcl_AFM_exprmt_xi_c_max_overline_g_c_crit_list = (
            RC_typcl_AFM_exprmt_xi_c_max_overline_g_c_crit_list
        )

    def finalization(self):
        """Define finalization analysis"""
        cp  = self.parameters.characterizer
        ppp = self.parameters.post_processing

        # plot results
        latex_formatting_figure(ppp)

        fig, (ax1, ax2) = plt.subplots(
            2, 1, gridspec_kw={'height_ratios': [2, 1]}, sharex=True)
        
        ax1.semilogx(
            cp.nu_list, self.A_nu_list, linestyle='-',
            color='blue', alpha=1, linewidth=2.5,
            label=r'$u\textrm{FJC}$')
        ax1.semilogx(
            cp.nu_list, self.inext_gaussian_A_nu_list, linestyle='--',
            color='red', alpha=1, linewidth=2.5,
            label=r'$\textrm{inextensible Gaussian chain}$')
        ax1.legend(loc='best', fontsize=14)
        ax1.tick_params(axis='y', labelsize=14)
        ax1.set_ylabel(r'$\mathcal{A}_{\nu}$', fontsize=20)
        ax1.grid(True, alpha=0.25)
        
        ax2.loglog(
            cp.nu_list, self.inext_gaussian_A_nu_err_list,
            linestyle='-', color='blue', alpha=1, linewidth=2.5)
        ax2.tick_params(axis='y', labelsize=14)
        ax2.set_ylabel(r'$\%~\textrm{error}$', fontsize=20)
        ax2.grid(True, alpha=0.25)
        
        plt.xticks(fontsize=14)
        plt.xlabel(r'$\nu$', fontsize=20)
        save_current_figure_no_labels(
            self.savedir,
            "A_nu-gen-ufjc-model-framework-and-inextensible-Gaussian-chain-comparison")


        # # plot curve fit results
        # fig = plt.figure()
        # plt.scatter(
        #     self.r_nu, self.f_c, color='blue', marker='o', alpha=1,
        #     linewidth=2.5,
        #     label=cp.paper_authors2polymer_type_label_dict[self.paper_authors])
        # plt.plot(
        #     self.r_nu_fit, self.f_c_fit, linestyle='--', color='red', alpha=1,
        #     linewidth=2.5, label=r'$u\textrm{FJC model fit}$')
        # plt.legend(loc='best', fontsize=18)
        # plt.grid(True, alpha=0.25)
        # plt.xlim(
        #     cp.paper_authors_chain2xlim_chain_mechanical_response_plot[self.paper_authors][self.chain])
        # plt.xticks(fontsize=20)
        # plt.ylim(
        #     cp.paper_authors_chain2ylim_chain_mechanical_response_plot[self.paper_authors][self.chain])
        # plt.yticks(fontsize=20)
        # save_current_figure(
        #     self.savedir, r'$r_{\nu}~(nm)$', 30, r'$f_c~(nN)$', 30,
        #     self.data_file_prefix+"-f_c-vs-r_nu-composite-uFJC-curve-fit")

        # fig = plt.figure()
        # plt.scatter(
        #     self.r_nu, self.f_c, color='blue', marker='o', alpha=1,
        #     linewidth=2.5,
        #     label=cp.paper_authors2polymer_type_label_dict[self.paper_authors])
        # plt.plot(
        #     self.r_nu_fit_intgr_nu, self.f_c_fit_intgr_nu, linestyle='--',
        #     color='red', alpha=1, linewidth=2.5,
        #     label=r'$u\textrm{FJC model fit}$')
        # plt.legend(loc='best', fontsize=18)
        # plt.grid(True, alpha=0.25)
        # plt.xlim(
        #     cp.paper_authors_chain2xlim_chain_mechanical_response_plot[self.paper_authors][self.chain])
        # plt.xticks(fontsize=20)
        # plt.ylim(
        #     cp.paper_authors_chain2ylim_chain_mechanical_response_plot[self.paper_authors][self.chain])
        # plt.yticks(fontsize=20)
        # save_current_figure(
        #     self.savedir, r'$r_{\nu}~(nm)$', 30, r'$f_c~(nN)$', 30,
        #     self.data_file_prefix+"-intgr_nu-f_c-vs-r_nu-composite-uFJC-curve-fit")
        
        # fig = plt.figure()
        # plt.scatter(
        #     self.lmbda_c_eq, self.xi_c, color='blue', marker='o', alpha=1,
        #     linewidth=2.5,
        #     label=cp.paper_authors2polymer_type_label_dict[self.paper_authors])
        # plt.plot(
        #     self.lmbda_c_eq_fit, self.xi_c_fit, linestyle='--', color='red',
        #     alpha=1, linewidth=2.5, label=r'$u\textrm{FJC model fit}$')
        # plt.legend(loc='best', fontsize=18)
        # plt.grid(True, alpha=0.25)
        # plt.xlim([0, 1.15])
        # plt.xticks(fontsize=20)
        # plt.yticks(fontsize=20)
        # save_current_figure(
        #     self.savedir, r'$\lambda_c^{eq}$', 30, r'$\xi_{c}$', 30,
        #     self.data_file_prefix+"-xi_c-vs-lmbda_c_eq-composite-uFJC-curve-fit")
        
        # fig = plt.figure()
        # plt.scatter(
        #     self.lmbda_c_eq_intgr_nu, self.xi_c_intgr_nu, color='blue',
        #     marker='o', alpha=1, linewidth=2.5,
        #     label=cp.paper_authors2polymer_type_label_dict[self.paper_authors])
        # plt.plot(
        #     self.lmbda_c_eq_fit_intgr_nu, self.xi_c_fit_intgr_nu,
        #     linestyle='--', color='red', alpha=1, linewidth=2.5,
        #     label=r'$u\textrm{FJC model fit}$')
        # plt.legend(loc='best', fontsize=18)
        # plt.grid(True, alpha=0.25)
        # plt.xlim([0, 1.15])
        # plt.xticks(fontsize=20)
        # plt.yticks(fontsize=20)
        # save_current_figure(
        #     self.savedir, r'$\lambda_c^{eq}$', 30, r'$\xi_{c}$', 30,
        #     self.data_file_prefix+"-intgr_nu-xi_c-vs-lmbda_c_eq-gen-uFJC-curve-fit")

        # # plot rate-dependent chain results
        # t_max = 0
        # fig, (ax1, ax2, ax3, ax4) = plt.subplots(
        #     4, 1, gridspec_kw={'height_ratios': [2, 1, 1, 1]}, sharex=True)
        # for f_c_dot_indx in range(len(cp.f_c_dot_list)):
        #     t_max = max(
        #         [t_max, self.rate_dependent_t_steps___f_c_dot_chunk[f_c_dot_indx][-1]])
        #     ax1.semilogx(
        #         self.rate_dependent_t_steps___f_c_dot_chunk[f_c_dot_indx],
        #         self.rate_dependent_xi_c___f_c_dot_chunk[f_c_dot_indx],
        #         linestyle='-', color=cp.f_c_dot_color_list[f_c_dot_indx],
        #         alpha=1, linewidth=2.5,
        #         label=cp.f_c_dot_label_list[f_c_dot_indx])
        #     ax2.semilogx(
        #         self.rate_dependent_t_steps___f_c_dot_chunk[f_c_dot_indx],
        #         self.rate_dependent_lmbda_nu___f_c_dot_chunk[f_c_dot_indx],
        #         linestyle='-', color=cp.f_c_dot_color_list[f_c_dot_indx],
        #         alpha=1, linewidth=2.5)
        #     ax3.semilogx(
        #         self.rate_dependent_t_steps___f_c_dot_chunk[f_c_dot_indx],
        #         self.rate_dependent_gamma_c___f_c_dot_chunk[f_c_dot_indx],
        #         linestyle='-', color=cp.f_c_dot_color_list[f_c_dot_indx],
        #         alpha=1, linewidth=2.5)
        #     ax4.semilogx(
        #         self.rate_dependent_t_steps___f_c_dot_chunk[f_c_dot_indx],
        #         self.rate_dependent_overline_epsilon_cnu_diss_hat___f_c_dot_chunk[f_c_dot_indx],
        #         linestyle='-', color=cp.f_c_dot_color_list[f_c_dot_indx],
        #         alpha=1, linewidth=2.5)
        # ax1.hlines(
        #     y=self.xi_c_max, xmin=0, xmax=t_max, linestyle='--', color='black',
        #     alpha=1, linewidth=1)
        # ax1.legend(loc='best', fontsize=12)
        # ax1.tick_params(axis='y', labelsize=16)
        # ax1.set_ylabel(r'$\xi_c$', fontsize=20)
        # ax1.grid(True, alpha=0.25)
        # ax2.tick_params(axis='y', labelsize=16)
        # ax2.set_ylabel(r'$\lambda_{\nu}$', fontsize=20)
        # ax2.grid(True, alpha=0.25)
        # ax3.tick_params(axis='y', labelsize=16)
        # ax3.set_ylabel(r'$\gamma_c$', fontsize=20)
        # ax3.grid(True, alpha=0.25)
        # ax4.tick_params(axis='y', labelsize=16)
        # ax4.set_yticks([0.0, 0.25, 0.5])
        # ax4.set_ylabel(
        #     r'$\overline{\hat{\varepsilon}_{c\nu}^{diss}}$', fontsize=20)
        # ax4.grid(True, alpha=0.25)
        # plt.xticks(fontsize=16)
        # plt.xlabel(r'$t~(sec)$', fontsize=20)
        # save_current_figure_no_labels(
        #     self.savedir,
        #     self.data_file_prefix+"-rate-dependent-xi_c-lmbda_nu-gamma_c-overline_epsilon_cnu_diss_hat-vs-time")
        
        # # plot rate-independent and rate-dependent chain results
        # # together
        # # plot rate-dependent chain results
        # lmbda_c_eq_max = 0
        # fig, (ax1, ax2, ax3, ax4) = plt.subplots(
        #     4, 1, gridspec_kw={'height_ratios': [1.25, 0.75, 2, 1.5]},
        #     sharex=True)
        # ax1.scatter(
        #     self.lmbda_c_eq_intgr_nu, self.xi_c_intgr_nu, color='red',
        #     marker='o', alpha=1, linewidth=1,
        #     label=cp.paper_authors2polymer_type_label_dict[self.paper_authors])
        # for f_c_dot_indx in range(len(cp.f_c_dot_list)):
        #     lmbda_c_eq_max = max(
        #         [lmbda_c_eq_max, self.rate_dependent_lmbda_c_eq___f_c_dot_chunk[f_c_dot_indx][-1]])
        #     ax1.plot(
        #         self.rate_dependent_lmbda_c_eq___f_c_dot_chunk[f_c_dot_indx],
        #         self.rate_dependent_xi_c___f_c_dot_chunk[f_c_dot_indx],
        #         linestyle='-', color=cp.f_c_dot_color_list[f_c_dot_indx],
        #         alpha=1, linewidth=2.5)
        #     ax2.plot(
        #         self.rate_dependent_lmbda_c_eq___f_c_dot_chunk[f_c_dot_indx],
        #         self.rate_dependent_lmbda_nu___f_c_dot_chunk[f_c_dot_indx],
        #         linestyle='-', color=cp.f_c_dot_color_list[f_c_dot_indx],
        #         alpha=1, linewidth=2.5)
        #     ax3.plot(
        #         self.rate_dependent_lmbda_c_eq___f_c_dot_chunk[f_c_dot_indx],
        #         self.rate_dependent_gamma_c___f_c_dot_chunk[f_c_dot_indx],
        #         linestyle='-', color=cp.f_c_dot_color_list[f_c_dot_indx],
        #         alpha=1, linewidth=2.5, label=cp.f_c_dot_label_list[f_c_dot_indx])
        #     ax4.plot(
        #         self.rate_dependent_lmbda_c_eq___f_c_dot_chunk[f_c_dot_indx],
        #         self.rate_dependent_overline_epsilon_cnu_diss_hat___f_c_dot_chunk[f_c_dot_indx],
        #         linestyle='-', color=cp.f_c_dot_color_list[f_c_dot_indx],
        #         alpha=1, linewidth=2.5)
        # ax1.hlines(
        #     y=self.xi_c_max, xmin=-0.05, xmax=lmbda_c_eq_max + 0.05,
        #     linestyle='--', color='black', alpha=1, linewidth=1)
        # # plot rate-independent chain results
        # ax1.plot(
        #     self.rate_independent_lmbda_c_eq,
        #     self.rate_independent_xi_c, linestyle='-', color='blue',
        #     alpha=1, linewidth=2.5, label=r'$\textrm{rate-independent chain}$')
        # ax2.plot(
        #     self.rate_independent_lmbda_c_eq,
        #     self.rate_independent_lmbda_nu, linestyle='-', color='blue',
        #     alpha=1, linewidth=2.5)
        # ax3.plot(
        #     self.rate_independent_lmbda_c_eq,
        #     self.rate_independent_p_c_sci_hat, linestyle='-', color='blue',
        #     alpha=1, linewidth=2.5)
        # ax4.plot(
        #     self.rate_independent_lmbda_c_eq,
        #     self.rate_independent_overline_epsilon_cnu_sci_hat, linestyle='-',
        #     color='black', alpha=1, linewidth=2.5)
        # ax4.plot(
        #     self.rate_independent_lmbda_c_eq,
        #     self.rate_independent_overline_epsilon_cnu_diss_hat, linestyle='-',
        #     color='blue', alpha=1, linewidth=2.5)

        # ax1.legend(loc='best', fontsize=12)
        # ax1.tick_params(axis='y', labelsize=16)
        # ax1.set_ylabel(r'$\xi_c$', fontsize=20)
        # ax1.grid(True, alpha=0.25)
        # ax2.tick_params(axis='y', labelsize=16)
        # ax2.set_ylabel(r'$\lambda_{\nu}$', fontsize=20)
        # ax2.grid(True, alpha=0.25)
        # ax3.legend(loc='best', fontsize=12)
        # ax3.tick_params(axis='y', labelsize=16)
        # ax3.set_ylabel(r'$\gamma_c,~\hat{p}_c^{sci}$', fontsize=20)
        # ax3.grid(True, alpha=0.25)
        # ax4.set_yticks([0.0, 0.25, 0.5])
        # ax4.tick_params(axis='y', labelsize=16)
        # ax4.set_ylabel(
        #     r'$\overline{\hat{\varepsilon}_{c\nu}^{sci}},~\overline{\hat{\varepsilon}_{c\nu}^{diss}}$',
        #     fontsize=20)
        # ax4.grid(True, alpha=0.25)
        # plt.xlim([-0.05, lmbda_c_eq_max + 0.05])
        # plt.xticks(fontsize=16)
        # plt.xlabel(r'$\lambda_c^{eq}$', fontsize=20)
        # save_current_figure_no_labels(
        #     self.savedir,
        #     self.data_file_prefix+"-rate-independent-and-rate-dependent-chains-vs-lmbda_c_eq")

        # # plot rate-independent and rate-dependent chain results
        # # together while omitting chain scission energy
        # # plot rate-dependent chain results
        # lmbda_c_eq_max = 0
        # fig, (ax1, ax2, ax3, ax4) = plt.subplots(
        #     4, 1, gridspec_kw={'height_ratios': [1.25, 0.75, 2, 1.5]},
        #     sharex=True)
        # ax1.scatter(
        #     self.lmbda_c_eq_intgr_nu, self.xi_c_intgr_nu, color='red',
        #     marker='o', alpha=1, linewidth=1,
        #     label=cp.paper_authors2polymer_type_label_dict[self.paper_authors])
        # for f_c_dot_indx in range(len(cp.f_c_dot_list)):
        #     lmbda_c_eq_max = max([lmbda_c_eq_max, self.rate_dependent_lmbda_c_eq___f_c_dot_chunk[f_c_dot_indx][-1]])
        #     ax1.plot(
        #         self.rate_dependent_lmbda_c_eq___f_c_dot_chunk[f_c_dot_indx],
        #         self.rate_dependent_xi_c___f_c_dot_chunk[f_c_dot_indx],
        #         linestyle='-', color=cp.f_c_dot_color_list[f_c_dot_indx],
        #         alpha=1, linewidth=2.5)
        #     ax2.plot(
        #         self.rate_dependent_lmbda_c_eq___f_c_dot_chunk[f_c_dot_indx],
        #         self.rate_dependent_lmbda_nu___f_c_dot_chunk[f_c_dot_indx],
        #         linestyle='-', color=cp.f_c_dot_color_list[f_c_dot_indx],
        #         alpha=1, linewidth=2.5)
        #     ax3.plot(
        #         self.rate_dependent_lmbda_c_eq___f_c_dot_chunk[f_c_dot_indx],
        #         self.rate_dependent_gamma_c___f_c_dot_chunk[f_c_dot_indx],
        #         linestyle='-', color=cp.f_c_dot_color_list[f_c_dot_indx],
        #         alpha=1, linewidth=2.5,
        #         label=cp.f_c_dot_label_list[f_c_dot_indx])
        #     ax4.plot(
        #         self.rate_dependent_lmbda_c_eq___f_c_dot_chunk[f_c_dot_indx],
        #         self.rate_dependent_overline_epsilon_cnu_diss_hat___f_c_dot_chunk[f_c_dot_indx],
        #         linestyle='-', color=cp.f_c_dot_color_list[f_c_dot_indx],
        #         alpha=1, linewidth=2.5)
        # ax1.hlines(
        #     y=self.xi_c_max, xmin=-0.05, xmax=lmbda_c_eq_max + 0.05,
        #     linestyle='--', color='black', alpha=1, linewidth=1)
        # # plot rate-independent chain results
        # ax1.plot(
        #     self.rate_independent_lmbda_c_eq,
        #     self.rate_independent_xi_c, linestyle='-', color='blue',
        #     alpha=1, linewidth=2.5,
        #     label=r'$\textrm{rate-independent chain}$')
        # ax2.plot(
        #     self.rate_independent_lmbda_c_eq,
        #     self.rate_independent_lmbda_nu, linestyle='-', color='blue',
        #     alpha=1, linewidth=2.5)
        # ax3.plot(
        #     self.rate_independent_lmbda_c_eq,
        #     self.rate_independent_p_c_sci_hat, linestyle='-', color='blue',
        #     alpha=1, linewidth=2.5)
        # ax4.plot(
        #     self.rate_independent_lmbda_c_eq,
        #     self.rate_independent_overline_epsilon_cnu_diss_hat, linestyle='-',
        #     color='blue', alpha=1, linewidth=2.5)

        # ax1.legend(loc='best', fontsize=12)
        # ax1.tick_params(axis='y', labelsize=16)
        # ax1.set_ylabel(r'$\xi_c$', fontsize=20)
        # ax1.grid(True, alpha=0.25)
        # ax2.tick_params(axis='y', labelsize=16)
        # ax2.set_ylabel(r'$\lambda_{\nu}$', fontsize=20)
        # ax2.grid(True, alpha=0.25)
        # ax3.tick_params(axis='y', labelsize=16)
        # ax3.legend(loc='best', fontsize=12)
        # ax3.set_ylabel(r'$\gamma_c,~\hat{p}_c^{sci}$', fontsize=20)
        # ax3.grid(True, alpha=0.25)
        # ax4.set_yticks([0.0, 0.25, 0.5])
        # ax4.tick_params(axis='y', labelsize=16)
        # ax4.set_ylabel(
        #     r'$\overline{\hat{\varepsilon}_{c\nu}^{sci}},~\overline{\hat{\varepsilon}_{c\nu}^{diss}}$',
        #     fontsize=20)
        # ax4.grid(True, alpha=0.25)
        # plt.xlim([-0.05, lmbda_c_eq_max + 0.05])
        # plt.xticks(fontsize=16)
        # plt.xlabel(r'$\lambda_c^{eq}$', fontsize=20)
        # save_current_figure_no_labels(
        #     self.savedir,
        #     self.data_file_prefix+"-rate-independent-and-rate-dependent-chains-vs-lmbda_c_eq-no-epsilon_cnu_diss")


if __name__ == '__main__':

    T = 298 # absolute room temperature, K

    AFM_chain_tensile_tests_dict = {
        "al-maawali-et-al": "chain-a", "hugel-et-al": "chain-a"
    }

    # AFM_chain_tensile_tests_characterizer_list = [
    #     AFMChainTensileTestCurveFitCharacterizer(
    #         paper_authors=AFM_chain_tensile_test[0],
    #         chain=AFM_chain_tensile_test[1], T=T)
    #     for AFM_chain_tensile_test in AFM_chain_tensile_test_list
    # ]

    # for AFM_chain_tensile_test_indx \
    #     in range(len(AFM_chain_tensile_tests_characterizer_list)):
    #     AFM_chain_tensile_tests_characterizer_list[AFM_chain_tensile_test_indx].characterization()
    #     AFM_chain_tensile_tests_characterizer_list[AFM_chain_tensile_test_indx].finalization()
    
    # characterizer = AFMChainTensileTestCurveFitCharacterizer(
    #     paper_authors=AFM_chain_tensile_test_list[0][0],
    #     chain=AFM_chain_tensile_test_list[0][1], T=T)
    # characterizer.characterization()
    # characterizer.finalization()

    characterizer = AFMChainTensileTestCurveFitCharacterizer(
        paper_authors="al-maawali-et-al", chain="chain-a", T=T)
    characterizer.characterization()
    characterizer.finalization()