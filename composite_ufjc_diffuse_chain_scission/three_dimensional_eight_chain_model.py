# Import necessary libraries
from __future__ import division
from dolfin import *
from .composite_ufjc_network import CompositeuFJCNetwork
import numpy as np
from types import SimpleNamespace
from copy import deepcopy

class ThreeDimensionalIncompressibleNonaffineEightChainModelEqualStrainRateIndependentCompositeuFJCNetwork:

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

class ThreeDimensionalIncompressibleNonaffineEightChainModelEqualStrainRateDependentCompositeuFJCNetwork:

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

class ThreeDimensionalIncompressibleNonaffineEightChainModelEqualForceRateIndependentCompositeuFJCNetwork:

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

class ThreeDimensionalIncompressibleNonaffineEightChainModelEqualForceRateDependentCompositeuFJCNetwork:

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

class ThreeDimensionalNearlyIncompressibleNonaffineEightChainModelEqualStrainRateIndependentCompositeuFJCNetwork:

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

class ThreeDimensionalNearlyIncompressibleNonaffineEightChainModelEqualStrainRateDependentCompositeuFJCNetwork:

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

class ThreeDimensionalNearlyIncompressibleNonaffineEightChainModelEqualForceRateIndependentCompositeuFJCNetwork:

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

class ThreeDimensionalNearlyIncompressibleNonaffineEightChainModelEqualForceRateDependentCompositeuFJCNetwork:

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

class ThreeDimensionalCompressibleNonaffineEightChainModelEqualStrainRateIndependentCompositeuFJCNetwork:

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

class ThreeDimensionalCompressibleNonaffineEightChainModelEqualStrainRateDependentCompositeuFJCNetwork:

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

class ThreeDimensionalCompressibleNonaffineEightChainModelEqualForceRateIndependentCompositeuFJCNetwork:

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

class ThreeDimensionalCompressibleNonaffineEightChainModelEqualForceRateDependentCompositeuFJCNetwork:

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
