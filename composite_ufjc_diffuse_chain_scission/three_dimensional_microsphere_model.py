# Import necessary libraries
from __future__ import division
from dolfin import *
from .composite_ufjc_network import CompositeuFJCNetwork
import numpy as np
from types import SimpleNamespace
from copy import deepcopy

class ThreeDimensionalIncompressibleNonaffineFullNetworkMicrosphereModelEqualForceRateIndependentCompositeuFJCNetwork:

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

class ThreeDimensionalIncompressibleNonaffineFullNetworkMicrosphereModelEqualForceRateDependentCompositeuFJCNetwork:

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

class ThreeDimensionalIncompressibleAffineFullNetworkMicrosphereModelEqualStrainRateIndependentCompositeuFJCNetwork:

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

class ThreeDimensionalIncompressibleAffineFullNetworkMicrosphereModelEqualStrainRateDependentCompositeuFJCNetwork:

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

class ThreeDimensionalIncompressibleAffineFullNetworkMicrosphereModelEqualForceRateIndependentCompositeuFJCNetwork:

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

class ThreeDimensionalIncompressibleAffineFullNetworkMicrosphereModelEqualForceRateDependentCompositeuFJCNetwork:

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

class ThreeDimensionalNearlyIncompressibleNonaffineFullNetworkMicrosphereModelEqualForceRateIndependentCompositeuFJCNetwork:

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

class ThreeDimensionalNearlyIncompressibleNonaffineFullNetworkMicrosphereModelEqualForceRateDependentCompositeuFJCNetwork:

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

class ThreeDimensionalNearlyIncompressibleAffineFullNetworkMicrosphereModelEqualStrainRateIndependentCompositeuFJCNetwork:

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

class ThreeDimensionalNearlyIncompressibleAffineFullNetworkMicrosphereModelEqualStrainRateDependentCompositeuFJCNetwork:

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

class ThreeDimensionalNearlyIncompressibleAffineFullNetworkMicrosphereModelEqualForceRateIndependentCompositeuFJCNetwork:

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

class ThreeDimensionalNearlyIncompressibleAffineFullNetworkMicrosphereModelEqualForceRateDependentCompositeuFJCNetwork:

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

class ThreeDimensionalCompressibleNonaffineFullNetworkMicrosphereModelEqualForceRateIndependentCompositeuFJCNetwork:

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

class ThreeDimensionalCompressibleNonaffineFullNetworkMicrosphereModelEqualForceRateDependentCompositeuFJCNetwork:

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

class ThreeDimensionalCompressibleAffineFullNetworkMicrosphereModelEqualStrainRateIndependentCompositeuFJCNetwork:

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

class ThreeDimensionalCompressibleAffineFullNetworkMicrosphereModelEqualStrainRateDependentCompositeuFJCNetwork:

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

class ThreeDimensionalCompressibleAffineFullNetworkMicrosphereModelEqualForceRateIndependentCompositeuFJCNetwork:

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

class ThreeDimensionalCompressibleAffineFullNetworkMicrosphereModelEqualForceRateDependentCompositeuFJCNetwork:

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
