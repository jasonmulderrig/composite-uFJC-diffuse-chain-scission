# Import necessary libraries
from __future__ import division
from dolfin import *
from .composite_ufjc_network import CompositeuFJCNetwork
import numpy as np
from types import SimpleNamespace
from copy import deepcopy

class TwoDimensionalPlaneStrainIncompressibleNonaffineFullNetworkMicrocircleModelEqualForceRateIndependentCompositeuFJCNetwork:

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

class TwoDimensionalPlaneStrainIncompressibleNonaffineFullNetworkMicrocircleModelEqualForceRateDependentCompositeuFJCNetwork:

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

class TwoDimensionalPlaneStrainIncompressibleAffineFullNetworkMicrocircleModelEqualStrainRateIndependentCompositeuFJCNetwork:

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

class TwoDimensionalPlaneStrainIncompressibleAffineFullNetworkMicrocircleModelEqualStrainRateDependentCompositeuFJCNetwork:

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

class TwoDimensionalPlaneStrainIncompressibleAffineFullNetworkMicrocircleModelEqualForceRateIndependentCompositeuFJCNetwork:

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

class TwoDimensionalPlaneStrainIncompressibleAffineFullNetworkMicrocircleModelEqualForceRateDependentCompositeuFJCNetwork:

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

class TwoDimensionalPlaneStrainNearlyIncompressibleNonaffineFullNetworkMicrocircleModelEqualForceRateIndependentCompositeuFJCNetwork:

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

class TwoDimensionalPlaneStrainNearlyIncompressibleNonaffineFullNetworkMicrocircleModelEqualForceRateDependentCompositeuFJCNetwork:

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

class TwoDimensionalPlaneStrainNearlyIncompressibleAffineFullNetworkMicrocircleModelEqualStrainRateIndependentCompositeuFJCNetwork:

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

class TwoDimensionalPlaneStrainNearlyIncompressibleAffineFullNetworkMicrocircleModelEqualStrainRateDependentCompositeuFJCNetwork:

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

class TwoDimensionalPlaneStrainNearlyIncompressibleAffineFullNetworkMicrocircleModelEqualForceRateIndependentCompositeuFJCNetwork:

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

class TwoDimensionalPlaneStrainNearlyIncompressibleAffineFullNetworkMicrocircleModelEqualForceRateDependentCompositeuFJCNetwork:

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

class TwoDimensionalPlaneStrainCompressibleNonaffineFullNetworkMicrocircleModelEqualForceRateIndependentCompositeuFJCNetwork:

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

class TwoDimensionalPlaneStrainCompressibleNonaffineFullNetworkMicrocircleModelEqualForceRateDependentCompositeuFJCNetwork:

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

class TwoDimensionalPlaneStrainCompressibleAffineFullNetworkMicrocircleModelEqualStrainRateIndependentCompositeuFJCNetwork:

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

class TwoDimensionalPlaneStrainCompressibleAffineFullNetworkMicrocircleModelEqualStrainRateDependentCompositeuFJCNetwork:

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

class TwoDimensionalPlaneStrainCompressibleAffineFullNetworkMicrocircleModelEqualForceRateIndependentCompositeuFJCNetwork:

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

class TwoDimensionalPlaneStrainCompressibleAffineFullNetworkMicrocircleModelEqualForceRateDependentCompositeuFJCNetwork:

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

class TwoDimensionalGeneralizedPlaneStrainIncompressibleNonaffineFullNetworkMicrocircleModelEqualForceRateIndependentCompositeuFJCNetwork:

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

class TwoDimensionalGeneralizedPlaneStrainIncompressibleNonaffineFullNetworkMicrocircleModelEqualForceRateDependentCompositeuFJCNetwork:

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

class TwoDimensionalGeneralizedPlaneStrainIncompressibleAffineFullNetworkMicrocircleModelEqualStrainRateIndependentCompositeuFJCNetwork:

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

class TwoDimensionalGeneralizedPlaneStrainIncompressibleAffineFullNetworkMicrocircleModelEqualStrainRateDependentCompositeuFJCNetwork:

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

class TwoDimensionalGeneralizedPlaneStrainIncompressibleAffineFullNetworkMicrocircleModelEqualForceRateIndependentCompositeuFJCNetwork:

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

class TwoDimensionalGeneralizedPlaneStrainIncompressibleAffineFullNetworkMicrocircleModelEqualForceRateDependentCompositeuFJCNetwork:

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

class TwoDimensionalGeneralizedPlaneStrainNearlyIncompressibleNonaffineFullNetworkMicrocircleModelEqualForceRateIndependentCompositeuFJCNetwork:

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

class TwoDimensionalGeneralizedPlaneStrainNearlyIncompressibleNonaffineFullNetworkMicrocircleModelEqualForceRateDependentCompositeuFJCNetwork:

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

class TwoDimensionalGeneralizedPlaneStrainNearlyIncompressibleAffineFullNetworkMicrocircleModelEqualStrainRateIndependentCompositeuFJCNetwork:

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

class TwoDimensionalGeneralizedPlaneStrainNearlyIncompressibleAffineFullNetworkMicrocircleModelEqualStrainRateDependentCompositeuFJCNetwork:

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

class TwoDimensionalGeneralizedPlaneStrainNearlyIncompressibleAffineFullNetworkMicrocircleModelEqualForceRateIndependentCompositeuFJCNetwork:

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

class TwoDimensionalGeneralizedPlaneStrainNearlyIncompressibleAffineFullNetworkMicrocircleModelEqualForceRateDependentCompositeuFJCNetwork:

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

class TwoDimensionalGeneralizedPlaneStrainCompressibleNonaffineFullNetworkMicrocircleModelEqualForceRateIndependentCompositeuFJCNetwork:

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

class TwoDimensionalGeneralizedPlaneStrainCompressibleNonaffineFullNetworkMicrocircleModelEqualForceRateDependentCompositeuFJCNetwork:

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

class TwoDimensionalGeneralizedPlaneStrainCompressibleAffineFullNetworkMicrocircleModelEqualStrainRateIndependentCompositeuFJCNetwork:

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

class TwoDimensionalGeneralizedPlaneStrainCompressibleAffineFullNetworkMicrocircleModelEqualStrainRateDependentCompositeuFJCNetwork:

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

class TwoDimensionalGeneralizedPlaneStrainCompressibleAffineFullNetworkMicrocircleModelEqualForceRateIndependentCompositeuFJCNetwork:

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

class TwoDimensionalGeneralizedPlaneStrainCompressibleAffineFullNetworkMicrocircleModelEqualForceRateDependentCompositeuFJCNetwork:

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

class TwoDimensionalPlaneStressIncompressibleNonaffineFullNetworkMicrocircleModelEqualForceRateIndependentCompositeuFJCNetwork:

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

class TwoDimensionalPlaneStressIncompressibleNonaffineFullNetworkMicrocircleModelEqualForceRateDependentCompositeuFJCNetwork:

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

class TwoDimensionalPlaneStressIncompressibleAffineFullNetworkMicrocircleModelEqualStrainRateIndependentCompositeuFJCNetwork:

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

class TwoDimensionalPlaneStressIncompressibleAffineFullNetworkMicrocircleModelEqualStrainRateDependentCompositeuFJCNetwork:

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

class TwoDimensionalPlaneStressIncompressibleAffineFullNetworkMicrocircleModelEqualForceRateIndependentCompositeuFJCNetwork:

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

class TwoDimensionalPlaneStressIncompressibleAffineFullNetworkMicrocircleModelEqualForceRateDependentCompositeuFJCNetwork:

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

class TwoDimensionalPlaneStressNearlyIncompressibleNonaffineFullNetworkMicrocircleModelEqualForceRateIndependentCompositeuFJCNetwork:

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

class TwoDimensionalPlaneStressNearlyIncompressibleNonaffineFullNetworkMicrocircleModelEqualForceRateDependentCompositeuFJCNetwork:

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

class TwoDimensionalPlaneStressNearlyIncompressibleAffineFullNetworkMicrocircleModelEqualStrainRateIndependentCompositeuFJCNetwork:

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

class TwoDimensionalPlaneStressNearlyIncompressibleAffineFullNetworkMicrocircleModelEqualStrainRateDependentCompositeuFJCNetwork:

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

class TwoDimensionalPlaneStressNearlyIncompressibleAffineFullNetworkMicrocircleModelEqualForceRateIndependentCompositeuFJCNetwork:

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

class TwoDimensionalPlaneStressNearlyIncompressibleAffineFullNetworkMicrocircleModelEqualForceRateDependentCompositeuFJCNetwork:

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

class TwoDimensionalPlaneStressCompressibleNonaffineFullNetworkMicrocircleModelEqualForceRateIndependentCompositeuFJCNetwork:

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

class TwoDimensionalPlaneStressCompressibleNonaffineFullNetworkMicrocircleModelEqualForceRateDependentCompositeuFJCNetwork:

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

class TwoDimensionalPlaneStressCompressibleAffineFullNetworkMicrocircleModelEqualStrainRateIndependentCompositeuFJCNetwork:

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

class TwoDimensionalPlaneStressCompressibleAffineFullNetworkMicrocircleModelEqualStrainRateDependentCompositeuFJCNetwork:

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

class TwoDimensionalPlaneStressCompressibleAffineFullNetworkMicrocircleModelEqualForceRateIndependentCompositeuFJCNetwork:

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

class TwoDimensionalPlaneStressCompressibleAffineFullNetworkMicrocircleModelEqualForceRateDependentCompositeuFJCNetwork:

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