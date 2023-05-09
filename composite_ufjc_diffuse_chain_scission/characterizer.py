# import necessary libraries
from __future__ import division
from dolfin import *
from .default_parameters import default_parameters
from .utility import generate_savedir

class CompositeuFJCDiffuseChainScissionCharacterizer(object):
    
    """
    Characterizer class for composite uFJCs
    """

    def __init__(self):

        # Set the mpi communicator of the object
        self.comm_rank = MPI.rank(MPI.comm_world)
        self.comm_size = MPI.size(MPI.comm_world)

        # Parameters
        self.parameters = default_parameters()
        self.set_user_parameters()

        # Setup filesystem
        self.savedir = generate_savedir(self.prefix())
    
    def set_user_parameters(self):
        """
        Set the user parameters defining the problem
        """
        pass

    def prefix(self):
        return "characterizer"

    def characterization(self):
        pass

    def finalization(self):
        pass