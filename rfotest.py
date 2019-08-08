import numpy as np
from gradient import Grad_sum
import os

def get_RFOStep(curr_hess = None, curr_grad = None, curr_vars = None, maxstep = None):
    assert curr_hess is not None
    assert curr_grad is not None
    assert curr_vars is not None
    assert maxstep is not None

    #First build augmented Hessian matrix
    lastrow = np.append(curr_grad, 0.0)
    aughess = np.vstack((np.column_stack((curr_hess, curr_grad)), lastrow))
    eigenvals, eigenvecs = np.linalg.eigh(aughess) # <---- Could use a Davidson routine maybe.
    # scale eigenvector
    lowestev = np.argmin(eigenvals)
    if eigenvals[lowestev] > 1.0E-4:
        raise ValueError('ERROR: I don\'t want to go up the PES!')
    else:
        stepvec = eigenvecs[:, lowestev]
        if stepvec[-1] == 0.0:
            scaledstepvec = stepvec[:-1]
        else:
            scaledstepvec = stepvec[:-1]/stepvec[-1]
    #utils.log_Vector('SimpleRFO Step vector:', scaledstepvec, logging.debug)
    if np.linalg.norm(scaledstepvec) > 10:
        print('    WARNING: Step is weirdly large!')
    slen, smax = np.linalg.norm(scaledstepvec), np.amax(np.absolute(scaledstepvec))
    
    if slen > maxstep:
             print('    RFO Step too large, scaling it.')
             if slen > smax:
                 print('    Scaling by length. Factor: {:12.6f}'.format(maxstep/slen))
                 scaledstepvec *= maxstep/slen
             else:
                 print('    Scaling by maximum component. Factor: {:12.6f}'.format(maxstep/smax))
                 scaledstepvec *= maxstep/smax
    return scaledstepvec
