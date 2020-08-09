# -*- coding: utf-8 -*-
"""
Solution 2
"""


import numpy as np
import random
from all_functions import get_next_move, generate_sequence, prob_log_likelihood

# =============================================================================
# initialize
# =============================================================================
p = np.array([[0.12, 0.1, 0.02],
              [0.8, 0.8, 0.8],
             [0.08, 0.1, 0.18]])

freeze_random_state = True  # use False, if new solution needs to be generated in every run, otherwise use True

seed = 42  # seed for random state generator. Applicable if "freeze_random_state" is True


# =============================================================================
# Define functions
# =============================================================================

# main function block
def main(p, freeze_random_state, seed = 21):
    
    if freeze_random_state:
        random.seed(seed)
    
    # generate sequence
    sequence = generate_sequence(p, start='B', size=19)
    print("2a) Sequence is: \n", sequence)
    
    # compute log-likelihood
    _, log_likelihood = prob_log_likelihood(sequence, p)
    print("\n2b) Log-likelihood: ", np.round(log_likelihood,2))    
    
    # generate samples
    samples = []
    for i in range(10000):
        samples.append(generate_sequence(p, start='B', size=19))
    
    # compute and print average log-likelihood of all samples
    log_likelihood_list = [prob_log_likelihood(s, p)[1] for s in samples]
    avg_log_likelihood = np.average(log_likelihood_list)
    print("\n2c) Expected log-likelihood: E(L) = ", avg_log_likelihood)
    
    varience = np.average([(l - avg_log_likelihood)**2 for l in log_likelihood_list])
    print("\n2c) Varience all sequences: var(L) = ", varience)
        
 
# =============================================================================
# invoke and run main    
# =============================================================================
if __name__ == "__main__":
    main(p, freeze_random_state, seed)
    