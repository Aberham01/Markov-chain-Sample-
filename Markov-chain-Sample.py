# -*- coding: utf-8 -*-
 


import numpy as np
from all_functions import get_stationary_distribution, prob_log_likelihood, learn_Markov_model

# =============================================================================
# initialize
# =============================================================================
p = np.array([[0.12, 0.1, 0.02],
              [0.8, 0.8, 0.8],
             [0.08, 0.1, 0.18]])

s = 'ACAACACAACACAACCCCBCBAAAAACCCABCCCABCAAACACCCCAAACACAACBBBCACCAACAABCCAACCCCACABACACCACCCCCCBACCBACABCCAACCCCCABAAAACABCABCAACAACCBAAACACAAAACCBAAAABCCABCACAABAABAAACACBBABBAC'


# =============================================================================
# Define functions
# =============================================================================

# main function block
def main(p, s):
    
    # learn and print a Markov model
    transition_matrix = learn_Markov_model(s, dim=3)
    print("3a) Transition Matrix: \n", transition_matrix)
    
    # compute log-likelihood
    _, log_likelihood = prob_log_likelihood(s, transition_matrix)
    print("\n3b) Log-likelihood: ", np.round(log_likelihood,2))    
    
    # compute and display stationary distribution
    pi_3 = get_stationary_distribution(p)
    print("\n4a) Stationary distribution pi_lambda3: \n", pi_3)
    
    # compute and display stationary distribution
    pi_4 = get_stationary_distribution(transition_matrix)
    print("\n4a) Stationary distribution pi_lambda4: \n", pi_4)
    
    # find likeliest room
    index_dict = {0: 'A', 1: 'B', 2: 'C'}
    
    room_model_l3 = index_dict[list(pi_3).index(max(pi_3))]
    print("\n4b) Likeliest room under model lambda3 is:", room_model_l3)
    
    room_model_l4 = index_dict[list(pi_4).index(max(pi_4))]
    print("\n4b) Likeliest room under model lambda4 is:", room_model_l4)
    
 
# =============================================================================
# invoke and run main    
# =============================================================================
if __name__ == "__main__":
    main(p, s)
    