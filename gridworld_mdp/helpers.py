import numpy as np

def create_mdp(reward_in,trans_in):
    """
    Inputs: 1) rewards for [non-terminal states, 
                           good terminal state,
                            bad terminal state]
            2) transition probabilities for intended
              and unintended directions
    Outputs: transition matrix T of shape (A,S,S) 
             reward matrix R of shape (S,)
    """
    ###########################
    # Extract MDP Inputs
    ###########################

    r_s, r_g, r_b = reward_in
    
    p_intended, p_opposite, p_right, p_left = trans_in

    assert ((1.0 - sum(trans_in)) < 10**(-3)), "Inputted transition probabilities do not sum to 1."

    ###########################
    # Create Reward Matrix
    ###########################

    #flatten reward so we have reward matrix of size (S,)
    R = np.ravel(np.array([[r_s, r_s, r_s, r_g], 
                                [r_s, 100, r_s, r_b], 
                                [r_s, r_s, r_s, r_s]]))


    ###########################
    # Create Transition Matrix
    ###########################

    #initialize empty transition array of size (A,S,S)
    T = np.zeros((4, len(R),len(R)))

    #action transitions in order up, right, down, left
    act_trans = [-4, +1, +4, -1]
    act_prob = np.array([[p_intended, p_right, p_opposite, p_left],
                        [p_left, p_intended, p_right, p_opposite],
                        [p_opposite, p_left,p_intended, p_right],
                        [p_right, p_opposite, p_left, p_intended]])

    #transition probabilities for each cardinal direction
    for state in range(len(R)):
        for intended_action in range(4): 
            for actual_action in range(4): 
                new_state = state + act_trans[actual_action]
                curr_prob = act_prob[intended_action,actual_action]

                #these conditions depend only on state
                if ((state == 3) or (state == 7)):
                    #if terminal states, self transitions are only possible
                    T[intended_action, state, state] += curr_prob
                elif ((new_state == 5) or (state == 5)):
                    #cannot end up in the block in the middle
                    T[intended_action, state, state] += curr_prob 

                #these conditions focus on staying within bounds
                elif ((state % 4 == 3) and (actual_action == 1)):
                    #if on righthand side, we can't go right
                    T[intended_action, state, state] += curr_prob
                elif ((state % 4 == 0) and (actual_action == 3)):
                    #if on lefthand side, we can't go left
                    T[intended_action, state, state] += curr_prob
                elif ((state > 7) and (actual_action == 2)):
                    #if on bottom, we can't go down
                    T[intended_action, state, state] += curr_prob
                elif ((state < 4) and (actual_action == 0)):
                    #if on top, we can't go up
                    T[intended_action, state, state] += curr_prob 

                #if non of above conditions are met, move is possible
                else:
                    #otherwise we do end up in the new state
                    T[intended_action, state, new_state] +=  curr_prob
    return T,R

def get_q_values(mdp):
    """
    Input: mdp object
    Outputs: policy with -1 for states where all
             actions have the same value
    Verbose: if mdp is in verbose mode, prints out state-action 
             values (Q values) for each state
    """
    #calculate Q-values, similar to _bellmanOperator method
    Q = np.empty((mdp.A, mdp.S))
    for aa in range(mdp.A):
        Q[aa] = mdp.R[aa] + mdp.discount * mdp.P[aa].dot(mdp.V)

    #calculate policy based on found Q-values
    policy = Q.argmax(axis=0)
    for ss in range(mdp.S):
        if mdp.verbose:
            print("State {}: {}".format(ss, Q[:,ss]))
        #check if numbers are all equal to max Q value for state
        if (np.sum(np.abs(Q[:,ss] - Q[:,ss].max())) < mdp.epsilon):
            policy[ss] = -1

    return policy
