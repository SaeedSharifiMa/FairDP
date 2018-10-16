import numpy as np
import cvxopt as cvx
 
def dp_eq_odds(pred, protected, label, epsilon, beta):
    m = len(label) #number of samples
    stacked_df = pd.DataFrame( np.column_stack((np.asarray(pred),np.asarray(protected),np.asarray(protected))) ,
                               columns = ["Yhat", "A", "Y"])
    q_dp = np.empty(shape=(2,2,2))
    for yhat in [0,1]:
        for a in [0,1]:
            for y in [0,1]:
                q_dp[yhat, a, y] = np.mean( (stacked_df.Yhat == yhat) & (stacked_df.A == a) & (stacked_df.Y == y) ) 
                                    #+ np.random.laplace(scale= 2/(m*epsilon))
    
    fpr0_dp = q_dp[1,0,0]/(q_dp[1,0,0] + q_dp[0,0,0])
    fpr1_dp = q_dp[1,1,0]/(q_dp[1,1,0] + q_dp[0,1,0])
    tpr0_dp = q_dp[1,0,1]/(q_dp[1,0,1] + q_dp[0,0,1])
    tpr1_dp = q_dp[1,1,1]/(q_dp[1,1,1] + q_dp[0,1,1])
    
    minq_dp = np.min([q_dp[1,0,0] + q_dp[0,0,0], q_dp[1,1,0] + q_dp[0,1,0], q_dp[1,0,1] + q_dp[0,0,1], q_dp[1,1,1] + q_dp[0,1,1]])
    
    constraint_bound = (8*np.log(8/beta))/(minq_dp*m*epsilon - 4*np.log(8/beta))
                                           
    A = cvx.matrix([ [fpr0_dp-1, 1-fpr0_dp, tpr0_dp-1, 1-tpr0_dp, 1, -1, 0, 0, 0, 0, 0, 0],
                 [1-fpr1_dp, fpr1_dp-1, 1-tpr1_dp, tpr1_dp-1, 0, 0, 1, -1, 0, 0, 0, 0],
                 [-fpr0_dp, fpr0_dp, -tpr0_dp, tpr0_dp, 0, 0, 0, 0, 1, -1, 0, 0],
                 [fpr1_dp, -fpr1_dp, tpr1_dp, -tpr1_dp, 0, 0, 0, 0, 0, 0, 1, -1]
               ])
    b = cvx.matrix([ constraint_bound, constraint_bound, constraint_bound, constraint_bound, 1, 0, 1, 0, 1, 0, 1, 0 ])
    c = cvx.matrix([ q_dp[0,0,0] - q_dp[0,0,1], q_dp[0,1,0] - q_dp[0,1,1], q_dp[1,0,0] - q_dp[1,0,1], q_dp[1,1,0] - q_dp[1,1,1]])
    sol=cvx.solvers.lp(c,A,b)

    return(sol['x'], constraint_bound, q_dp)