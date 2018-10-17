import numpy as np
import cvxopt as cvx

class dp_fair_postproc:
    
    def eq_odds(pred_label, protected_attr, true_label, epsilon, beta):

        # This function ...

        # Inputs:
        # pred_label: Yhat
        # protected_attribute: A
        # true_label: Y
        # epsilon: privacy parameter
        # beta: confidence parameter

        # Outputs: vector p_{yhat, a} = Pr[Ytilde = 1 | Yhat = yhat, A = a]

        m = len(true_label) #number of samples
        temp_df = pd.DataFrame( np.column_stack((np.asarray(pred_label), np.asarray(protected_attr), np.asarray(true_label))) ,
                                  columns = ["Yhat", "A", "Y"])
        q_dp = np.empty(shape=(2,2,2))
        for yhat in [0,1]:
            for a in [0,1]:
                for y in [0,1]:
                    q_dp[yhat, a, y] = np.mean( (temp_df.Yhat == yhat) & (temp_df.A == a) & (temp_df.Y == y) ) \
                                        + np.random.laplace(scale= 2/(m*epsilon))

        fpr0_dp = q_dp[1,0,0]/(q_dp[1,0,0] + q_dp[0,0,0])
        fpr1_dp = q_dp[1,1,0]/(q_dp[1,1,0] + q_dp[0,1,0])
        tpr0_dp = q_dp[1,0,1]/(q_dp[1,0,1] + q_dp[0,0,1])
        tpr1_dp = q_dp[1,1,1]/(q_dp[1,1,1] + q_dp[0,1,1])

        minq_dp = np.min([q_dp[1,0,0] + q_dp[0,0,0], q_dp[1,1,0] + q_dp[0,1,0], 
                          q_dp[1,0,1] + q_dp[0,0,1], q_dp[1,1,1] + q_dp[0,1,1]])

        constraint_bound = (8*np.log(8/beta))/(minq_dp*m*epsilon - 4*np.log(8/beta))
        if constraint_bound < 0:
            print("You were unlucky! Try again.")
            return

        A = cvx.matrix([ [fpr0_dp-1, 1-fpr0_dp, tpr0_dp-1, 1-tpr0_dp, 1, -1, 0, 0, 0, 0, 0, 0],
                     [1-fpr1_dp, fpr1_dp-1, 1-tpr1_dp, tpr1_dp-1, 0, 0, 1, -1, 0, 0, 0, 0],
                     [-fpr0_dp, fpr0_dp, -tpr0_dp, tpr0_dp, 0, 0, 0, 0, 1, -1, 0, 0],
                     [fpr1_dp, -fpr1_dp, tpr1_dp, -tpr1_dp, 0, 0, 0, 0, 0, 0, 1, -1]
                   ])
        b = cvx.matrix([ constraint_bound, constraint_bound, constraint_bound, constraint_bound, 1, 0, 1, 0, 1, 0, 1, 0 ])
        c = cvx.matrix([ q_dp[0,0,0] - q_dp[0,0,1], q_dp[0,1,0] - q_dp[0,1,1], q_dp[1,0,0] - q_dp[1,0,1], q_dp[1,1,0] - q_dp[1,1,1]])
        solvers.options['show_progress'] = False
        sol=cvx.solvers.lp(c,A,b)

        return(np.array(sol['x']).tolist())

    def postproc_label(pred_label, protected_attr, p):
        return

    def error(pred_label, true_label):
        return

    def unfairness(pred_label, protected_attribute, true_label):
        return
    
#Example:
pred = np.random.binomial(n=1,p=0.5,size=1000)
protected = np.random.binomial(n=1,p=0.5,size=1000)
label = pred
epsilon = 0.1
beta = 0.1
dp_fair_postproc.eq_odds(pred, protected, label, epsilon, beta)