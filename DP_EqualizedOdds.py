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
        temp_df = pd.DataFrame( np.column_stack((np.array(pred_label), np.array(protected_attr), np.array(true_label))) ,
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
            #print("You were unlucky! Try again.")
            return(0)

        A = cvx.matrix([ [fpr0_dp-1, 1-fpr0_dp, tpr0_dp-1, 1-tpr0_dp, 1, -1, 0, 0, 0, 0, 0, 0], #p00
                     [1-fpr1_dp, fpr1_dp-1, 1-tpr1_dp, tpr1_dp-1, 0, 0, 1, -1, 0, 0, 0, 0], #p01
                     [-fpr0_dp, fpr0_dp, -tpr0_dp, tpr0_dp, 0, 0, 0, 0, 1, -1, 0, 0], #p10
                     [fpr1_dp, -fpr1_dp, tpr1_dp, -tpr1_dp, 0, 0, 0, 0, 0, 0, 1, -1]]) #p11
        b = cvx.matrix([ constraint_bound, constraint_bound, constraint_bound, constraint_bound, 
                        1, 0, 1, 0, 1, 0, 1, 0 ])
        c = cvx.matrix([ q_dp[0,0,0] - q_dp[0,0,1], q_dp[0,1,0] - q_dp[0,1,1], # p00, p01
                        q_dp[1,0,0] - q_dp[1,0,1], q_dp[1,1,0] - q_dp[1,1,1]]) # p10, p11
        solvers.options['show_progress'] = False
        sol = cvx.solvers.lp(c,A,b)
        opt = np.array(sol['x']).reshape((2,2))
        return(opt)

    def postproc_label(pred_label, protected_attr, p):
        postproc_label = (p[ pred_label.tolist(), protected_attr.tolist() ] >= 0.5).astype(int)
        return(postproc_label)

    def error(pred_label, true_label):
        err = np.mean(pred_label != true_label)
        return(err)
    
    def qhat(pred_label, protected_attr, true_label):
        temp_df = pd.DataFrame( np.column_stack((np.array(pred_label), np.array(protected_attr), np.array(true_label))) ,
                                  columns = ["Yhat", "A", "Y"])
        qhat = np.empty(shape=(2,2,2))
        for yhat in [0,1]:
            for a in [0,1]:
                for y in [0,1]:
                    qhat[yhat, a, y] = np.mean( (temp_df.Yhat == yhat) & (temp_df.A == a) & (temp_df.Y == y) )
        return(qhat)

    def unfairness(pred_label, protected_attr, true_label):
        qhat = dp_fair_postproc.qhat(pred_label, protected_attr, true_label)
        fpr0 = qhat[1,0,0]/(qhat[1,0,0] + qhat[0,0,0])
        fpr1 = qhat[1,1,0]/(qhat[1,1,0] + qhat[0,1,0])
        tpr0 = qhat[1,0,1]/(qhat[1,0,1] + qhat[0,0,1])
        tpr1 = qhat[1,1,1]/(qhat[1,1,1] + qhat[0,1,1])
        unfairness = np.max([abs(fpr1-fpr0), abs(tpr1-tpr0)])
        return(unfairness)
    
#Example:
true_label = np.random.binomial(n=1,p=0.5,size=1000)
protected_attr = np.random.binomial(n=1,p=0.5,size=1000)
pred_label = np.random.binomial(n=1,p=0.5,size=1000)
epsilon = 0.1
beta = 0.05

counter = -1
p = 0
while(type(p) == int):
    counter = counter + 1
    p = dp_fair_postproc.eq_odds(pred, protected, label, epsilon, beta)
    if counter == 1000:
        print("epsilon is probably too small")
        break
postproc_label = dp_fair_postproc.postproc_label(pred_label, protected_attr, p)
err = dp_fair_postproc.error(postproc_label, true_label)
unf = dp_fair_postproc.unfairness(postproc_label, protected_attr, true_label)
qhat = dp_fair_postproc.qhat(pred_label, protected_attr, true_label)

print("error:", err)
print("unfairness:", unf)
print("number of failures:", counter)