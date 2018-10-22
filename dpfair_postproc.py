import numpy as np
import pandas as pd
import cvxopt as cvx

class dp_fair_postproc:
    
    def eq_odds(Yhat, A, Y, epsilon, gamma, beta):

        # This function ...

        # Inputs:
        # Yhat
        # A: binary
        # Y
        # epsilon: privacy parameter
        # gamma: fairness violation
        # beta: confidence parameter

        # Outputs: vector p_{yhat, a} = Pr[Ytilde = 1 | Yhat = yhat, A = a]

        m = len(Y) #number of samples
        q_dp = dp_fair_postproc.qhat(Yhat, A, Y) + np.random.laplace(scale= 2/(m*epsilon), size=8).reshape((2,2,2))
        
        minq_dp0 = np.min([q_dp[1,0,0] + q_dp[0,0,0], q_dp[1,1,0] + q_dp[0,1,0]]) 
        minq_dp1 = np.min([q_dp[1,0,1] + q_dp[0,0,1], q_dp[1,1,1] + q_dp[0,1,1]])
        fpr_bound = gamma + (4*np.log(8/beta))/(minq_dp0*m*epsilon)
        tpr_bound = gamma + (4*np.log(8/beta))/(minq_dp1*m*epsilon)
        if ( (fpr_bound - gamma < 0) or (tpr_bound - gamma < 0) ):
            return(0,0,0)
        
        rates_dp = dp_fair_postproc.rates(q_dp)
        fpr0_dp = rates_dp[0]
        fpr1_dp = rates_dp[1]
        tpr0_dp = rates_dp[2]
        tpr1_dp = rates_dp[3]

        B = cvx.matrix([ [fpr0_dp-1, 1-fpr0_dp, tpr0_dp-1, 1-tpr0_dp, 1, -1, 0, 0, 0, 0, 0, 0], #p00
                     [1-fpr1_dp, fpr1_dp-1, 1-tpr1_dp, tpr1_dp-1, 0, 0, 1, -1, 0, 0, 0, 0], #p01
                     [-fpr0_dp, fpr0_dp, -tpr0_dp, tpr0_dp, 0, 0, 0, 0, 1, -1, 0, 0], #p10
                     [fpr1_dp, -fpr1_dp, tpr1_dp, -tpr1_dp, 0, 0, 0, 0, 0, 0, 1, -1]], (12,4), 'd') #p11
        b = cvx.matrix([ fpr_bound, fpr_bound, tpr_bound, tpr_bound, 
                        1, 0, 1, 0, 1, 0, 1, 0 ], (12,1), 'd')
        c = cvx.matrix([ q_dp[0,0,0] - q_dp[0,0,1], q_dp[0,1,0] - q_dp[0,1,1], # p00, p01
                        q_dp[1,0,0] - q_dp[1,0,1], q_dp[1,1,0] - q_dp[1,1,1]], (4,1), 'd') # p10, p11
        cvx.solvers.options['show_progress'] = False
        sol = cvx.solvers.lp(c,B,b)
        opt = np.array(sol['x']).reshape((2,2))
        opt[opt > 1] = 1
        opt[opt < 0] = 0
        return(opt, fpr_bound, tpr_bound)

    def postproc_label(Yhat, A, p):
        Ytilde = np.random.binomial( 1, p[ Yhat.astype(int).tolist(), A.values.reshape((1,len(Yhat))).tolist()[0] ] )
        return(Ytilde)

    def error(Yhat, Y):
        err = np.mean(Yhat != Y)
        return(err)
    
    def qhat(Yhat, A, Y):
        temp_df = pd.DataFrame( np.column_stack((np.array(Yhat), np.array(A), np.array(Y))), columns = ["Yhat", "A", "Y"])
        qhat = np.empty(shape=(2,2,2))
        for yhat in [0,1]:
            for a in [0,1]:
                for y in [0,1]:
                    qhat[yhat, a, y] = np.mean( (temp_df.Yhat == yhat) & (temp_df.A == a) & (temp_df.Y == y) )
        return(qhat)
    
    def rates(q):
        fpr0 = q[1,0,0]/(q[1,0,0] + q[0,0,0])
        fpr1 = q[1,1,0]/(q[1,1,0] + q[0,1,0])
        tpr0 = q[1,0,1]/(q[1,0,1] + q[0,0,1])
        tpr1 = q[1,1,1]/(q[1,1,1] + q[0,1,1])
        return [fpr0, fpr1, tpr0, tpr1]
    
    def unfairness(Yhat, A, Y):
        qhat = dp_fair_postproc.qhat(Yhat, A, Y)
        rates = dp_fair_postproc.rates(qhat)
        unfairness_fpr = abs(rates[0]-rates[1])
        unfairness_tpr = abs(rates[2]-rates[3])
        return unfairness_fpr, unfairness_tpr

# num_sens in 1:18
def clean_communities(num_sens):
    # Data Cleaning and Import
    df = pd.read_csv('dataset/communities.csv')
    df = df.fillna(0)

    # sensitive variables are just racial distributions in the population and police force as well as foreign status
    # median income and pct of illegal immigrants / related variables are not labeled sensitive
    sens_features = [3, 4, 5, 6, 22, 23, 24, 25, 26, 27, 61, 62, 92, 105, 106, 107, 108, 109]
    df_sens = df.iloc[:, [sens_features[num_sens-1]]]
    y = df['ViolentCrimesPerPop']
    q_y = np.percentile(y, 20)
    # convert y's to binary predictions on whether the neighborhood is
    # especially violent
    y = pd.Series([np.round((1 + np.sign(s - q_y)) / 2) for s in y])
    X = df.iloc[:, 0:122]
    X = X.drop(X.columns[sens_features[num_sens-1]], axis = 1)
    X_prime = df_sens
    for s in range(1):
        median = np.median(df_sens.iloc[:, s])
        X_prime.iloc[:, s] = 1*(X_prime > median)
    return X, X_prime, y

# num_sens 1:9
def clean_lawschool(num_sens):
    # Data Cleaning and Import
    df = pd.read_csv('dataset/lawschool.csv')
    df = df.dropna()
    # convert categorical column variables to 0,1
    df['gender'] = df['gender'].map({'female': 1, 'male': 0})
    # remove y from df
    df_y = df['bar1']
    df = df.drop('bar1', 1)
    y = [int(a == 'P') for a in df_y]
    # one hot coding of race variable
    for i in range(1,9):
        col_name = 'race{}'.format(i)
        race_code = [np.int(r == i) for r in df['race']]
        df[col_name] = race_code
    df = df.drop('race', 1)
    # sensitive variables are just racial distributions in the population and police force as well as foreign status
    # median income and pct of illegal immigrants / related variables are not labeled sensitive
    sens_features = range(8, 17)
    x_prime = df.iloc[:, sens_features[num_sens-1]]
    df = df.drop(df.columns[sens_features[num_sens-1]], axis = 1)
    return df, x_prime, pd.Series(y)

#Example:
from sklearn.linear_model import LogisticRegression as LR
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier as RF

X, A, Y = clean_lawschool(1)
#X = X.iloc[:,2:5]
epsilon = 0.25
gamma = 0.01
beta = 0.05
m = Y.shape[0]
base_clf = LR(random_state=0).fit(X, Y)
Yhat = base_clf.predict(X)
errhat = dp_fair_postproc.error(Yhat, Y)
unfhat = dp_fair_postproc.unfairness(Yhat, A, Y)
qhat = dp_fair_postproc.qhat(Yhat, A, Y)
minq = np.min([qhat[1,0,0] + qhat[0,0,0], qhat[1,1,0] + qhat[0,1,0], 
            qhat[1,0,1] + qhat[0,0,1], qhat[1,1,1] + qhat[0,1,1]])
print("error before postprocessing:", errhat)
print("unfairness before postprocessing:", unfhat)
print("minq:", minq, ", assumption:", 4*np.log(8/beta)/(m*epsilon))

counter = -1 # counts the number of failures
maxcounter = 100
p = 0
while(type(p) == int):
    counter = counter + 1
    p, fpr_bound, tpr_bound = dp_fair_postproc.eq_odds(Yhat, A, Y, epsilon, gamma, beta)
    if counter == maxcounter:
        print("epsilon is probably too small!")
        break
if counter < maxcounter:
    Ytilde = dp_fair_postproc.postproc_label(Yhat, A, p)
    errtilde = dp_fair_postproc.error(Ytilde, Y)
    unftilde = dp_fair_postproc.unfairness(Ytilde, A, Y)
    print("\n error after postprocessing:", errtilde)
    print("unfairness after postprocessing:", unftilde)
    print("number of failures in the algorithm:", counter)
    print("\n p:", p)
    print("fpr bound:", fpr_bound)
    print("tpr bound:", tpr_bound)