import numpy as np
import pandas as pd

def center(X):
    for col in X.columns:
        X.loc[:, col] = X.loc[:, col]-np.mean(X.loc[:, col])
    return X


def add_intercept(X):
    """Add all 1's column to predictor matrix"""
    X['intercept'] = [1]*X.shape[0]


def one_hot_code(df1, sens_dict):
    cols = df1.columns
    for c in cols:
        if isinstance(df1[c][0], str):
            column = df1[c]
            df1 = df1.drop(c, 1)
            unique_values = list(set(column))
            n = len(unique_values)
            if n > 2:
                for i in range(n):
                    col_name = '{}.{}'.format(c, i)
                    col_i = [1 if el == unique_values[i] else 0 for el in column]
                    df1[col_name] = col_i
                    sens_dict[col_name] = sens_dict[c]
                del sens_dict[c]
            else:
                col_name = c
                col = [1 if el == unique_values[0] else 0 for el in column]
                df1[col_name] = col
    return df1, sens_dict

# num_sens in 1:18
def clean_communities(num_sens):
    """Clean communities & crime data set."""
    # Data Cleaning and Import
    df = pd.read_csv('dataset/communities.csv')
    df = df.fillna(0)
    y = df['ViolentCrimesPerPop']
    q_y = np.percentile(y, 70)   ################### 70 or 20 ????????????????????????? #####################
    # convert y's to binary predictions on whether the neighborhood is
    # especially violent
    y = [np.round((1 + np.sign(s - q_y)) / 2) for s in y]
    X = df.iloc[:, 0:122]
    # hot code categorical variables
    sens_df = pd.read_csv('dataset/communities_protected.csv')
    sens_cols = [str(c) for c in sens_df.columns if sens_df[c][0] == 1]
    #print('sensitive features: {}'.format(sens_cols))
    sens_dict = {c: 1 if c in sens_cols else 0 for c in df.columns}
    df, sens_dict = one_hot_code(df, sens_dict)
    sens_names = [key for key in sens_dict.keys() if sens_dict[key] == 1]
    #print('there are {} sensitive features including derivative features'.format(len(sens_names)))
    x_prime = df[sens_names[num_sens-1]].copy()
    x_prime = 1*(x_prime > np.median(x_prime))
    X = X.drop(sens_names[num_sens-1], axis = 1)
    return X, x_prime, pd.Series(y)


# num_sens in 1:11
def clean_lawschool(num_sens):
    """Clean law school data set."""
    # Data Cleaning and Import
    df = pd.read_csv('dataset/lawschool.csv')
    df = df.dropna()
    # convert categorical column variables to 0,1
    df['gender'] = df['gender'].map({'female': 1, 'male': 0})
    # remove y from df
    df_y = df['bar1']
    df = df.drop('bar1', 1)
    y = [int(a == 'P') for a in df_y]
    y = pd.Series(y)
    sens_df = pd.read_csv('dataset/lawschool_protected.csv')
    sens_cols = [str(c) for c in sens_df.columns if sens_df[c][0] == 1]
    sens_dict = {c: 1 if c in sens_cols else 0 for c in df.columns}
    # one hot coding of race variable
    for i in range(1, 9):
        col_name = 'race{}'.format(i)
        if 'race' in sens_cols:
            sens_dict[col_name] = 1
        else:
            sens_dict[col_name] = 0
        race_code = [np.int(r == i) for r in df['race']]
        df[col_name] = race_code
    sens_dict['race'] = 0
    df = df.drop('race', 1)
    sens_names = [key for key in sens_dict.keys() if sens_dict[key] == 1]
    #print('there are {} sensitive features including derivative features'.format(len(sens_names)))
    x_prime = df[sens_names].copy()
    x_prime.age = 1*(x_prime.age > np.median(x_prime.age)) ########## OK ??????????? ##############
    x_prime.fam_inc = 1*(x_prime.fam_inc > np.median(x_prime.fam_inc)) ########## OK ????????? ##############
    x_prime = x_prime[sens_names[num_sens-1]]
    df = df.drop(sens_names[num_sens-1], axis = 1)
    df.index = range(len(df))
    x_prime.index = range(len(x_prime))
    return df, x_prime, pd.Series(y)

# num_sens 1:7
def clean_adult(num_sens):
    df = pd.read_csv('dataset/adult.csv')
    df = df.dropna()
    # binarize and remove y value
    df['income'] = df['income'].map({' <=50K': 0, ' >50K': 1})
    y = df['income']
    df = df.drop('income', 1)
    # hot code categorical variables
    sens_df = pd.read_csv('dataset/adult_protected.csv')
    sens_cols = [str(c) for c in sens_df.columns if sens_df[c][0] == 1]
    #print('sensitive features: {}'.format(sens_cols))
    sens_dict = {c: 1 if c in sens_cols else 0 for c in df.columns}
    df, sens_dict = one_hot_code(df, sens_dict)
    sens_names = [key for key in sens_dict.keys() if sens_dict[key] == 1]
    #print('there are {} sensitive features including derivative features'.format(len(sens_names)))
    x_prime = df[sens_names].copy()
    x_prime.age = 1*(x_prime.age > np.median(x_prime.age))
    x_prime = x_prime[sens_names[num_sens-1]]
    df = df.drop(sens_names[num_sens-1], axis = 1)
    return df, x_prime, y

# num_sens 1:5
# binarize the sensitive features
def clean_student(num_sens):
    df = pd.read_csv('dataset/student-mat.csv', sep=';')
    df = df.dropna()
    y = df['G3']
    y = [0 if y < 11 else 1 for y in y]
    df = df.drop(['G3', 'G2', 'G1'], 1)
    sens_df = pd.read_csv('dataset/student_protected.csv')
    sens_cols = [str(c) for c in sens_df.columns if sens_df[c][0] == 1]
    print('sensitive features: {}'.format(sens_cols))
    sens_dict = {c: 1 if c in sens_cols else 0 for c in df.columns}
    df, sens_dict = one_hot_code(df, sens_dict)
    sens_names = [key for key in sens_dict.keys() if sens_dict[key] == 1]
    print('there are {} sensitive features including derivative features'.format(len(sens_names)))
    x_prime = df[sens_names].copy()
    df = df.drop(sens_names, axis = 1)
    return df, x_prime, pd.Series(y)