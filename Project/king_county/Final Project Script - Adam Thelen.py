import numpy as np
from sklearn.preprocessing import StandardScaler
import itertools
import matplotlib.pyplot as plt


# load in the data from the CSV file
data = np.loadtxt('Data Updated with Removed Variables.csv' , delimiter=',' , skiprows=1)


# normalize the data per the CS229 notes on PCA
# zero mean, unit variance
scaler = StandardScaler()
normalized_data = scaler.fit_transform(data)


# split the data into price and features
prices = normalized_data[:,0]
X = normalized_data[:,1:np.size(normalized_data, axis=1)]


# PCA function for computing the principal subspace and projecting onto it
def PCA(data):
    U,S,V = np.linalg.svd(data,full_matrices = False)
    V_transpose = np.transpose(V)
    B = np.dot(data,V_transpose)
    return B

# linear regression using pseudo-inverse
def linear_regression_pseudo_inv(X, y):
    theta_hat = np.dot(np.linalg.pinv(X), y)
    return theta_hat

# compute the prediction
def compute_y_hat(theta_hat, X):
    y_hat = np.dot(X, theta_hat)
    return y_hat

# compute the error
def compute_error(y_hat, y_test):
    normalized_test_MSE = (np.linalg.norm(y_test - y_hat))**2 / (np.linalg.norm(y_test))**2
    return normalized_test_MSE


##############################################################################
# begin testing multiple permutations of the features to find best one

# Referencing X:
# 0 - Bedrooms
# 1 - Bathrooms
# 2 - sqft_living
# 3 - sqft_lot
# 4 - floors
# 5 - sqft_above
# 6 - sqft_base
# 7 - yr_built
# 8 - zipcode
# 9 - lat
# 10 - long
# 11 - sqft_living15
# 12 - sqft_lot15
# 13 - house_age

# this specifies each of the experiments that will be performed
# they are chosen via a permutation "14 choose 4"
features = {'feature1': [0,1,2,3,4,5,6,7,8,9,10,11,12,13],
            'feature2': [0,1,2,3,4,5,6,7,8,9,10,11,12,13],
            'feature3': [0,1,2,3,4,5,6,7,8,9,10,11,12,13],
            'feature4': [0,1,2,3,4,5,6,7,8,9,10,11,12,13],
            }
keys, values = zip(*features.items())
experiments = [dict(zip(keys, v)) for v in itertools.product(*values)]
len(experiments)


# run regression on selected experiments
def run_selected_features(experiments, X, y):
    
    Test_MSE = np.zeros((len(experiments),1))
    for i in range(0,len(experiments)):
        experiment = experiments[i]
        features = list(experiment.values())
        X_short = X[:,features]
        
        
        # define the test and training data
        # 20613 training points, 1000 test points
        X_train = X_short[0:20613,:]
        X_test = X_short[20613:21613,:]
        y_train = prices[0:20613]
        y_test = prices[20613:21613]
        
        theta_hat = linear_regression_pseudo_inv(X_train, y_train)
        y_hat = compute_y_hat(theta_hat, X_test)
        Test_MSE[i] = compute_error(y_hat, y_test)
        
    return Test_MSE

Test_MSE = run_selected_features(experiments, X, prices)


# plot the Test-MSE for the experiments
xvalues = np.arange(0,len(Test_MSE),1)
plt.plot(xvalues, Test_MSE)
plt.ylabel('Normalized Test-MSE')
plt.xlabel('Experiment Index')
plt.title('Test-MSE vs Experiment Index (Permutations: 14 choose 4)')
plt.show()


# extract the index of the 200 smallest Test-MSE values
k = 200
idx = np.argpartition(Test_MSE, k, axis=0)
idx = np.reshape(idx[0:k], (k))

# find the Test-MSE of the corresponding index
corresponding_Test_MSE = Test_MSE[idx].flatten()

def extract_corresponding_features(experiments, idx):
    corresponding_experiment = np.zeros((len(idx),1))
    corresponding_features = np.zeros((len(idx),4))
    for i in range(0,len(idx)):
        corresponding_experiment = experiments[idx[i]]
        corresponding_features[i] = list(corresponding_experiment.values())
    
    
    return corresponding_features

corresponding_features = extract_corresponding_features(experiments, idx)

# plot the results in an easy to interpret way
# define the columns for the bar chart
fig = plt.figure()
ax = fig.add_axes([0,0,1,1])
columns = ['bedrooms', 
           'bathrooms',
           'sqft_living',
           'sqft_lot',
           'floors',
           'sqft_above',
           'swft_base',
           'yr_built',
           'zipcode',
           'lat',
           'long',
           'sqft_living15',
           'sqft_lot15',
           'house_age',
           ]

# count the times a feature occurs in the best k number of experiments
def count_occurances_of_each_feature(corresponding_features):
    occurances_0 = np.count_nonzero(corresponding_features == 0)
    occurances_1 = np.count_nonzero(corresponding_features == 1)
    occurances_2 = np.count_nonzero(corresponding_features == 2)
    occurances_3 = np.count_nonzero(corresponding_features == 3)
    occurances_4 = np.count_nonzero(corresponding_features == 4)
    occurances_5 = np.count_nonzero(corresponding_features == 5)
    occurances_6 = np.count_nonzero(corresponding_features == 6)
    occurances_7 = np.count_nonzero(corresponding_features == 7)
    occurances_8 = np.count_nonzero(corresponding_features == 8)
    occurances_9 = np.count_nonzero(corresponding_features == 9)
    occurances_10 = np.count_nonzero(corresponding_features == 10)
    occurances_11 = np.count_nonzero(corresponding_features == 11)
    occurances_12 = np.count_nonzero(corresponding_features == 12)
    occurances_13 = np.count_nonzero(corresponding_features == 13)
    
    occurances = [occurances_0,occurances_1,occurances_2,occurances_3,occurances_4,occurances_5,occurances_6,occurances_7,occurances_8,occurances_9,occurances_10,occurances_11,occurances_12,occurances_13]
    
    return occurances

# gather the occurances and plot the data as a bar graph
occurances = count_occurances_of_each_feature(corresponding_features)
ax.bar(columns, occurances)
ax.set_title('Feature occurance in the 200 experiments with lowest Test-MSE')
ax.set_ylabel('Count')
plt.show()


# looking at the resulting bar chart, the features that appear the most are
# bedrooms, sqft_living, sqft_above, and lat. We will take these four features 
# and run linear regression on them to get a baseline to compare PCA
# and L1/L2 regularization to.

# selected features
selected_features = [0, 2, 5, 9]

# Run the top four features through linear regression as a baseline
def run_linear_regression(X, y, selected_features):
    X_short = X[:,selected_features]
    
    # define the test and training data
    # 20613 training points, 1000 test points
    X_train = X_short[0:20613,:]
    X_test = X_short[20613:21613,:]
    y_train = prices[0:20613]
    y_test = prices[20613:21613]
        
    theta_hat = linear_regression_pseudo_inv(X_train, y_train)
    y_hat = compute_y_hat(theta_hat, X_test)
    Test_MSE = compute_error(y_hat, y_test)
    return Test_MSE

Baseline_Test_MSE = run_linear_regression(X, prices, selected_features)
print('Baseline Test-MSE: %r' %Baseline_Test_MSE)


# Now that we have the baseline Test-MSE determined via our best judgement,
# we would like to compare it with PCA and L1/L2 regularization to investigate
# further.

# Begin by using PCA on the dataset
B = PCA(X)

# loop over r and record the Test-MSE each time
def loop_over_r(B, y):
    Test_MSE = np.zeros((np.size(B, axis=1),1))
    for i in range(0,len(Test_MSE)):
        B_short = B[:,0:i]
        
        X_train = B_short[0:20613,:]
        X_test = B_short[20613:21613,:]
        y_train = prices[0:20613]
        y_test = prices[20613:21613]
        
        theta_hat = linear_regression_pseudo_inv(X_train, y_train)
        y_hat = compute_y_hat(theta_hat, X_test)
        Test_MSE[i] = compute_error(y_hat, y_test)
    return Test_MSE

Test_MSE_PCA = loop_over_r(B, prices)

# plot the results to determine the best value of r
xvalues = np.arange(0,len(Test_MSE_PCA),1)
plt.plot(xvalues, Test_MSE_PCA)
plt.ylabel('Normalized Test-MSE')
plt.xlabel('r')
plt.title('Test-MSE as a function of r (PCA)')
plt.show()





