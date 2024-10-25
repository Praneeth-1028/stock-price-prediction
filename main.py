import warnings
import os
import sys
import numpy as np
import matplotlib.pyplot as plt
from hmmlearn import hmm

warnings.filterwarnings("ignore", category=DeprecationWarning)

STOCKS = {"apple": 'data/apple.csv', "dell": 'data/dell.csv', "google": 'data/google.csv'}
NUM_TEST = 100
K = 50
NUM_ITERS=10000

DIR = 'performance'

if not os.path.exists(DIR):
    os.makedirs(DIR, exist_ok=True) 

possible_states = np.arange(2, 15)

for company, stock in STOCKS.items():
    dataset = np.genfromtxt(stock, delimiter = ',', skip_header = 1)
    predicted_stock_data = np.empty([0,dataset.shape[1]])
    aic_vect = np.empty([0,1])
    bic_vect = np.empty([0,1])
    likelihood = np.empty([0,1])

    output_file = f"{DIR}/{company}-performance.csv"
    for states in range(2,15):
        num_params = states**2 + 14*states # Transition Probabilities: N^2, Means: 4N, Covariance Matrices: d(d+1)/2 (because of symmetry), 4(5)/2 = 10N => N^2 + 14N

        model = hmm.GaussianHMM(n_components=states, covariance_type='full',tol=0.0001,n_iter=NUM_ITERS)
        model.fit(dataset[NUM_TEST:,:])
    
        if model.monitor_.iter == NUM_ITERS:
            print('Increase number of iterations')
            sys.exit(1)
        score = model.score(dataset)
        aic_vect = np.vstack((aic_vect, -2 * score + 2 * num_params))
        bic_vect = np.vstack((bic_vect, -2 * score +  num_params * np.log(dataset.shape[0])))
        likelihood = np.vstack((likelihood, score))

    data = np.column_stack((possible_states, likelihood, aic_vect, bic_vect))

    np.savetxt(output_file, data, delimiter=',', header='States,likelihood,aic_vect,bic_vect', comments='', fmt='%f')

    PLOT_DIR = "plots"

    
    if not os.path.exists(PLOT_DIR):
        os.makedirs(PLOT_DIR, exist_ok=True) 

    plt.figure()
    
    plt.plot(possible_states, aic_vect, label='AIC values', color='blue', linestyle='-') 
    plt.plot(possible_states, bic_vect, label='BIC values', color='red', linestyle='--')

    plt.title(f'Plot of AIC and BIC values for {company} stock')
    plt.xlabel('Number of states')
    plt.ylabel('AIC and BIC values')

    plt.grid(True)
    plt.legend(loc='upper right')

    plt.savefig(os.path.join(PLOT_DIR, f'{company}.png'))
    plt.close()

    opt_states = np.argmin(bic_vect) + 2
    print('Optimum number of states are {}'.format(opt_states))

