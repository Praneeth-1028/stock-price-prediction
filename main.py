import warnings
import os
import sys
import numpy as np
import matplotlib.pyplot as plt
from hmmlearn import hmm
import data

warnings.filterwarnings("ignore", category=DeprecationWarning)

STOCKS = {"apple": 'data/apple.csv', "dell": 'data/dell.csv', "google": 'data/google.csv'}

NUM_TEST = 100
K = 50
NUM_ITERS=10000

PLOT_SHOW = True

DIR = 'performance'
DIR_1 = 'prediction'

if not os.path.exists(DIR):
    os.makedirs(DIR, exist_ok=True) 

if not os.path.exists(DIR_1):
    os.makedirs(DIR_1, exist_ok=True) 

possible_states = np.arange(2, 15)

def calc_mape(predicted_data, true_data):
    return np.divide(np.sum(np.divide(np.absolute(predicted_data - true_data), true_data), 0), true_data.shape[0])

for company, stock in STOCKS.items():
    dataset = np.genfromtxt(stock, delimiter = ',', skip_header = 2)
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

    PLOT_DIR_1 = "plots/aic-bic"

    if not os.path.exists(PLOT_DIR_1):
        os.makedirs(PLOT_DIR_1, exist_ok=True) 

    plt.figure()
    
    plt.plot(possible_states, aic_vect, label='AIC values', color='blue', linestyle='-') 
    plt.plot(possible_states, bic_vect, label='BIC values', color='red', linestyle='--')

    plt.title(f'Plot of AIC and BIC values for {company} stock')
    plt.xlabel('Number of states')
    plt.ylabel('AIC and BIC values')

    plt.grid(True)
    plt.legend(loc='upper right')

    plt.savefig(os.path.join(PLOT_DIR_1, f'{company}.png'))
    plt.close()

    opt_states = np.argmin(bic_vect) + 2
    print('Optimum number of states are {}'.format(opt_states))



    for idx in reversed(range(NUM_TEST)):
        train_data = dataset[idx + 1:,:]
        test_data = dataset[idx,:]
        length = train_data.shape[0]

        if idx == NUM_TEST - 1:
            model = hmm.GaussianHMM(n_components = opt_states, covariance_type = 'full', tol = 0.0001, n_iter = NUM_ITERS, init_params = 'stmc')
        else:
            model = hmm.GaussianHMM(n_components=opt_states, covariance_type='full', tol=0.0001, n_iter=NUM_ITERS, init_params='')
            model.transmat_ = transmat 
            model.startprob_ = startprob
            model.means_ = means
            model.covars_ = covars

        model.fit(np.flipud(train_data))

        transmat = model.transmat_
        startprob = model.startprob_
        means = model.means_
        covars = model.covars_

        if model.monitor_.iter == NUM_ITERS:
            print('Increase number of iterations')
            sys.exit(1)

        num = 1;
        past_likelihood = []
        curr_likelihood = model.score(np.flipud(train_data[0:K - 1, :]))

        while num < (length / K - 1):
            past_likelihood = np.append(past_likelihood, model.score(np.flipud(train_data[num:num + K - 1, :])))
            num = num + 1

        likelihood_diff = np.argmin(np.absolute(past_likelihood - curr_likelihood))
        predicted_change = train_data[likelihood_diff,:] - train_data[likelihood_diff + 1,:]
        predicted_stock_data = np.vstack((predicted_stock_data, dataset[idx + 1,:] + predicted_change))

    np.savetxt(f'{ DIR_1}/{company}_forecast.csv', predicted_stock_data,delimiter = ',',fmt = '%.2f')

    mape = calc_mape(predicted_stock_data, np.flipud(dataset[range(100),:]))

    PLOT_DIR_2 = 'results'

    if not os.path.exists(PLOT_DIR_2):
        os.makedirs(PLOT_DIR_2, exist_ok=True) 

    print('MAPE for the {} stock is '.format(company),mape)
    np.savetxt(f'results/{company}_results.csv', mape)


    PLOT_DIR_3 = 'plots/prediction'

    if not os.path.exists(PLOT_DIR_3):
        os.makedirs(PLOT_DIR_3, exist_ok=True) 

    LABELS = ['High', 'Low', 'Open', 'Close']

    # Plot predicted vs actual stock prices
    for i, label in enumerate(LABELS):
        plt.figure()
        plt.plot(range(100), predicted_stock_data[:, i], 'k-', label=f'Predicted {label} price')
        plt.plot(range(100), np.flipud(dataset[range(100), i]), 'r--', label=f'Actual {label} price')
        plt.xlabel('Time steps')
        plt.ylabel('Price')
        plt.title(f'{label} price for {company}')
        plt.grid(True)
        plt.legend(loc='upper left')
        
        # Save the plot
        plt.savefig(os.path.join(PLOT_DIR_3, f'{label}_price_{company}.png'))
        plt.close()
        

    if PLOT_SHOW:
        plt.show(block=False)
