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


for company, stock in STOCKS.items():
    dataset = np.genfromtxt(stock, delimiter = ',', skip_header = 1)
    predicted_stock_data = np.empty([0,dataset.shape[1]])
    aic_vect = np.empty([0,1])
    bic_vect = np.empty([0,1])
    likelihood = np.empty([0,1])
    state_vect = np.empty([0,1])

    output_file = f"{DIR}/{company}-performance.csv"
    for states in range(2,15):
        num_params = states**2 + 10*states

        model = hmm.GaussianHMM(n_components=states, covariance_type='full',tol=0.0001,n_iter=NUM_ITERS)
        model.fit(dataset[NUM_TEST:,:])
    
        if model.monitor_.iter == NUM_ITERS:
            print('Increase number of iterations')
            sys.exit(1)
        score = model.score(dataset)
        aic_vect = np.vstack((aic_vect, -2 * score + 2 * num_params))
        bic_vect = np.vstack((bic_vect, -2 * score +  num_params * np.log(dataset.shape[0])))
        likelihood = np.vstack((likelihood, score))
        state_vect = np.vstack((state_vect, int(states)))

    data = np.column_stack((state_vect, likelihood, aic_vect, bic_vect))

    np.savetxt(output_file, data, delimiter=',', header='States,likelihood,aic_vect,bic_vect', comments='', fmt='%f')


    print({np.argmin(aic_vect)}, {np.argmin(bic_vect)})
    opt_states = np.argmin(bic_vect) + 2
    print('Optimum number of states are {}'.format(opt_states))





#     Iterations(N), Loglikelihood, AIC Value,   BIC Value
# Model is not converging.  Current: -15223.03556590746 is not greater than -15221.50475496995. Delta is -1.530810937509159
# Model is not converging.  Current: -14868.740114493396 is not greater than -14855.246232834157. Delta is -13.493881659238468
# Model is not converging.  Current: -12341.974488553487 is not greater than -12341.123854259653. Delta is -0.850634293834446
# Model is not converging.  Current: -12613.440247601082 is not greater than -12612.470890627435. Delta is -0.9693569736464269
# Model is not converging.  Current: -13986.804423952884 is not greater than -13979.65355221161. Delta is -7.150871741274386
# Model is not converging.  Current: -12761.71453841999 is not greater than -12760.704518594777. Delta is -1.0100198252130212
# Model is not converging.  Current: -10871.901752569695 is not greater than -10870.922886832432. Delta is -0.9788657372628222
# Model is not converging.  Current: -10516.053868974457 is not greater than -10515.144048827506. Delta is -0.9098201469514606
# Model is not converging.  Current: -12109.298310500993 is not greater than -12101.816467496621. Delta is -7.481843004372422
# Model is not converging.  Current: -10488.11211825845 is not greater than -10485.662683238006. Delta is -2.4494350204440707
# Model is not converging.  Current: -17101.75413976192 is not greater than -16530.168964117594. Delta is -571.5851756443262
# {np.int64(8)} {np.int64(8)}
# Optimum number of states are 10
# Iterations(N), Loglikelihood, AIC Value,   BIC Value
# Model is not converging.  Current: -7154.756550490363 is not greater than -7152.591898014699. Delta is -2.1646524756642975
# Model is not converging.  Current: -6634.7584003396305 is not greater than -6633.954204072591. Delta is -0.8041962670395151
# Model is not converging.  Current: -5359.116804073249 is not greater than -5358.935411596297. Delta is -0.1813924769521691
# Model is not converging.  Current: -6242.64866879311 is not greater than -6242.647616018055. Delta is -0.001052775055541133
# Model is not converging.  Current: -4513.273992718368 is not greater than -4513.040865755784. Delta is -0.23312696258471988
# Model is not converging.  Current: -5730.563815423961 is not greater than -5729.482690910655. Delta is -1.0811245133063494
# Model is not converging.  Current: -5607.620405255717 is not greater than -5605.9325287919855. Delta is -1.6878764637312997
# Model is not converging.  Current: -3839.9738189765476 is not greater than -3837.4161305703424. Delta is -2.557688406205216
# Model is not converging.  Current: -3736.7055693975567 is not greater than -3734.657120178461. Delta is -2.048449219095801
# Model is not converging.  Current: -5009.61070943127 is not greater than -5009.481683550151. Delta is -0.12902588111955993
# Model is not converging.  Current: -4785.798954021492 is not greater than -4785.735065515991. Delta is -0.06388850550138159
# Model is not converging.  Current: -4692.416107193219 is not greater than -4688.089846667858. Delta is -4.326260525361249
# Model is not converging.  Current: -7409.383621053307 is not greater than -5340.0018531152145. Delta is -2069.3817679380927
# {np.int64(8)} {np.int64(7)}
# Optimum number of states are 9
# Iterations(N), Loglikelihood, AIC Value,   BIC Value
# Model is not converging.  Current: -14542.116648416293 is not greater than -14535.276629779353. Delta is -6.840018636939931
# Model is not converging.  Current: -13384.585196053866 is not greater than -13384.312363428653. Delta is -0.27283262521268625
# Model is not converging.  Current: -13077.284041497587 is not greater than -13077.162336235693. Delta is -0.12170526189402153
# Model is not converging.  Current: -12861.656228918828 is not greater than -12861.382979498429. Delta is -0.2732494203992246
# Model is not converging.  Current: -11584.218556920892 is not greater than -11581.905048273733. Delta is -2.313508647159324
# Model is not converging.  Current: -12345.801512841455 is not greater than -12343.498391134888. Delta is -2.3031217065672536
# Model is not converging.  Current: -11075.100912021073 is not greater than -11064.120504057832. Delta is -10.980407963241305
# Model is not converging.  Current: -13007.777449008487 is not greater than -12999.039684737942. Delta is -8.737764270545085
# Model is not converging.  Current: -10290.953184500497 is not greater than -10274.630541175853. Delta is -16.32264332464365
# Model is not converging.  Current: -11666.350739190118 is not greater than -11665.442619999933. Delta is -0.9081191901859711
# Model is not converging.  Current: -11389.44591177344 is not greater than -11387.958542378241. Delta is -1.4873693951994937
# Model is not converging.  Current: -11719.18872304185 is not greater than -11718.551397007242. Delta is -0.6373260346081224
# {np.int64(9)} {np.int64(9)}
# Optimum number of states are 11