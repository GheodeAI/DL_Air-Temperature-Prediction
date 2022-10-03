import numpy as np
import pandas as pd

import Methods
import Data_read
from sklearn.metrics import mean_squared_error
import sys
sys.path.append('../')
from niapy.algorithms.modified import SelfAdaptiveDifferentialEvolution
from niapy.task import Task
from niapy.problems.problem import Problem



class run_optimization(Problem):
    def _evaluate(self, x):
        x = np.rint(x).astype(int)
        if x[:9].sum() == 0:
            x[np.random.randint(0, 9)] = 1
        X, X_rp, X_cnn, y, y_cnn, y_orig, scaler, y_max, y_min, y_cnn_max, y_cnn_min, perc = Data_read.init(city=self.kwargs['city'], week2period=self.kwargs['week2period'], perc=10)
        mt = Methods.Methods(X, X_rp, X_cnn, y, y_cnn, y_max, y_min, y_cnn_max, y_cnn_min)
        feats = np.argwhere(x[:9] == 1).ravel()

        pred = pd.DataFrame()
        for i in range(self.kwargs['runs']):
            func = getattr(mt, self.kwargs['methodology'])
            temp, param = func(feats=feats, f1=32, f2=64, f3=128, k1=2, k2=2, k3=2, e=100, i=i)

            for t in range(20):
                temp[t] = temp[t] * (scaler[52 + t, 0, 0] - scaler[52 + t, 0, 1]) + scaler[52 + t, 0, 1]
            pred[self.kwargs['methodology'] + str(i)] = temp.ravel()
        error = mean_squared_error(y_orig[52:], pred.mean(axis=1))
        pred.attrs = param
        pred.attrs['perc'] = perc
        pred.to_pickle('results/' + self.kwargs["methodology"] + '/FS/' + str(error) + '_' + self.kwargs["city"] +
                       '_' + str(self.kwargs['week2period']) + '.pkl')
        return error


def jde(city, week2period, methodology, runs):
    # we will run jDE algorithm for 1 independent run
    algo = SelfAdaptiveDifferentialEvolution(f_lower=0.0, f_upper=2.0, tao1=0.9, tao2=0.45, population_size=20,
                                             differential_weight=0.5, crossover_probability=0.5)
    task = Task(problem=run_optimization(dimension=9,
                                         lower=0,  #, 2, 4, 1, 1, 1, 1],
                                         upper=1,  #, 64, 128, 3, 3, 3, 150],
                                         city=city, week2period=week2period, methodology=methodology, runs=runs),
                max_evals=1000, enable_logging=True)
    best = algo.run(task)
    print('%s -> %s' % (best[0], best[1]))
    print(algo.get_parameters())

#jde('Cordoba', 1, 'RP_CNN', 10)
jde(sys.argv[1], int(sys.argv[2]), sys.argv[3], int(sys.argv[4]))