import numpy as np
import Methods
import Data_read
from sklearn.metrics import mean_squared_error


def run_exhaustive_search(x, **kwargs):
    feats = np.argwhere(x).ravel()
    func = getattr(kwargs['mt'], kwargs['methodology'])
    pred, param = func(feats=feats)
    # pred, param = func(feats=feats, f1=x[9], f2=x[10], f3=x[11], k1=x[12], k2=x[13], k3=x[14], e=x[15])

    error = mean_squared_error(y_orig[52:], pred.mean(axis=1))
    pred.attrs = param
    pred.attrs['perc'] = perc
    pred.to_pickle('results/' + kwargs["methodology"] + '/FS/' + str(error) + '_' + kwargs["city"] +
                   '_' + str(kwargs['week2period']) + '.pkl')
    return error


def manage_exhaustive_search(city, week2period, methodology, mt):
    best_res = 1e6
    for i in range(1, 512):
        comb = f'{i:b}'
        comb = comb.zfill(9)
        x = []
        for j in comb:
            x.append(int(j))
        test_res = run_exhaustive_search(x, city=city, week2period=week2period, methodology=methodology, mt=mt)
        if test_res < best_res:
            best_res = test_res
            best = comb
            print(best)


# Input arguments or manual definition:

#city = sys.argv[1]
#week2period = int(sys.argv[2])
#methodology = sys.argv[3]

city = 'Paris'
week2period = 1
methodology = 'LR'
X, X_rp, X_rp_bin, X_cnn, y, y_cnn, y_orig, scaler, y_max, y_min, y_cnn_max, y_cnn_min, perc = Data_read.init(
    city=city, week2period=week2period, perc=20)
mt = Methods.Methods(X, X_rp, X_rp_bin, X_cnn, y, y_cnn, y_orig, scaler, y_max, y_min, y_cnn_max, y_cnn_min, perc)
manage_exhaustive_search(city=city, week2period=week2period, methodology=methodology, mt=mt)
