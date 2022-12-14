import numpy as np
from keras.layers import Dense, Flatten
from keras.losses import mean_squared_error
from keras.optimizers import Adam
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Conv2DTranspose, Conv3D
from keras.backend import clear_session

from sklearn import linear_model
from sklearn.preprocessing import PolynomialFeatures
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, AdaBoostRegressor


class Methods:
    def __init__(self, X, X_rp, X_cnn, y, y_cnn, y_max, y_min, y_cnn_max, y_cnn_min):
        self.X = X

        self.X_cnn = X_cnn
        self.y_cnn = y_cnn
        self.X_pred_cnn = np.copy(self.X_cnn[52:])

        self.X_rp = X_rp
        self.X_pred = np.copy(self.X_rp[52:])

        self.y = y

        self.y_max = y_max
        self.y_min = y_min
        self.y_cnn_max = y_cnn_max
        self.y_cnn_min = y_cnn_min

    def CNN(self, *args, **kwargs):
        feats = kwargs['feats']
        k1 = kwargs['k1']
        k2 = kwargs['k2']
        k3 = kwargs['k3']

        if (k1 + k2 + k3) % 2 == 0:
            vec = np.array((k1, k2, k3))
            vec[vec.argmin()] = vec[vec.argmin()] + 1
            k1, k2, k3 = vec[0], vec[1], vec[2]
        dim = int((k1 + k2 + k3 - 1) / 2) - 1

        clear_session()
        model = Sequential()
        model.add(Conv3D(filters=kwargs['f1'], kernel_size=(3, k1, k1), activation='relu', padding='valid',
                         input_shape=(8, 33, 33, len(feats))))
        model.add(Conv3D(filters=kwargs['f2'], kernel_size=(3, k2, k2), activation='relu', padding='valid'))
        model.add(Conv3D(filters=kwargs['f3'], kernel_size=(4, k3, k3), activation='relu', padding='valid'))
        # model.add(Flatten())
        # model.add(Dense(1))

        model.compile(loss=mean_squared_error,
                      optimizer=Adam(),
                      metrics=['mse'])
        model.fit(
            self.X_cnn[:52, :, :, :, feats], self.y_cnn[:52, :, dim:-dim, dim:-dim, :],
            batch_size=12,
            epochs=kwargs['e'],
            verbose=0,
            # validation_data=(self.X_rp[52:], self.y[52:]),
            shuffle=True,

            # callbacks=[PlotLossesKeras()]
        )

        pred = model.predict(self.X_pred_cnn[:, :, :, :, feats], verbose=0)
        pred = pred[:, :, int(np.ceil(pred.shape[2] / 2)), int(np.ceil(pred.shape[3] / 2)), :].ravel()
        pred = pred * (self.y_cnn_max - self.y_cnn_min) + self.y_cnn_min

        return pred, kwargs

    def RP_CNN(self, *args, **kwargs):
        feats = kwargs['feats']
        k1 = kwargs['k1']
        k2 = kwargs['k2']
        k3 = kwargs['k3']

        clear_session()
        model = Sequential()
        model.add(Conv2D(filters=kwargs['f1'], kernel_size=(k1, k1), activation='relu', padding='valid',
                         input_shape=(8, 8, len(feats))))
        model.add(Conv2D(filters=kwargs['f2'], kernel_size=(k2, k2), activation='relu', padding='valid'))
        model.add(Conv2D(filters=kwargs['f3'], kernel_size=(k3, k3), activation='relu', padding='valid'))
        model.add(Flatten())
        model.add(Dense(1))

        model.compile(loss=mean_squared_error,
                      optimizer=Adam(),
                      metrics=['mse'])
        model.fit(
            self.X_rp[:52, :, :, feats], self.y[:52],
            batch_size=12,
            epochs=kwargs['e'],
            verbose=0,
            # validation_data=(self.X_rp[52:], self.y[52:]),
            shuffle=True,

            # callbacks=[PlotLossesKeras()]
        )

        pred = model.predict(self.X_pred[:, :, :, feats], verbose=0)
        pred = pred * (self.y_max - self.y_min) + self.y_min

        return pred, kwargs

    def LR(self, *args, **kwargs):
        feats = kwargs['feats']
        # LINEAR REGRESSION
        regr = linear_model.LinearRegression()
        regr.fit(self.X[:52, :, feats].reshape(52, -1), self.y[:52])
        pred = regr.predict(self.X[52:, :, feats].reshape(20, -1))
        pred = pred * (self.y_max - self.y_min) + self.y_min

        return pred, kwargs

    def Lasso(self, *args, **kwargs):
        feats = kwargs['feats']
        # LASSO REGRESSION
        regr = linear_model.Lasso(alpha=0.0005, fit_intercept=True)
        regr.fit(self.X[:52, :, feats].reshape(52, -1), self.y[:52])
        pred = regr.predict(self.X[52:, :, feats].reshape(20, -1))
        pred = pred * (self.y_max - self.y_min) + self.y_min

        return pred, kwargs

    def Poly(self, *args, **kwargs):
        feats = kwargs['feats']
        # POLYNOMIAL REGRESSION
        poly_reg = PolynomialFeatures(degree=4)
        X_train_poly = poly_reg.fit_transform(self.X[:52, :, feats].reshape(52, -1))
        X_test_poly = poly_reg.transform(self.X[52:, :, feats].reshape(20, -1))

        regr = linear_model.LinearRegression()
        regr.fit(X_train_poly, self.y[:52])
        pred = regr.predict(X_test_poly)
        pred = pred * (self.y_max - self.y_min) + self.y_min

        return pred, kwargs

    def DT(self, *args, **kwargs):
        feats = kwargs['feats']
        # DECISION TREES
        regr = DecisionTreeRegressor(max_depth=10, random_state=kwargs['i'])
        regr.fit(self.X[:52, :, feats].reshape(52, -1), self.y[:52])
        pred = regr.predict(self.X[52:, :, feats].reshape(20, -1))
        pred = pred * (self.y_max - self.y_min) + self.y_min

        return pred, kwargs

    def RF(self, *args, **kwargs):
        feats = kwargs['feats']
        # RANDOM FOREST
        regr = RandomForestRegressor(max_depth=10, random_state=kwargs['i'])
        regr.fit(self.X[:52, :, feats].reshape(52, -1), self.y[:52])
        pred = regr.predict(self.X[52:, :, feats].reshape(20, -1))
        pred = pred * (self.y_max - self.y_min) + self.y_min

        return pred, kwargs

    def AdaBoost(self, *args, **kwargs):
        feats = kwargs['feats']
        # AdaBoost
        regr = AdaBoostRegressor(n_estimators=100, random_state=kwargs['i'])
        regr.fit(self.X[:52, :, feats].reshape(52, -1), self.y[:52])
        pred = regr.predict(self.X[52:, :, feats].reshape(20, -1))
        pred = pred * (self.y_max - self.y_min) + self.y_min

        return pred, kwargs
