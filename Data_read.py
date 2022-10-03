import numpy as np
import xarray as xr
from pyts.image import RecurrencePlot


def init(city, week2period, perc=50):
    if city == 'Paris':  # Training - PARIS
        center_lat = 48.85
        center_long = 2.33
        vmin_set = 289
        vmax_set = 295
    elif city == 'Cordoba':  # Training - CORDOBA
        center_lat = 37.85
        center_long = -4.67
        vmin_set = 295
        vmax_set = 301
    else:
        print('Input the proper city.')

    feats = 9
    delta = 8.25
    lat_out = np.arange(center_lat - delta / 2, center_lat + delta / 2, .25)
    long_out = np.arange(center_long - delta / 2, center_long + delta / 2, .25)

    X, X_rp, y, y_cnn, y_orig, scaler = read(center_lat=center_lat, center_long=center_long, week2period=week2period,
                                             feats=feats, perc=perc, city=city)
    X_cnn = np.copy(X)
    X = X.max(axis=2).max(axis=2)

    y_min = y.min()
    y_max = y.max()
    y = (y - y_min) / (y_max - y_min)  # EXTEND the definition area

    y_cnn_min = y_cnn.min()
    y_cnn_max = y_cnn.max()
    y_cnn = (y_cnn - y_cnn_min) / (y_cnn_max - y_cnn_min)  # EXTEND the definition area
    y_cnn = np.moveaxis(y_cnn[np.newaxis, np.newaxis, ...], 2, 0)
    y_cnn = np.moveaxis(y_cnn, 1, -1)

    return X, X_rp, X_cnn, y, y_cnn, y_orig, scaler, y_max, y_min, y_cnn_max, y_cnn_min


def read(center_lat, center_long, week2period, feats, perc, city):
    ds_orig = xr.open_dataset('/home/comp/Dataset_2_week_extended_corrected.nc')
    ds_month = ds_orig.sel(time=ds_orig.time.dt.month.isin(range(4, 8)))
    delta = 8.25

    lat_out = np.arange(center_lat - delta / 2, center_lat + delta / 2, .25)
    long_out = np.arange(center_long - delta / 2, center_long + delta / 2, .25)

    y_temp = ds_orig.sel(time=ds_orig.time.dt.month == 8, latitude=center_lat, longitude=center_long,
                         method='nearest')
    y = np.array(y_temp.t2m.sel(time=y_temp.time.dt.day == week2period))
    y_orig = np.copy(y)

    y_cnn_temp = ds_orig.t2m.sel(time=ds_orig.time.dt.month == 8, latitude=lat_out, longitude=long_out,
                             method='nearest')
    y_cnn = y_cnn_temp.sel(time=y_cnn_temp.time.dt.day == week2period)

    X = np.zeros((72, 8, 33, 33, feats))
    X_rp = np.zeros((72, 8, 8, feats))
    easy_scale = 2
    rp = RecurrencePlot(threshold='point', percentage=perc) # threshold = point, distance; percentage=25
    # rp = RecurrencePlot()
    scaler = np.zeros((72, feats, 2))

    for t in range(1950, 2022):
        t2m = ds_month.t2m.sel(time=ds_month.time.dt.year == t,
                               latitude=lat_out, longitude=long_out,
                               method='nearest')  # filter LOCATION & YEAR
        u10 = ds_month.u10.sel(time=ds_month.time.dt.year == t,
                               latitude=np.arange(26, 26 + delta, .25),
                               longitude=np.arange(6, 6 + delta, .25),
                               method='nearest')  # filter LOCATION & YEAR
        v10 = ds_month.v10.sel(time=ds_month.time.dt.year == t,
                               latitude=np.arange(24, 24 + delta, .25),
                               longitude=np.arange(20, 20 + delta, .25),
                               method='nearest')  # filter LOCATION & YEAR
        u100 = ds_month.u100.sel(time=ds_month.time.dt.year == t,
                                 latitude=np.arange(25, 25 + delta, .25),
                                 longitude=np.arange(6, 6 + delta, .25),
                                 method='nearest')  # filter LOCATION & YEAR
        v100 = ds_month.v100.sel(time=ds_month.time.dt.year == t,
                                 latitude=np.arange(26, 26 + delta, .25),
                                 longitude=np.arange(20, 20 + delta, .25),
                                 method='nearest')  # filter LOCATION & YEAR
        msl = ds_month.msl.sel(time=ds_month.time.dt.year == t,
                               latitude=np.arange(55, 55 + delta, .25),
                               longitude=np.arange(-30, -30 + delta, .25),
                               method='nearest')  # filter LOCATION & YEAR
        sst = ds_month.sst.sel(time=ds_month.time.dt.year == t,
                               latitude=np.arange(41, 41 + delta, .25),
                               longitude=np.arange(-19, -19 + delta, .25),
                               method='nearest')  # filter LOCATION & YEAR
        z = ds_month.z.sel(time=ds_month.time.dt.year == t,
                           latitude=np.arange(31, 31 + delta, .25),
                           longitude=np.arange(1, 1 + delta, .25),
                           method='nearest')  # filter LOCATION & YEAR
        if city == 'Paris':
            swvl1 = ds_month.swvl1.sel(time=ds_month.time.dt.year == t,
                                       latitude=np.arange(42.5, 42.5 + delta, .25),
                                       longitude=long_out,
                                       method='nearest')  # filter LOCATION & YEAR
        elif city == 'Cordoba':
            swvl1 = ds_month.swvl1.sel(time=ds_month.time.dt.year == t,
                                       latitude=np.arange(36, 36 + delta, .25),
                                       longitude=long_out,
                                       method='nearest')  # filter LOCATION & YEAR

        X[t - 1950, :, :, :, 0] = np.array(t2m)
        X[t - 1950, :, :, :, 1] = np.array(u10)
        X[t - 1950, :, :, :, 2] = np.array(u100)
        X[t - 1950, :, :, :, 3] = np.array(v10)
        X[t - 1950, :, :, :, 4] = np.array(v100)
        X[t - 1950, :, :, :, 5] = np.array(msl)
        X[t - 1950, :, :, :, 6] = np.array(sst)
        X[t - 1950, :, :, :, 7] = np.array(z)
        X[t - 1950, :, :, :, 8] = np.array(swvl1)
        y_cnn = np.array(y_cnn)

        if easy_scale == 1:
            for i in range(feats):
                scaler[t, i, 0] = X[:, :, :, :, i].max()  # maximum
                scaler[t, i, 1] = X[:, :, :, :, i].min()  # minimum
                X[t, :, :, :, i] = (X[t, :, :, :, i] - scaler[t, i, 1]) / (scaler[t, i, 0] - scaler[t, i, 1])
            y[t] = (y[t] - scaler[t, 0, 1]) / (scaler[t, 0, 0] - scaler[t, 0, 1])
            y_cnn[t] = (y_cnn[t] - scaler[t, 0, 1]) / (scaler[t, 0, 0] - scaler[t, 0, 1])
        else:
            for i in range(feats):
                scaler[t - 1950, i, 0] = X[t - 1950, :, :, :, i].max()  # maximum
                scaler[t - 1950, i, 1] = X[t - 1950, :, :, :, i].min()  # minimum
                X[t - 1950, :, :, :, i] = (X[t - 1950, :, :, :, i] - scaler[t - 1950, i, 1]) / (scaler[t - 1950, i, 0] - scaler[t - 1950, i, 1])
            y[t - 1950] = (y[t - 1950] - scaler[t - 1950, 0, 1]) / (scaler[t - 1950, 0, 0] - scaler[t - 1950, 0, 1])
            y_cnn[t - 1950] = (y_cnn[t - 1950] - scaler[t - 1950, 0, 1]) / (scaler[t - 1950, 0, 0] - scaler[t - 1950, 0, 1])

        for i in range(feats):
            X_rp[t - 1950, :, :, i] = rp.fit_transform(np.array(X[t - 1950, :, :, :, i]).max(axis=1).max(axis=1).reshape(1, -1))

    return X, X_rp, y, y_cnn, y_orig, scaler


def get_data(ds_orig, center_long, center_lat, long_out, lat_out, delta, feats, week2period, init_size):
	'''
	Not used anymore.
	'''
    # suppl = Supplements()
    ds_loc_in_t2m = ds_orig.t2m.sel(time=ds_orig.time.dt.year.isin(range(1950, 2002)),
                                    latitude=lat_out, longitude=long_out,
                                    method='nearest')  # filter LOCATION & YEAR
    ds_loc_in_u10 = ds_orig.u10.sel(time=ds_orig.time.dt.year.isin(range(1950, 2002)),
                                    latitude=np.arange(26, 26 + delta, .25), longitude=np.arange(6, 6 + delta, .25),
                                    method='nearest')  # filter LOCATION & YEAR
    ds_loc_in_v10 = ds_orig.v10.sel(time=ds_orig.time.dt.year.isin(range(1950, 2002)),
                                    latitude=np.arange(24, 24 + delta, .25), longitude=np.arange(20, 20 + delta, .25),
                                    method='nearest')  # filter LOCATION & YEAR
    ds_loc_in_u100 = ds_orig.u100.sel(time=ds_orig.time.dt.year.isin(range(1950, 2002)),
                                      latitude=np.arange(25, 25 + delta, .25), longitude=np.arange(6, 6 + delta, .25),
                                      method='nearest')  # filter LOCATION & YEAR
    ds_loc_in_v100 = ds_orig.v100.sel(time=ds_orig.time.dt.year.isin(range(1950, 2002)),
                                      latitude=np.arange(26, 26 + delta, .25), longitude=np.arange(20, 20 + delta, .25),
                                      method='nearest')  # filter LOCATION & YEAR
    ds_loc_in_msl = ds_orig.msl.sel(time=ds_orig.time.dt.year.isin(range(1950, 2002)),
                                    latitude=np.arange(55, 55 + delta, .25), longitude=np.arange(-30, -30 + delta, .25),
                                    method='nearest')  # filter LOCATION & YEAR
    ds_loc_in_sst = ds_orig.sst.sel(time=ds_orig.time.dt.year.isin(range(1950, 2002)),
                                    latitude=np.arange(41, 41 + delta, .25), longitude=np.arange(-19, -19 + delta, .25),
                                    method='nearest')  # filter LOCATION & YEAR
    ds_loc_in_z = ds_orig.z.sel(time=ds_orig.time.dt.year.isin(range(1950, 2002)),
                                latitude=np.arange(31, 31 + delta, .25), longitude=np.arange(1, 1 + delta, .25),
                                method='nearest')  # filter LOCATION & YEAR
    ds_loc_in_swvl1 = ds_orig.swvl1.sel(time=ds_orig.time.dt.year.isin(range(1950, 2002)),
                                        latitude=np.arange(36, 36 + delta, .25), longitude=long_out,
                                        method='nearest')  # filter LOCATION & YEAR

    # suppl.exhibit(ds_loc_in_t2m)
    # suppl.exhibit(ds_loc_in_u10)
    # suppl.exhibit(ds_loc_in_v10)
    # suppl.exhibit(ds_loc_in_u100)
    # suppl.exhibit(ds_loc_in_v100)
    # suppl.exhibit(ds_loc_in_msl)
    # suppl.exhibit(ds_loc_in_sst)
    # suppl.exhibit(ds_loc_in_z)
    # suppl.exhibit(ds_loc_in_swvl1)
    # suppl.finish(ds_loc_in_t2m)

    ds_loc_in_t2m = np.array(ds_loc_in_t2m.sel(time=ds_loc_in_t2m.time.dt.month.isin(range(4, 8))))
    ds_loc_in_u10 = np.array(ds_loc_in_u10.sel(time=ds_loc_in_u10.time.dt.month.isin(range(4, 8))))
    ds_loc_in_u100 = np.array(ds_loc_in_u100.sel(time=ds_loc_in_u100.time.dt.month.isin(range(4, 8))))
    ds_loc_in_v10 = np.array(ds_loc_in_v10.sel(time=ds_loc_in_v10.time.dt.month.isin(range(4, 8))))
    ds_loc_in_v100 = np.array(ds_loc_in_v100.sel(time=ds_loc_in_v100.time.dt.month.isin(range(4, 8))))
    ds_loc_in_msl = np.array(ds_loc_in_msl.sel(time=ds_loc_in_msl.time.dt.month.isin(range(4, 8))))
    ds_loc_in_sst = np.array(ds_loc_in_sst.sel(time=ds_loc_in_sst.time.dt.month.isin(range(4, 8))))
    ds_loc_in_z = np.array(ds_loc_in_z.sel(time=ds_loc_in_z.time.dt.month.isin(range(4, 8))))
    ds_loc_in_swvl1 = np.array(ds_loc_in_swvl1.sel(time=ds_loc_in_swvl1.time.dt.month.isin(range(4, 8))))

    x_train = np.zeros((416, init_size, init_size, feats))
    x_train[:, :, :, 0] = ds_loc_in_t2m
    x_train[:, :, :, 1] = ds_loc_in_u10
    x_train[:, :, :, 2] = ds_loc_in_u100
    x_train[:, :, :, 3] = ds_loc_in_v10
    x_train[:, :, :, 4] = ds_loc_in_v100
    x_train[:, :, :, 5] = ds_loc_in_msl
    x_train[:, :, :, 6] = ds_loc_in_sst
    x_train[:, :, :, 7] = ds_loc_in_z
    x_train[:, :, :, 8] = ds_loc_in_swvl1

    ds_loc_out = ds_orig.sel(time=ds_orig.time.dt.year.isin(range(1950, 2002)),
                             latitude=lat_out, longitude=long_out, method='nearest')  # filter LOCATION & YEAR
    y_train_intermediate = ds_loc_out.sel(
        time=ds_loc_out.time.dt.month == 8).t2m  # ¡¡ahora primera dos semanas del Julio!! cuidado
    y_train = np.array(y_train_intermediate.sel(time=y_train_intermediate.time.dt.day == week2period))
    y_train_dense = ds_loc_out.sel(time=ds_loc_out.time.dt.month == 8, longitude=center_long, latitude=center_lat,
                                   method='nearest').t2m
    y_train_dense = np.array(y_train_dense.sel(time=y_train_dense.time.dt.day == week2period))

    # Validation and Testing - PARIS
    # ds_loc_in = ds_orig.sel(time=ds_orig.time.dt.year.isin(range(2002, 2022)),
    #                        latitude=lat_out, longitude=long_out, method='nearest')  # filter LOCATION & YEAR
    ds_loc_out = ds_orig.sel(time=ds_orig.time.dt.year.isin(range(2002, 2022)),
                             latitude=lat_out, longitude=long_out, method='nearest')  # filter LOCATION & YEAR

    y_val_intermediate = ds_loc_out.sel(time=ds_loc_out.time.dt.month == 8,
                                        latitude=lat_out, longitude=long_out, method='nearest')
    y_val = y_val_intermediate.sel(time=y_val_intermediate.time.dt.day == week2period).t2m.values

    # X TEST PART
    ds_loc_in_t2m = ds_orig.t2m.sel(time=ds_orig.time.dt.year.isin(range(2002, 2022)),
                                    latitude=lat_out, longitude=long_out,
                                    method='nearest')  # filter LOCATION & YEAR
    ds_loc_in_u10 = ds_orig.u10.sel(time=ds_orig.time.dt.year.isin(range(2002, 2022)),
                                    latitude=np.arange(26, 26 + delta, .25), longitude=np.arange(6, 6 + delta, .25),
                                    method='nearest')  # filter LOCATION & YEAR
    ds_loc_in_v10 = ds_orig.v10.sel(time=ds_orig.time.dt.year.isin(range(2002, 2022)),
                                    latitude=np.arange(24, 24 + delta, .25), longitude=np.arange(20, 20 + delta, .25),
                                    method='nearest')  # filter LOCATION & YEAR
    ds_loc_in_u100 = ds_orig.u100.sel(time=ds_orig.time.dt.year.isin(range(2002, 2022)),
                                      latitude=np.arange(25, 25 + delta, .25), longitude=np.arange(6, 6 + delta, .25),
                                      method='nearest')  # filter LOCATION & YEAR
    ds_loc_in_v100 = ds_orig.v100.sel(time=ds_orig.time.dt.year.isin(range(2002, 2022)),
                                      latitude=np.arange(26, 26 + delta, .25), longitude=np.arange(20, 20 + delta, .25),
                                      method='nearest')  # filter LOCATION & YEAR
    ds_loc_in_msl = ds_orig.msl.sel(time=ds_orig.time.dt.year.isin(range(2002, 2022)),
                                    latitude=np.arange(55, 55 + delta, .25), longitude=np.arange(-30, -30 + delta, .25),
                                    method='nearest')  # filter LOCATION & YEAR
    ds_loc_in_sst = ds_orig.sst.sel(time=ds_orig.time.dt.year.isin(range(2002, 2022)),
                                    latitude=np.arange(41, 41 + delta, .25), longitude=np.arange(-19, -19 + delta, .25),
                                    method='nearest')  # filter LOCATION & YEAR
    ds_loc_in_z = ds_orig.z.sel(time=ds_orig.time.dt.year.isin(range(2002, 2022)),
                                latitude=lat_out, longitude=long_out,
                                method='nearest')  # filter LOCATION & YEAR
    ds_loc_in_swvl1 = ds_orig.swvl1.sel(time=ds_orig.time.dt.year.isin(range(2002, 2022)),
                                        latitude=np.arange(36, 36 + delta, .25), longitude=long_out,
                                        method='nearest')  # filter LOCATION & YEAR

    ds_loc_out = ds_orig.sel(time=ds_orig.time.dt.year.isin(range(2002, 2022)),
                             latitude=lat_out, longitude=long_out, method='nearest')  # filter LOCATION & YEAR

    ds_loc_in_t2m = np.array(ds_loc_in_t2m.sel(time=ds_loc_in_t2m.time.dt.month.isin(range(4, 8))))
    ds_loc_in_u10 = np.array(ds_loc_in_u10.sel(time=ds_loc_in_u10.time.dt.month.isin(range(4, 8))))
    ds_loc_in_u100 = np.array(ds_loc_in_u100.sel(time=ds_loc_in_u100.time.dt.month.isin(range(4, 8))))
    ds_loc_in_v10 = np.array(ds_loc_in_v10.sel(time=ds_loc_in_v10.time.dt.month.isin(range(4, 8))))
    ds_loc_in_v100 = np.array(ds_loc_in_v100.sel(time=ds_loc_in_v100.time.dt.month.isin(range(4, 8))))
    ds_loc_in_msl = np.array(ds_loc_in_msl.sel(time=ds_loc_in_msl.time.dt.month.isin(range(4, 8))))
    ds_loc_in_sst = np.array(ds_loc_in_sst.sel(time=ds_loc_in_sst.time.dt.month.isin(range(4, 8))))
    ds_loc_in_z = np.array(ds_loc_in_z.sel(time=ds_loc_in_z.time.dt.month.isin(range(4, 8))))
    ds_loc_in_swvl1 = np.array(ds_loc_in_swvl1.sel(time=ds_loc_in_swvl1.time.dt.month.isin(range(4, 8))))

    x_test = np.zeros((160, init_size, init_size, feats))
    x_test[:, :, :, 0] = ds_loc_in_t2m
    x_test[:, :, :, 1] = ds_loc_in_u10
    x_test[:, :, :, 2] = ds_loc_in_u100
    x_test[:, :, :, 3] = ds_loc_in_v10
    x_test[:, :, :, 4] = ds_loc_in_v100
    x_test[:, :, :, 5] = ds_loc_in_msl
    x_test[:, :, :, 6] = ds_loc_in_sst
    x_test[:, :, :, 7] = ds_loc_in_z
    x_test[:, :, :, 8] = ds_loc_in_swvl1

    y_test_intermediate = ds_loc_out.sel(time=ds_loc_out.time.dt.month == 8,
                                         latitude=center_lat, longitude=center_long, method='nearest')
    y_test = y_test_intermediate.sel(time=y_test_intermediate.time.dt.day == week2period).t2m.values
    ds_orig.close()

    x_train = np.nan_to_num(x_train)
    x_test = np.nan_to_num(x_test)

    return x_train, y_train, x_test, y_test, y_val, ds_loc_out, y_train_dense