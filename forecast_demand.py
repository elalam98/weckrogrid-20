"""
Created on Sat Mar 21 14:23:25 2020
Forecast demand
@author: Stamatis
"""

""" model id and choose save/load """
nn_model_id = 1
nn_model_save = False
nn_model_load = False

""" set initial seed """
initial_seed = 13
np.random.seed(initial_seed)
random.seed(initial_seed)

""" nn and forecasting settings """
settings['architecture'] = 'ffnn'
settings['input_dim'] = 24
settings['output_dim'] = 1
settings['days_for_testing'] = 20
settings['split_ratio'] = (365 - settings['days_for_testing']) / 365
settings['use_forecasted_demand'] = False
settings['reserve_margin_rate'] = 0.05

""" nn hyperparameters """
nn_hparams = dict()
nn_hparams['activation'] = 'relu'
nn_hparams['num_layers'] = 1
nn_hparams['units'] = 32
nn_hparams['dropout'] = 0.2
nn_hparams['optimizer'] = 'Adam'
nn_hparams['learning_rate'] = 0.001
nn_hparams['beta_1'] = 0.9
nn_hparams['beta_2'] = 0.999
nn_hparams['loss'] = 'mse'
nn_hparams['epochs'] = 100
nn_hparams['early_stopping'] = True
nn_hparams['patience'] = int(0.03*nn_hparams['epochs'])
nn_hparams['filters'] = 2
nn_hparams['kernel_size'] = 2
nn_hparams['pool_size'] = 2

class StopExecution(Exception):
    def _render_traceback_(self):
        pass

def preprocess_data(x):
    scaler = MinMaxScaler()
    x = x.reshape(-1, 1)
    scaled_data = scaler.fit_transform(x)
    
    return scaled_data, scaler

def stack_multiple_series(elist, preprocessed=False, scaler=dict()):
    multiple_series = tuple()
    for i, facility in enumerate(facilities):
        sliced_df = elist[i]
        if i == 0:
            for wp in settings['weather_predictors']:
                arr = np.array(sliced_df[wp])
                if preprocessed:
                    arr = arr.reshape(-1, 1)
                    arr = scaler[wp].transform(arr)
                else:
                    arr, scaler[wp] = preprocess_data(arr)
                multiple_series += (arr.reshape((len(arr), 1)),)
            
        dem = np.array(sliced_df['Demand'])
        if preprocessed:
            dem = dem.reshape(-1, 1)
            dem = scaler['Demand_' + facility].transform(dem)
        else:
            dem, scaler['Demand_' + facility] = preprocess_data(dem)
        multiple_series += (dem.reshape((len(dem), 1)),)
    
    return np.hstack(multiple_series), scaler

def split_train_test(x, ratio):
    x_train = x[:int(ratio*len(x))]
    x_test = x[int(ratio*len(x)):]
    
    return x_train, x_test
    
def split_sequence(sequence, input_dim, output_dim, first_demand_column=0):
    """ split a univariate sequence into samples """
    x, y = list(), list()
    for i in range(input_dim,len(sequence),1):
        """ gather input and output parts of the pattern """
        seq_x, seq_y = sequence[i-input_dim:i], sequence[i:i+output_dim,first_demand_column:]
        if len(seq_y) < output_dim:
            break
        x.append(seq_x)
        y.append(seq_y)
        
    return np.array(x), np.array(y)

def model_create_ffnn(input_dim, output_dim, n_features, n_houses, x_train, y_train, x_test, y_test, early=None):
    """ model creation and training """
    model = Sequential()
    for _ in range(nn_hparams['num_layers']):
        model.add(Dense(nn_hparams['units'], activation=nn_hparams['activation'], input_shape=(input_dim,n_features)))
        model.add(Dropout(nn_hparams['dropout']))
    model.add(Flatten())
    model.add(Dense(y_train.shape[1]*y_train.shape[2]))
    custom_optimizer = getattr(optimizers, nn_hparams['optimizer'])(lr=nn_hparams['learning_rate'], beta_1=nn_hparams['beta_1'], beta_2=nn_hparams['beta_2'])
    model.compile(optimizer=custom_optimizer, loss=nn_hparams['loss'])
    y_train = y_train.reshape((y_train.shape[0], y_train.shape[1]*y_train.shape[2]))
    y_test = y_test.reshape((y_test.shape[0], y_test.shape[1]*y_test.shape[2]))
    if early:
        model.fit(x_train, y_train, validation_data=(x_test,y_test), epochs=nn_hparams['epochs'], verbose=1, callbacks=[early])
    else:
        model.fit(x_train, y_train, validation_data=(x_test,y_test), epochs=nn_hparams['epochs'], verbose=1)
    model_loss = model.evaluate(x_train, y_train, verbose=0)
    
    return model, model_loss

def model_create_lstm(input_dim, output_dim, n_features, n_houses, x_train, y_train, x_test, y_test, early=None):
    """ model creation and training """
    model = Sequential()
    for _ in range(nn_hparams['num_layers']):
        model.add(LSTM(nn_hparams['units'], activation=nn_hparams['activation'], input_shape=(input_dim,n_features), return_sequences=True))
        model.add(Dropout(nn_hparams['dropout']))
    model.add(Flatten())
    model.add(Dense(y_train.shape[1]*y_train.shape[2]))
    custom_optimizer = getattr(optimizers, nn_hparams['optimizer'])(lr=nn_hparams['learning_rate'], beta_1=nn_hparams['beta_1'], beta_2=nn_hparams['beta_2'])
    model.compile(optimizer=custom_optimizer, loss=nn_hparams['loss'])
    y_train = y_train.reshape((y_train.shape[0], y_train.shape[1]*y_train.shape[2]))
    y_test = y_test.reshape((y_test.shape[0], y_test.shape[1]*y_test.shape[2]))
    if early:
        model.fit(x_train, y_train, validation_data=(x_test,y_test), epochs=nn_hparams['epochs'], verbose=1, callbacks=[early])
    else:
        model.fit(x_train, y_train, validation_data=(x_test,y_test), epochs=nn_hparams['epochs'], verbose=1)
    model_loss = model.evaluate(x_train, y_train, verbose=0)
    
    return model, model_loss

def model_create_gru(input_dim, output_dim, n_features, n_houses, x_train, y_train, x_test, y_test, early=None):
    """ model creation and training """
    model = Sequential()
    for _ in range(nn_hparams['num_layers']):
        model.add(GRU(nn_hparams['units'], activation=nn_hparams['activation'], input_shape=(input_dim,n_features), return_sequences=True))
        model.add(Dropout(nn_hparams['dropout']))
    model.add(Flatten())
    model.add(Dense(y_train.shape[1]*y_train.shape[2]))
    custom_optimizer = getattr(optimizers, nn_hparams['optimizer'])(lr=nn_hparams['learning_rate'], beta_1=nn_hparams['beta_1'], beta_2=nn_hparams['beta_2'])
    model.compile(optimizer=custom_optimizer, loss=nn_hparams['loss'])
    y_train = y_train.reshape((y_train.shape[0], y_train.shape[1]*y_train.shape[2]))
    y_test = y_test.reshape((y_test.shape[0], y_test.shape[1]*y_test.shape[2]))
    if early:
        model.fit(x_train, y_train, validation_data=(x_test,y_test), epochs=nn_hparams['epochs'], verbose=1, callbacks=[early])
    else:
        model.fit(x_train, y_train, validation_data=(x_test,y_test), epochs=nn_hparams['epochs'], verbose=1)
    model_loss = model.evaluate(x_train, y_train, verbose=0)
    
    return model, model_loss

def model_create_cnn(input_dim, output_dim, n_features, n_houses, x_train, y_train, x_test, y_test, early=None):
    """ model creation and training """
    model = Sequential()
    for _ in range(nn_hparams['num_layers']):
        model.add(Conv1D(filters=nn_hparams['filters'], kernel_size=nn_hparams['kernel_size'], activation=nn_hparams['activation'], input_shape=(input_dim,n_features)))
        model.add(MaxPooling1D(pool_size=nn_hparams['pool_size']))
    model.add(Flatten())
    model.add(Dropout(nn_hparams['dropout']))
    model.add(Dense(nn_hparams['units'], activation=nn_hparams['activation']))
    model.add(Dropout(nn_hparams['dropout']))
    model.add(Dense(nn_hparams['units'], activation=nn_hparams['activation']))
    model.add(Dropout(nn_hparams['dropout']))
    model.add(Dense(y_train.shape[1]*y_train.shape[2]))
    custom_optimizer = getattr(optimizers, nn_hparams['optimizer'])(lr=nn_hparams['learning_rate'], beta_1=nn_hparams['beta_1'], beta_2=nn_hparams['beta_2'])
    model.compile(optimizer=custom_optimizer, loss=nn_hparams['loss'])
    y_train = y_train.reshape((y_train.shape[0], y_train.shape[1]*y_train.shape[2]))
    y_test = y_test.reshape((y_test.shape[0], y_test.shape[1]*y_test.shape[2]))
    if early:
        model.fit(x_train, y_train, validation_data=(x_test,y_test), epochs=nn_hparams['epochs'], verbose=1, callbacks=[early])
    else:
        model.fit(x_train, y_train, validation_data=(x_test,y_test), epochs=nn_hparams['epochs'], verbose=1)
    model_loss = model.evaluate(x_train, y_train, verbose=0)
    
    return model, model_loss

def model_predict(model, x_input):
    """ use the created model to get predictions for any unprocessed input """
    stacked, _ = stack_multiple_series(x_input, preprocessed=True, scaler=scales)
    stacked = stacked.reshape(-1,stacked.shape[0],stacked.shape[1])
    res = model.predict(stacked)
    res = res.reshape(-1, 1, 1)
    ans = []
    for i, facility in enumerate(facilities):
        res[i] = scales['Demand_' + facility].inverse_transform(res[i])
        ans.append(res[i][0][0])
    return ans
    
def performance_metrics(true, predicted, max_val, min_val, avg_val, choice):
    true = true.flatten()
    predicted = predicted.flatten()
    mse = mean_squared_error(true, predicted)
    rmse = sqrt(mse)
    if choice == 'divide_average':
      nrmse = rmse / avg_val
    elif choice == 'divide_range':
      nrmse = rmse / (max_val-min_val)
    errors = sorted(abs(true-predicted), reverse=True)
    errors = errors[:int(0.1*len(errors))]
    maxerr = sum(errors) / len(errors)
    if choice == 'divide_average':
      maxerr /= avg_val
    elif choice == 'divide_range':
      maxerr /= (max_val-min_val)
    
    return mse, rmse, nrmse, maxerr

def compute_performance_metrics():
    metrics, facility_profile = dict(), dict()
    metrics['mse_train'], metrics['rmse_train'], metrics['nrmse_train'], metrics['maxerr_train'] = 0, 0, 0, 0
    metrics['mse_test'], metrics['rmse_test'], metrics['nrmse_test'], metrics['maxerr_test'] = 0, 0, 0, 0
    for i, facility in enumerate(facilities):
        max_demand, min_demand, avg_demand = np.amax(np.concatenate((y_train[:,:,i], y_test[:,:,i]))), np.amin(np.concatenate((y_train[:,:,i], y_test[:,:,i]))), np.mean(np.concatenate((y_train[:,:,i], y_test[:,:,i])))
        facility_profile[facility] = dict()
        facility_profile[facility]['max_demand'] = max_demand
        facility_profile[facility]['min_demand'] = min_demand
        facility_profile[facility]['avg_demand'] = avg_demand
        metrics['mse_train_' + facility], metrics['rmse_train_' + facility], metrics['nrmse_train_' + facility], metrics['maxerr_train_' + facility] = performance_metrics(y_train[:,:,i], pred_train[:,:,i], max_demand, min_demand, avg_demand, 'divide_average')
        metrics['mse_test_' + facility], metrics['rmse_test_' + facility], metrics['nrmse_test_' + facility], metrics['maxerr_test_' + facility] = performance_metrics(y_test[:,:,i], pred_test[:,:,i], max_demand, min_demand, avg_demand, 'divide_average')
        metrics['mse_train'] += metrics['mse_train_' + facility]
        metrics['rmse_train'] += metrics['rmse_train_' + facility]
        metrics['nrmse_train'] += metrics['nrmse_train_' + facility]
        metrics['maxerr_train'] += metrics['maxerr_train_' + facility]
        metrics['mse_test'] += metrics['mse_test_' + facility]
        metrics['rmse_test'] += metrics['rmse_test_' + facility]
        metrics['nrmse_test'] += metrics['nrmse_test_' + facility]
        metrics['maxerr_test'] += metrics['maxerr_test_' + facility]  
    metrics['mse_train'] /= num_facilities
    metrics['rmse_train'] /= num_facilities
    metrics['nrmse_train'] /= num_facilities
    metrics['maxerr_train'] /= num_facilities
    metrics['mse_test'] /= num_facilities
    metrics['rmse_test'] /= num_facilities
    metrics['nrmse_test'] /= num_facilities
    metrics['maxerr_test'] /= num_facilities
    
    return metrics, facility_profile

def print_performance_metrics():
    print('\n---------- Facilities Profiles ----------')
    for facility in facilities:
        print('Maximum demand for the ', facility, ' facility is: ', profiles[facility]['max_demand'])
        print('Minimum demand for the ', facility, ' facility is: ', profiles[facility]['min_demand'])
        print('Average demand for the ', facility, ' facility is: ', profiles[facility]['avg_demand'])
    print('\n---------- Total Performance Metrics ----------')
    print('Total MSE (train): ', metrics['mse_train'])
    print('Total MSE (test): ', metrics['mse_test'])
    print('Total RMSE (train): ', metrics['rmse_train'])
    print('Total RMSE (test): ', metrics['rmse_test'])
    print('Total NRMSE (train): ', metrics['nrmse_train'])
    print('Total NRMSE (test): ', metrics['nrmse_test'])
    print('Total maxERR (train): ', metrics['maxerr_train'])
    print('Total maxERR (test): ', metrics['maxerr_test'])

""" stop execution if needed """
if nn_model_load:
    nn_model = load_model(file_path + '/nn_model_v' + str(nn_model_id) + '.h5')
    scales = pickle.load(open(file_path + '/scales.pkl', 'rb'))
    raise StopExecution
if not settings['use_forecasted_demand']:
    raise StopExecution
    
""" define existing function list for architectures and locations """
architecture_list = {'ffnn': model_create_ffnn, 'lstm': model_create_lstm, 'gru': model_create_gru, 'cnn': model_create_cnn}

""" stack multiple series """
serialized, scales = stack_multiple_series(single_df)

""" split to train and test set and then split sequences  """
train_data, test_data = split_train_test(serialized, settings['split_ratio'])
x_train, y_train = split_sequence(train_data, settings['input_dim'], settings['output_dim'], weather.shape[1]-1)
x_test, y_test = split_sequence(test_data, settings['input_dim'], settings['output_dim'], weather.shape[1]-1)
    
""" create and fit the model """
n_features, n_facilities = x_train.shape[2], y_train.shape[2]
if nn_hparams['early_stopping']:
    es = EarlyStopping(monitor='val_loss', mode='min', patience=nn_hparams['patience'], verbose=1)
    nn_model, nn_model_loss = architecture_list[settings['architecture']](settings['input_dim'], settings['output_dim'], n_features, n_facilities, x_train, y_train, x_test, y_test, es)
else:
    nn_model, nn_model_loss = architecture_list[settings['architecture']](settings['input_dim'], settings['output_dim'], n_features, n_facilities, x_train, y_train, x_test, y_test)

""" make predictions for train and test set and inverse transform """
pred_train = nn_model.predict(x_train)
pred_test = nn_model.predict(x_test)
pred_train, pred_test = pred_train.reshape((y_train.shape[0], y_train.shape[1], y_train.shape[2])), pred_test.reshape((y_test.shape[0], y_test.shape[1], y_test.shape[2]))
for i, facility in enumerate(facilities):
    pred_train[:,:,i] = scales['Demand_' + facility].inverse_transform(pred_train[:,:,i])
    pred_test[:,:,i] = scales['Demand_' + facility].inverse_transform(pred_test[:,:,i])
    y_train[:,:,i] = scales['Demand_' + facility].inverse_transform(y_train[:,:,i])
    y_test[:,:,i] = scales['Demand_' + facility].inverse_transform(y_test[:,:,i])
    
""" compute performance metrics for train and test set """
metrics, profiles = compute_performance_metrics()

""" print performance metrics  for train and test """
print_performance_metrics()

""" save results if needed """
if nn_model_save:
    nn_model.save(file_path + '/nn_model_v' + str(nn_model_id) + '.h5') # save model
    pickle.dump(scales, open(file_path + '/scales.pkl', 'wb')) # save scalers
    nn_parameters = dict()
    nn_parameters['settings'], nn_parameters['nn_hparams'], nn_parameters['metrics'] = settings, nn_hparams, metrics
    with open(file_path + '/nn_parameters_v' + str(nn_model_id) + '.txt', 'w') as outfile: # save parameters
        json.dump(nn_parameters, outfile)