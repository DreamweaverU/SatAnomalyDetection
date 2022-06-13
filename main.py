import io
import os
import urllib.request
import zipfile
import datetime
import pdb
import holidays
import json
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from datetime import datetime
import pandas as pd
from data import load_signal
import plotly.express as px
from sklearn.preprocessing import MinMaxScaler
DATA_URL = 'https://s3-us-west-2.amazonaws.com/telemanom/data.zip'
from statistics import mean
from sklearn.metrics import confusion_matrix
import itertools
import matplotlib.pyplot as plt
if not os.path.exists('data'):
    response = urllib.request.urlopen(DATA_URL)
    bytes_io = io.BytesIO(response.read())

    with zipfile.ZipFile(bytes_io) as zf:
        zf.extractall()

train_signals = os.listdir('data/train')
test_signals = os.listdir('data/test')
def plot_confusion_matrix(cm, classes,
                        normalize=False,
                        title='Confusion matrix',
                        cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)
def build_df(data, start=0):
#    pdb.set_trace()
    index = np.array(range(start, start + len(data)))
    timestamp = index * 86400 + 1022819200

    return pd.DataFrame({'timestamp': timestamp.astype(int), 'value': data[:, 0], 'index': index.astype(int)})


def create_sliding_window(data, sequence_length, stride=1):
    X_list, y_list = [], []
    for i in range(len(data)):
      if (i + sequence_length) < len(data):
        X_list.append(data.iloc[i:i+sequence_length:stride, :].values)
        y_list.append(data.iloc[i+sequence_length, -1])
    return np.array(X_list), np.array(y_list)


def inverse_transform(y):
    return target_scaler.inverse_transform(y.reshape(-1, 1))
class BayesianLSTM(nn.Module):

    def __init__(self, n_features, output_length, batch_size):
        super(BayesianLSTM, self).__init__()

        self.batch_size = batch_size  # user-defined

        self.hidden_size_1 = 128  # number of encoder cells (from paper)
        self.hidden_size_2 = 32  # number of decoder cells (from paper)
        self.stacked_layers = 2  # number of (stacked) LSTM layers for each stage
        self.dropout_probability = 0.5  # arbitrary value (the paper suggests that performance is generally stable across all ranges)

        self.lstm1 = nn.LSTM(n_features,
                             self.hidden_size_1,
                             num_layers=self.stacked_layers,
                             batch_first=True)
        self.lstm2 = nn.LSTM(self.hidden_size_1,
                             self.hidden_size_2,
                             num_layers=self.stacked_layers,
                             batch_first=True)

        self.fc = nn.Linear(self.hidden_size_2, output_length)
        self.loss_fn = nn.MSELoss()

    def forward(self, x):
        batch_size, seq_len, _ = x.size()

        hidden = self.init_hidden1(batch_size)
        output, _ = self.lstm1(x, hidden)
        output = F.dropout(output, p=self.dropout_probability, training=True)
        state = self.init_hidden2(batch_size)
        output, state = self.lstm2(output, state)
        output = F.dropout(output, p=self.dropout_probability, training=True)
        output = output[:, -1, :]  # take the last decoder cell's outputs
        y_pred = self.fc(output)
        return y_pred

    def init_hidden1(self, batch_size):
        hidden_state = Variable(torch.zeros(self.stacked_layers, batch_size, self.hidden_size_1))
        cell_state = Variable(torch.zeros(self.stacked_layers, batch_size, self.hidden_size_1))
        return hidden_state, cell_state

    def init_hidden2(self, batch_size):
        hidden_state = Variable(torch.zeros(self.stacked_layers, batch_size, self.hidden_size_2))
        cell_state = Variable(torch.zeros(self.stacked_layers, batch_size, self.hidden_size_2))
        return hidden_state, cell_state

    def loss(self, pred, truth):
        return self.loss_fn(pred, truth)

    def predict(self, X):
        return self(torch.tensor(X, dtype=torch.float32)).view(-1).detach().numpy()

###Prepare Labels
CSV_URL = 'https://github.com/khundman/telemanom/raw/master/labeled_anomalies.csv'

# %%
os.makedirs('csv', exist_ok=True)

# %%
selected_columns = ['index','date', 'name', 'hour_of_day', 'value']



df_label = pd.read_csv(CSV_URL)

#name='F-7'
MSL=df_label[df_label.spacecraft=='MSL']['chan_id']
SMAP=df_label[df_label.spacecraft=='SMAP']['chan_id']

print(SMAP)
pdb.set_trace()
avg=[]
train_signals=SMAP
precision=[]
recall=[]
Accuracy=[]
F1=[]
im=0
training_truth_df=pd.DataFrame()
for name in SMAP:

        label_row=df_label[df_label.chan_id==name]

        labels= label_row['anomaly_sequences'][label_row['anomaly_sequences'].index]

        appended_data = []



        labels= eval(labels[im])

        for i in range(len(labels)):
            anom=labels[i]
            start=anom[0]
            end=anom[1]

            index = np.array(range(start, end))

            timestamp = index * 86400 + 1022819200

            anomalies=pd.DataFrame({'timestamp': timestamp.astype(int), 'value': 1, 'index':index})
            appended_data.append(anomalies)

        label_data = pd.concat(appended_data)

        label_data ['date'] = pd.to_datetime(label_data ['timestamp'], unit='s')
        label_data ['month'] = label_data ['date'].dt.month.astype(int)
        label_data ['name'] = name
        label_data ['day_of_week'] = label_data ['date'].dt.dayofweek.astype(int)
        label_data ['hour_of_day'] = label_data ['date'].dt.hour.astype(int)
        label_data  = label_data [selected_columns]
        #label_data .to_csv('csv/' + name + '.csv', index=False)


        signal=name
        train_np = np.load('data/train/' + signal + '.npy')
        test_np = np.load('data/test/' + signal + '.npy')

        data = build_df(np.concatenate([train_np, test_np]))
        data['date'] = pd.to_datetime(data['timestamp'], unit='s')
        data['month'] = data['date'].dt.month.astype(int)
        data['name'] = name
        data['index'] = data['index'].astype(int)
        data['day_of_week'] = data['date'].dt.dayofweek.astype(int)
        data['hour_of_day'] = data['date'].dt.hour.astype(int)
        data = data[selected_columns]

        train = build_df(train_np)
        train['date'] = pd.to_datetime(train['timestamp'], unit='s')
        train['month'] = train['date'].dt.month.astype(int)
        train['day_of_month'] = train['date'].dt.day.astype(int)
        train['name'] = name
        train['day_of_week'] = train['date'].dt.dayofweek.astype(int)
        train['hour_of_day'] = train['date'].dt.hour.astype(int)
        train['index'] = train['index'].astype(int)
        train = train[selected_columns]
        # train.to_csv('csv/' + name + '.csv', index=False)
        # train.to_csv('csv/' + name + '-train.csv', index=False)

        test = build_df(test_np, start=len(train))
        test['date'] = pd.to_datetime(test['timestamp'], unit='s')
        test['month'] = test['date'].dt.month.astype(int)
        test['name'] = name
        test['day_of_week'] = test['date'].dt.dayofweek.astype(int)
        test['hour_of_day'] = test['date'].dt.hour.astype(int)
        test['index'] = test['index'].astype(int)
        test = test[selected_columns]
        # test.to_csv('csv/' + name + '.csv', index=False)
        # test.to_csv('csv/' + name + '-train.csv', index=False)
        # test.to_csv('csv/' + name + '-test.csv', index=False)

        datetime_columns = ['index', 'date', 'name', 'hour_of_day']
        target_column = 'value'

        feature_columns = datetime_columns + ['value']

        resample_df = train[feature_columns]
        resample_df_test = test[feature_columns]
        print(resample_df)

        plot_length = 1000000
        plot_df = resample_df.copy(deep=True).iloc[:plot_length]
        plot_df['weekday'] = plot_df['date'].dt.day_name()

        n_train = len(train_np)
        n_test = len(test_np)

        features = ['index', 'hour_of_day', 'value']
        feature_array = resample_df[features].values
        feature_array_test = resample_df_test[features].values
        # Fit Scaler only on Training features
        feature_scaler = MinMaxScaler()
        feature_scaler.fit(feature_array[:n_train])
        # Fit Scaler only on Training target values
        feature_scaler.fit(feature_array_test[:n_test])

        target_scaler = MinMaxScaler()
        target_scaler.fit(feature_array[:n_train, -1].reshape(-1, 1))

        # Transfom on both Training and Test data
        scaled_array = pd.DataFrame(feature_scaler.transform(feature_array),
                                    columns=features)

        scaled_array_test = pd.DataFrame(feature_scaler.transform(feature_array_test),
                                         columns=features)

        sequence_length = 10

        X_train, y_train = create_sliding_window(scaled_array,
                                                 sequence_length)

        X_test, y_test = create_sliding_window(scaled_array_test,
                                               sequence_length)

        n_features = scaled_array.shape[-1]
        sequence_length = 10
        output_length = 1

        batch_size = 128
        n_epochs = 300
        learning_rate = 0.01

        bayesian_lstm = BayesianLSTM(n_features=n_features,
                                     output_length=output_length,
                                     batch_size=batch_size)

        criterion = torch.nn.MSELoss()
        optimizer = torch.optim.Adam(bayesian_lstm.parameters(), lr=learning_rate)
        bayesian_lstm.train()

        for e in range(1, n_epochs + 1):
            for b in range(0, len(X_train), batch_size):
                features = X_train[b:b + batch_size, :, :]
                target = y_train[b:b + batch_size]

                X_batch = torch.tensor(features, dtype=torch.float32)
                y_batch = torch.tensor(target, dtype=torch.float32)

                output = bayesian_lstm(X_batch)
                loss = criterion(output.view(-1), y_batch)

                loss.backward()
                optimizer.step()
                optimizer.zero_grad()

            if e % 10 == 0:
                print('epoch', e, 'loss: ', loss.item())
                offset = sequence_length

        training_df = pd.DataFrame()
        training_df['date'] = resample_df['date'].iloc[offset:n_train + offset:1]
        training_df['index'] = resample_df['index'].iloc[offset:n_train + offset:1]
        training_predictions = bayesian_lstm.predict(X_train)

        training_df['value'] = inverse_transform(training_predictions)
        training_df['source'] = 'Training Prediction'


        training_truth_df['date'] = training_df['date']
        training_truth_df['index'] = training_df['index']
        training_truth_df['value'] = resample_df['value'].iloc[
                                     offset:n_train + offset:1]
        training_truth_df['source'] = 'True Values'

        testing_df = pd.DataFrame()
        testing_df['date'] = resample_df_test['date'].iloc[offset:n_test + offset:1]
        testing_df['index'] = resample_df_test['index'].iloc[offset:n_test + offset:1]

        testing_predictions = bayesian_lstm.predict(X_test)

        testing_df['value'] = inverse_transform(testing_predictions)
        testing_df['source'] = 'Test Prediction'

        testing_truth_df = pd.DataFrame()
        testing_truth_df['date'] = testing_df['date']
        testing_truth_df['index'] = testing_df['index']

        testing_truth_df['value'] = resample_df_test['value'].iloc[offset:n_test + offset:1]
        testing_truth_df['source'] = 'True Values'


        evaluation = pd.concat([training_df,
                                testing_df,
                                training_truth_df,
                                testing_truth_df
                                ], axis=0)

        fig = px.line(evaluation,
                             x="index",
                              y="value",
                              color="source")
        fig.show()

        n_experiments = 2
        test_uncertainty_df = pd.DataFrame()
        test_uncertainty_df['date'] = testing_df['date']
        test_uncertainty_df['index'] = testing_df['index']

        for i in range(n_experiments):
            experiment_predictions = bayesian_lstm.predict(X_test)

            test_uncertainty_df['value_{}'.format(i)] = inverse_transform(experiment_predictions)

        log_energy_consumption_df = test_uncertainty_df.filter(like='value', axis=1)
        test_uncertainty_df['value_mean'] = log_energy_consumption_df.mean(axis=1)
        test_uncertainty_df['value_std'] = log_energy_consumption_df.std(axis=1)


        test_uncertainty1 = test_uncertainty_df['value_mean']

        #########################Adaptive Smoothing

        #   pdb.set_trace()
        Threshold = 10
        gamma = 0.1
        betta = 0.1
        test_uncertainty2 = []
        ###########
        for jj in range(len(test_uncertainty1) - 2):
            test_uncertaintyy = (1 / (1 + gamma + betta)) * (
                        test_uncertainty1.iloc[jj] + gamma * test_uncertainty1.iloc[jj + 1] + betta *
                        test_uncertainty1.iloc[jj + 2])
            test_uncertainty2.append(test_uncertaintyy)
        test_uncertainty2 = pd.DataFrame(test_uncertainty2)

        test_uncertainty2.columns = ["value_mean"]


        test_uncertainty_df["value_mean"] = test_uncertainty2["value_mean"]

        ########################################################


        test_uncertainty_df = test_uncertainty_df[['index', 'date', 'value_mean', 'value_std']]
        test_uncertainty_df['lower_bound'] = test_uncertainty_df['value_mean'] - 15 * test_uncertainty_df['value_std']
        test_uncertainty_df['upper_bound'] = test_uncertainty_df['value_mean'] + 15 * test_uncertainty_df['value_std']
        import plotly.graph_objects as go

        test_uncertainty_plot_df = test_uncertainty_df.copy(deep=True)
        # test_uncertainty_plot_df = test_uncertainty_plot_df.loc[test_uncertainty_plot_df['date'].between('2016-05-01', '2016-05-09')]
        truth_uncertainty_plot_df = testing_truth_df.copy(deep=True)
        # truth_uncertainty_plot_df = truth_uncertainty_plot_df.loc[testing_truth_df['date'].between('2016-05-01', '2016-05-09')]

        upper_trace = go.Scatter(
            x=test_uncertainty_plot_df['index'],
           y=test_uncertainty_plot_df['upper_bound'],
          mode='lines',
         fill=None,
         name='99% Upper Confidence Bound'
         )
        lower_trace = go.Scatter(
            x=test_uncertainty_plot_df['index'],
           y=test_uncertainty_plot_df['lower_bound'],
          mode='lines',
          fill='tonexty',
         name= '99% Lower Confidence Bound',
         fillcolor='rgba(255, 211, 0, 0.1)',
         )
        real_trace = go.Scatter(
            x=truth_uncertainty_plot_df['index'],
            y=truth_uncertainty_plot_df['value'],
            mode='lines',
            fill=None,
            name='Real Values'
        )

        labels = go.Scatter(
           x=label_data['index'],
          y=label_data['value'],
         mode='lines',
         fill='tonexty' ,
         name='labels'
         )
        data = [upper_trace, lower_trace, real_trace]

        fig = go.Figure(data=data)
        fig.update_layout(title='Uncertainty MCDropout Test Data',
                            xaxis_title='index',
                            yaxis_title='value',
         legend_font_size=14,
                          )
        fig.show()
        bounds_df = pd.DataFrame()

        # Using 99% confidence bounds
        bounds_df['lower_bound'] = test_uncertainty_plot_df['lower_bound']
        bounds_df['prediction'] = test_uncertainty_plot_df['value_mean']
        bounds_df['real_value'] = truth_uncertainty_plot_df['value']
        bounds_df['upper_bound'] = test_uncertainty_plot_df['upper_bound']

        bounds_df['contained'] = ((bounds_df['real_value'] >= bounds_df['lower_bound']) &
                                  (bounds_df['real_value'] <= bounds_df['upper_bound']))

        print("Proportion of points contained within 99% confidence interval:",
              bounds_df['contained'].mean())
        predictedanomaly = bounds_df.index[~bounds_df['contained']]

        N = 15
        newarr = []


        for i in range(len(predictedanomaly) - N):
            if (predictedanomaly[i] + 1 == predictedanomaly[i + 1] and predictedanomaly[i + 1] + 1 == predictedanomaly[
               i + 2] and predictedanomaly[i+3]+1==predictedanomaly[i+4] and predictedanomaly[i+4]+1==predictedanomaly[i+5]
                and  predictedanomaly[i+5]+1==predictedanomaly[i+6] and predictedanomaly[i+6]+1==predictedanomaly[i+7]
                and predictedanomaly[i+7]+1==predictedanomaly[i+8]
                and predictedanomaly[i+8]+1==predictedanomaly[i+9]
                and predictedanomaly[i+9]+1==predictedanomaly[i+10]
                and predictedanomaly[i+10]+1==predictedanomaly[i+11]
                and predictedanomaly[i+11]+1==predictedanomaly[i+12]
                and predictedanomaly[i + 12] + 1 == predictedanomaly[i + 13]
                and predictedanomaly[i + 13] + 1 == predictedanomaly[i + 14]
                and predictedanomaly[i + 14] + 1 == predictedanomaly[i + 15]):

                newarr.append(predictedanomaly[i])
        #        newarr.append(predictedanomaly[i + 1])
         #       newarr.append(predictedanomaly[i + 2])


        predicteddanomaly = list(set(newarr))



        realanomaly = label_data['index']

        predicter=list(range(len(test_uncertainty2)))

        a1 = pd.DataFrame(index=range(len(test_uncertainty2)),columns=range(2))
        a1.columns=['index','value']

        a2 = pd.DataFrame(index=range(len(test_uncertainty2)), columns=range(2))
        a2.columns = ['index','value']




        for i in range(len(predicter)):
            if i in predicteddanomaly:
                a1.iloc[i,1]=1
            else:
                a1.iloc[i,1]=0

        for i in range(len(predicter)):
            if i in realanomaly:
                a2.iloc[i,1] = 1
            else:
                a2.iloc[i, 1] = 0



        y_real=a2.value
        y_real=y_real.astype(int)
        y_predi=a1.value
        y_predi=y_predi.astype(int)


        cm = confusion_matrix(y_true=y_real, y_pred=y_predi)
        cm_plot_labels = ['no_anomaly', 'had_anomaly']
        plot_confusion_matrix(cm=cm, classes=cm_plot_labels, title='Confusion Matrix')

        # tp = len([np.where(predicteddanomaly == x)[0] for x in realanomaly])
        #fp = len(predicteddanomaly) - tp
        #fn = 0
        #tn = len(truth_uncertainty_plot_df) - tp - fp - fn

        tp=cm[0][0]
        fp=cm[0][1]
        fn=cm[1][0]
        tn=cm[1][1]

        precision1 = tp / (tp + fp)
        recall1 = tp / (tp + fn)
        Accuracy1 = (tp + tn) / len(truth_uncertainty_plot_df)
        F11 = 2 / ((1 / precision1) + (1 / recall1))
        print('precision', precision1, 'Signal', name)
        print('recall', recall1, 'Signal', name)
        print('Accuracy', Accuracy1, 'Signal', name)
        print('F1', F11, 'Signal', name)
        precision.append(precision1)
        F1.append(F11)
        Accuracy.append(Accuracy1)
        recall.append(recall1)

        im=im+1

        #matched_indices = list(i_anom_predicted & true_indices_flat)


recall_final=mean(recall)
precision_final=mean(precision)
F1_final=mean(F1)
Accuracy_final=mean(Accuracy)
#cm = confusion_matrix(y_true=test_labels, y_pred=predicteddanomaly)

################################################################################

# %%

## New Test Bayesian LSTM AE





