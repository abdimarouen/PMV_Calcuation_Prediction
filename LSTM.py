import pandas as pd
import numpy as np
from numpy import array
from keras.models import Sequential
from keras.layers import Dense, Dropout, LSTM, SimpleRNN, Activation

#Read from Excel
df = pd.read_excel(r'C:/Users/abdim/Desktop/FS/exceldata/final/LSTM/data/04_Normiert_Nur_PMV_begrenzt_Araar_daten.xlsx')  # Normiert_Nur_PMV_Araar_daten.xlsx
read_data = df.drop('entry_id', 1)
print(read_data)

#Data times separation
times = 2222
x_train_times = int(times*60/100)
x_test_times = int(times*40/100)

X_train = read_data.iloc[0:x_train_times, 0]
print(X_train)
X_train = X_train.tolist()

X_test = read_data.iloc[x_train_times:times, 0] #x_train_times= 1333 so X_test: 1344->ende
print(X_test)
X_test = X_test.tolist()

# define a function that split a univariate sequence into samples
def split_sequence(sequence, n_steps):
    X, y = list(), list()
    for i in range(len(sequence)):
        # find the end of this pattern
        end_ix = i + n_steps
        # check if we are beyond the sequence
        if end_ix > len(sequence) - 1:
            break
        # gather input and output parts of the pattern
        seq_x, seq_y = sequence[i:end_ix], sequence[end_ix]
        X.append(seq_x)
        y.append(seq_y)
    return array(X), array(y)


n_steps = 10 #define time step
n_epoch = 200

# split data into samples
X, y = split_sequence(X_train, n_steps)
X_test, X_test_label = split_sequence(X_test, n_steps)
print("//////////////////")
print(X.shape)
print("//////////////////")
print(y.shape)
# reshape from [samples, timesteps] into [samples, timesteps, features]
n_features = 1

X = X.reshape((X.shape[0], X.shape[1], n_features))
X_test = X_test.reshape((X_test.shape[0], X_test.shape[1], n_features))

# define model
model = Sequential()
model.add(LSTM(10, activation='tanh', input_shape=(n_steps, n_features)))
model.add(Dense(1))
model.compile(optimizer='adam', loss='mse', metrics=['accuracy'])

# fit model
history = model.fit(X, y, epochs=n_epoch, verbose=2)

#Testing and Taking the test result
result = model.predict(X_test)
weight = model.get_weights()
#extract and order the results
X_test_predict = [] #list to store PMV-test from input data
predict = []  #list to store predicted PMV from input data
predict0 = np.array(result).tolist()

for i in range(0, len(result)):
    X_test_predict.append(X_test_label[i])
    predict.extend(predict0[i])

# Writing to excel datas

myDict = history.history
dfresult = pd.DataFrame.from_dict(myDict, orient='index')
dfresult = dfresult.transpose()

writer = pd.ExcelWriter('C:/Users/abdim/Desktop/FS/exceldata/outfinalLSTM.xlsx',
                        mode='w')  # for an earlier version of Excel, you may need to use the file extension of 'xls'
n_epoch_list = np.arange(1, 200 + 1, 1).tolist()
dfresult.insert(0, "n_epoch", n_epoch_list, True)
dfresult.to_excel(writer, index=False, sheet_name="tab", startcol=0, startrow=0)
writer.save()

writer = pd.ExcelWriter('C:/Users/abdim/Desktop/FS/exceldata/outpredictfinalLSTM.xlsx',
                        mode='w')  # for an earlier version of Excel, you may need to use the file extension of 'xls'


dfpredict = pd.DataFrame()
dfpredict.insert(0, "tests", X_test_predict, True)
dfpredict.insert(1, "predict", predict, True)

dfpredict.to_excel(writer, index=False, sheet_name="tab", startcol=0, startrow=0)
writer.save()

print("times: ", times, "--", "x_train_times: ", x_train_times, "--", "x_valid_times: ", x_test_times)

ls = pd.DataFrame()
writer = pd.ExcelWriter('C:/Users/abdim/Desktop/FS/exceldata/weight_lstm.xlsx', mode='w')
ls.to_excel(writer, index=False, sheet_name="tab", startcol=0, startrow=0)

startrow = writer.sheets['tab'].max_row

for w in weight:
    ls = pd.DataFrame(w)
    ls.to_excel(writer, index=False, sheet_name="tab", startcol=0, startrow=writer.sheets['tab'].max_row)

writer.save()