import pandas as pd
import numpy as np
import tensorflow as tf

#Read from Excel
df = pd.read_excel(r'C:/Users/abdim/Desktop/FS/exceldata/final/FFN/06_Nur_normiert_Shaffeln_beschränkt_und_nach_PMV_sortierte_Fanger_generierte_Daten.xlsx')
read_data = df.drop('entry_id', 1)

#Data times separation
times = 24103
var = 6
x_train_times = int(times*60/100)
x_valid_times = int(times*20/100)

# split data into samples
X_train = read_data.iloc[0:x_train_times, 0:var]
X_train = X_train.values.tolist()

X_train_label = read_data.iloc[0:x_train_times, var]
X_train_label = X_train_label.values.tolist()

X_valid = read_data.iloc[x_train_times:(x_train_times+x_valid_times), 0:var]
X_valid = X_valid.values.tolist()

X_valid_label = read_data.iloc[x_train_times:(x_train_times+x_valid_times), 6]
X_valid_label = X_valid_label.values.tolist()

X_test = read_data.iloc[(x_train_times+x_valid_times):times, 0:var]
X_test = X_test.values.tolist()

X_test_label = read_data.iloc[(x_train_times+x_valid_times):times, var]
X_test_label = X_test_label.values.tolist()

#Change data Type
X_train = np.array(X_train)
X_train_label = np.array(X_train_label)

X_valid = np.array(X_valid)
X_valid_label = np.array(X_valid_label)

X_test = np.array(X_test)
X_test_label = np.array(X_test_label)

#define epoch and batch size
n_epoch = 500
batch = 200

# define model type
model = tf.keras.models.Sequential()
# building the model
model.add(tf.keras.layers.Input(6))
model.add(tf.keras.layers.Dense(48, activation='tanh'))
model.add(tf.keras.layers.Dense(12, activation='tanh'))
model.add(tf.keras.layers.Dense(1, activation='tanh'))
#define optimizer
opt = tf.keras.optimizers.Adam(learning_rate=1e-3)
#compile the model
model.compile(optimizer=opt,  
              loss='mse',  # how will we calculate our "error." Neural network aims to minimize loss.
              metrics=['accuracy'])  # what to track
#fiting the model
history = model.fit(X_train, X_train_label, epochs=n_epoch, batch_size=batch, validation_data=(X_valid, X_valid_label))

#model evaluation
val_loss, val_acc = model.evaluate(X_valid, X_valid_label)  # evaluate the out of sample data with model
print(val_loss)  # model's loss (error)
print(val_acc)  # model's accuracy


#Testing and Taking the test result
result = model.predict(X_test)

#extract and order the results
X_test_predict = [] #list to store PMV-test from input data
predict = [] #list to store predicted PMV from input data
predict0 = np.array(result).tolist()
for i in range(0, len(result)):
    X_test_predict.append(X_test_label[i])
    predict.extend(predict0[i])

# Writing to excel datas
myDict = history.history
dfresult = pd.DataFrame.from_dict(myDict, orient='index')
dfresult = dfresult.transpose()


writer = pd.ExcelWriter('C:/Users/abdim/Desktop/FS/exceldata/out.xlsx', mode='w')
n_epoch_list = np.arange(1, n_epoch+1, 1).tolist()
dfresult.insert(0, "n_epoch", n_epoch_list, True)
dfresult.to_excel(writer, index=False, sheet_name="tab", startcol=0, startrow=0)
writer.save()

writer = pd.ExcelWriter('C:/Users/abdim/Desktop/FS/exceldata/outpredict.xlsx', mode='w')

dfpredict = pd.DataFrame()
dfpredict.insert(0, "tests", X_test_predict, True)
dfpredict.insert(1, "predict", predict, True)
dfpredict.to_excel(writer, index=False, sheet_name="tab", startcol=0, startrow=0)
writer.save()

