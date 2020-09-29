from keras import Sequential, Model
from keras.layers import Conv2D, MaxPooling2D, ZeroPadding2D, Input, Reshape
from keras.layers.core import Dense, Activation, Dropout, Flatten
from keras.models import load_model
from keras.utils.np_utils import to_categorical as to_cat 
from sklearn.preprocessing import StandardScaler 
import pandas
path_csv="user_data0.csv"
x_size = 13
data = pandas.read_csv(path_csv, sep=";")
Xr = data.values[:, 0:x_size]
Yr = data.values[:, x_size:]
Y = []
for i in range(4):
    Y.append(to_cat(Yr[:,i]))


print(Y[0].shape)
print(Y[1].shape)
print(Y[2].shape)
print(Y[3].shape)

scaler = StandardScaler()
X = scaler.fit_transform(Xr)

print(X.shape)
input = Input(shape=(x_size,))
h1 = Dense(100, activation="relu")(input)
h1 = Dropout(0.4)(h1)
h1 = Dense(100, activation="relu")(h1)
out_1 = Dense(Y[0].shape[1], activation='relu', name="interface_template")(h1)
out_2 = Dense(Y[1].shape[1], activation='relu', name="interface_font")(h1)
out_3 = Dense(Y[2].shape[1], activation='relu', name="interface_layout")(h1)
out_4 = Dense(Y[3].shape[1], activation='relu', name="interface_quality")(h1)
model = Model(input, [out_1,out_2,out_3,out_4])
model.compile(optimizer='adam', loss='mse', metrics=['accuracy'])
model.summary()


history = model.fit(X, [Y[0],Y[1],Y[2],Y[3]],
                    batch_size=10,
                    epochs=3,
                    validation_split=0.2)

model.save("adaptation_model.h5")