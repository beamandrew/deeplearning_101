from sklearn.datasets import make_circles

from keras.layers import Input, Dense
from keras.models import Model

X, y = make_circles(n_samples=5000, factor=.3, noise=.05)
X_train = X[:4000]
y_train = y[:4000]

X_val = X[4000:]
y_val = y[4000:]


n_cols = X.shape[1]

## Define neural network parameters ##
n_hidden = 512


### First do logistic regression model ##
## Define the input tensor ##
inputs = Input(shape=(n_cols,))
predictions = Dense(1, activation='sigmoid')(inputs)
# this creates a model that includes
# the Input layer and three Dense layers
model = Model(input=inputs, output=predictions)
model.compile(optimizer='adam',
              loss='binary_crossentropy',
              metrics=['accuracy'])
model.fit(X_train, y_train,validation_data=[X_val,y_val])


## Now do one hidden layer with 512 units ##
inputs = Input(shape=(n_cols,))
x = Dense(n_hidden, activation='relu')(inputs)
predictions = Dense(1, activation='sigmoid')(x)

# this creates a model that includes
# the Input layer and three Dense layers
model = Model(input=inputs, output=predictions)
model.compile(optimizer='adam',
              loss='binary_crossentropy',
              metrics=['accuracy'])
model.fit(X_train, y_train,validation_data=[X_val,y_val])

