from keras.models import Sequential
from keras.layers import Dense
from keras.wrappers.scikit_learn import KerasRegressor
import numpy
from keras import backend as K
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error
# fix random seed for reproducibility
seed = 7
numpy.random.seed(seed)
# load dataset with position,orientation and joint angles
dataset = numpy.loadtxt("data.csv", delimiter=",")
print(dataset.shape)
# split into input (X) and output (Y) variables
X = dataset[:1000,:7]
Y = dataset[:1000,7:]
X_test = dataset[1000:,:7]
Y_test = dataset[1000:,7:]
print(X.shape,Y.shape)

##define base model
def base_model():
     model = Sequential()
     model.add(Dense(32, input_dim=7, init='normal', activation='relu'))
     model.add(Dense(64, init='normal', activation='relu'))
     model.add(Dense(128, init='normal', activation='relu'))
     model.add(Dense(32, init='normal', activation='relu'))
     model.add(Dense(6, init='normal'))
     model.compile(loss='mean_absolute_error', optimizer = 'adam')
     return model

clf = KerasRegressor(build_fn=base_model, epochs=500, batch_size=20,verbose=2)

clf.fit(X,Y)
res = clf.predict(X_test)
print(res)


score = mean_absolute_error(Y_test, res)
print(score)
# ... code
K.clear_session()
