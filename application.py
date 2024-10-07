from main import model
import numpy as np

sample_ = 1000

# Количество примеров
in_ = 10  # Размерность каждого примера
out_ = 1
He = [32,128,32]
h = [in_,*He,out_]
# Генерация данных
X = np.random.rand(sample_, in_)

Y = np.sin(np.sum(X**2,axis=1)) 
Y = Y.reshape(Y.shape[0],1)

sem = 0.8
X_train,X_test,Y_train,Y_test = X[:int(sample_*sem)],X[int(sample_*sem):sample_],Y[:int(sample_*sem)],Y[int(sample_*sem):sample_]

NN = model(h,lr=1e-3)
NN.fit(X_train,X_test,Y_train,Y_test,epoch=1000,batch_size=128)
