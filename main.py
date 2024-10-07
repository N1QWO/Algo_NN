import numpy as np

class model:
  def __init__(self,l,lr=1e-3,alfa=0.9,beta = 0.8,labd=0.95):
    self.w = self.init_weights(l)
    self.grad = [np.zeros((l[i],l[i+1])) for i in range(len(l)-1)]
    self.backprog = [{'bw':np.zeros((l[i],l[i+1])),'bsi':np.zeros((l[i],l[i+1]))} for i in range(len(l)-1)]
    self.grad_g = [np.zeros((l[i],l[i+1])) for i in range(len(l)-1)]

    self.g = [np.zeros((l[i],l[i+1])) for i in range(len(l)-1)]
    self.u = [np.zeros((l[i],l[i+1])) for i in range(len(l)-1)]

    self.l = l
    self.lr = lr
    self.alfa = alfa
    self.beta = beta
    self.labd = labd

  def backward(self,y):
    ls = self.der_loss(self.backprog[-1]['bw'],y)
    grad = ls
    grad = grad.reshape(grad.shape[0], grad.shape[1], 1)

    for i in range(len(self.l)-2,-1,-1):

      acti = self.der_relu(self.backprog[i]['bsi'])

      tl = np.tile(acti.reshape((self.backprog[i]['bsi'].shape[0],self.backprog[i]['bsi'].shape[1],1)),(1,1,grad.shape[2]))
      grad_reshaped = (grad * tl)

      bw_reshaped = np.tile(self.backprog[i]['bw'].reshape(self.backprog[i]['bw'].shape[0], 1, self.backprog[i]['bw'].shape[1]),(1,grad.shape[2],1))
      grad = np.matmul(grad_reshaped ,bw_reshaped)

      grad = np.transpose(grad, (0, 2, 1))
      tm = np.mean(grad,axis=0)
      self.u[i] = self.beta*self.u[i] + (1-self.beta)*tm
      self.g[i] = self.labd*self.g[i] + (1-self.labd)*(tm)**2
      self.w[i] = self.w[i] - self.lr * self.u[i] / (np.sqrt(self.g[i]+1e-10) * (i+1))
      
  def z_score_normalization(self,grad):
      mean = np.mean(grad, axis=0)
      std = np.std(grad, axis=0)
      return (grad - mean) / std
  def init_weights(self,layers):
    weights = []
    for i in range(len(layers) - 1):
        a = layers[i]

        w = np.random.randn(layers[i], layers[i+1]) * np.sqrt(1 / (2 * a)) + 1
        weights.append(w)
    return weights

  def der_loss(self,pred,y):
    return pred-y

  def der_sigmoid(self,X):
    return self.softmax(X)*(1-self.softmax(X))

  def relu(self, X):
      return np.maximum(0, X)

  def der_relu(self, X):
      return (X > 0).astype(float)
  def der_softmax(self,X):
    sf = self.softmax(X)
    return sf*(1-sf)

  def softmax(self,X):
    ep = np.exp(X)
    sm = np.sum(ep,axis=1).reshape(ep.shape[0],1)
    prob = ep/sm
    return prob

  def sigmoid(self,X):
    return 1/(1+np.exp(-X))

  def loss(self,pred,y):
    return np.mean(0.5*(pred-y)**2)
  def rmse(self,pred,y):
    return np.sqrt(np.sum((pred-y)**2))
  def forward(self,X):
    h = X
    for i in range(len(self.l)-1):
        self.backprog[i]['bw'] = h

        h = np.dot(h,self.w[i])
        self.backprog[i]['bsi'] = h
        h = self.relu(h)
    return h

  def fit(self,X_train,X_test,Y_train,Y_test,epoch=100,batch_size = 100):
    #print('layers',self.l)
    num_samples = len(X_train)
    num_batches = (num_samples + batch_size - 1) // batch_size
    for i in range(epoch):
      for batch in range(num_batches):
        start_idx = batch * batch_size
        end_idx = min((batch + 1) * batch_size, num_samples)

        X_batch = X_train[start_idx:end_idx]
        y_batch = Y_train[start_idx:end_idx]
        predict =self.forward(X_batch)
        self.backward(y_batch)
      predict =self.forward(X_test)
      loss = self.loss(predict,Y_test)
      if i%10==0:
        print(f'epoch {i}; train loss {loss}')

  def predict(self,X,y=None):
    if y is None:
      return self.forward(X)
    else:
      predict =self.forward(X)
      loss = self.loss(predict,y)
      return loss

