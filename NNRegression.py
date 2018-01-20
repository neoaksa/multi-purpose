import numpy as np
import pylab as pl
import mlp

x = np.ones((1, 40)) * np.linspace(0, 1, 40)

t = np.sin(2 * np.pi * x) + np.cos(4 * np.pi * x) + np.random.randn(40) * 0.2

x = x.T
print(x.shape)
t = t.T
x = np.linspace(0, 1, 40)

# np.shape(x.T)
x = np.linspace(0, 1, 40).reshape((1, 40))

# pl.plot(x.T, t, '.')
# pl.show()
train = x[0::2,:]
test = x[1::4,:]
valid = x[3::4,:]
traintarget = t[0::2,:]
testtarget = t[1::4,:]
validtarget = t[3::4,:]

net = mlp.mlp(train,traintarget,3,outtype='linear')
net.mlptrain(train,traintarget,0.25,101)