import  numpy as np
a=np.random.rand(5,1,8,6)
print(a.shape)
b=np.random.rand(5,3,6,6)
print(b.shape)
c=a @ b
print(c.shape)