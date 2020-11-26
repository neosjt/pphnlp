# import  numpy as np
# a=np.random.rand(5,1,8,6)
# print(a.shape)
# b=np.random.rand(5,3,6,6)
# print(b.shape)
# c=a @ b
# print(c.shape)

# from src.utils.log import logger
#
# logger.debug("fuck")

# import  os
# ROOT_PATH= os.path.dirname(os.path.abspath(__file__))
#
# print(ROOT_PATH)

from src.dp.module import DPModule
from src.dp.model import BiaffineParser
from src.dp.config import CONFIG_PATH,MODEL_PATH,DEVICE
import pickle
import torch
# with open(CONFIG_PATH, 'rb') as fr:
#     config = pickle.load(fr)
#
# model = BiaffineParser(config)
# torch.save(model,"./dp.pth")
# model=torch.load("./dp.pth")
# model.to(DEVICE)

# module=DPModule()
# module.train()
# module.load()
# wordlist,poslit=['没有','任何','一个','政党'],['v','rz','m','n']
# a,b=module.predict(wordlist,poslit)
# print(a,b)
#module.train()

# import math
# a=[
#     [1,2,3],
#     [4,5]
#    ]
#
# b=max(list(map(len,a)))
# print(b)
# print(max(6,7))
# import numpy as np
# mask=np.array([[False,True,True],[False,True,True]])
# a=np.random.rand(2,3,4)
# print(a.shape)
# print(a)
# b=a[mask]
# print(b.shape)
# print(b)

# from flask import Flask
# app = Flask(__name__)
#
# @app.route('/')
# def hello_world():
#     return 'Hello, World!'
#
# if __name__ == '__main__':
#     app.run(debug=True, host='0.0.0.0', port=5000)
a=([1],[2])+[3]
print(a)