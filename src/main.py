import flask
from .utils.log import logger
from .dp.module import DPModule
import os

#********************************初始化工作***************************
HOST="localhost"
PORT=5000

app = flask.Flask(__name__)
dp_module=DPModule()
dp_module.load()

#*******************************http调用函数***********************
@app.route('/')
def hello_world():
    return 'Hello, World!'

#示例:http://localhost:5000/dp_predict?words=我 爱pos=n v

@app.route( "/dp_predict", methods=['POST', 'GET'])
def predict():
    words = flask.request.args.get('words', '')
    pos = flask.request.args.get('pos', '')
    words=list(str(words).strip().split())
    pos=list(str(pos).strip().split())
    # print(words)
    # print(pos)
    # return ' '.join(words)+"##"+' '.join(pos)
    result=dp_module.predict(words,pos)
    return flask.jsonify({
        'state': 'OK',
        'result': {
            'arcs': result[0],
            'rels': result[1]
        }
    })


def main():
    logger.info("flask服务启动")
    app.run(host=HOST, port=PORT)