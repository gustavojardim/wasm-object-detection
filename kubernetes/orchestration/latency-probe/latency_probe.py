from flask import Flask
import time

app = Flask(__name__)

@app.route('/ping')
def ping():
    return {'timestamp': time.time()}, 200

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8082)