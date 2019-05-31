from flask import Flask,request,jsonify
from flask import send_file
from Prediction import P_Model

app = Flask(__name__)


@app.route('/test',methods=["POST"])
def check():
    data = request.files
    # f = request.files['image']
    # print(f.filename)
    
    #print(list(data_dict.keys()))
    if data:
    	P = P_Model()
    	res = P.check(data)
    	# del P
    	return jsonify(res)
    return "Image not found"
    


if __name__ == '__main__':
    app.run(debug=True)