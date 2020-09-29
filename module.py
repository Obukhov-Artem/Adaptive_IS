from flask import Flask, jsonify,request
from keras.models import load_model
nn = load_model("model.h5")

app = Flask(__name__) 
  


@app.route('/adapt',  methods=['POST'])
def set_user_param():
    user_param = ["role","sex","age","education","exp","w_screen","h_screen","hw_type","browser","lang","cpu","lan","gpu"]
    if not request.json:
        return jsonify({'error': 'пустой запрос'})
    elif not all(key in request.json['data'] for key in user_param):
        return jsonify({'error': 'неправильный запрос'})
    user_data = []
    for i in user_param: 
        user_data.append(int(request.json['data'][i]))
    
    interface_param = nn.predict(user_data) 

    return jsonify({'success': 'OK','interface_param':{'template':interface_param[0],	'size':interface_param[1],	'layout':interface_param[2],	'quality':interface_param[3]}})


if __name__ == '__main__':
    app.run(debug=True)
