
# from flask import Flask, render_template, request
# import pickle
# import numpy as np

# model = pickle.load(open('model_train.pkl','rb'))
# app = Flask(__name__)

# @app.route('/')
# def index():
#     return render_template('index.html')

# @app.route('/predict', methods=['POST'])
# def predict_price():
#     sepal_length = str(request.form.get('sepal length (cm)'))
#     sepal_width = str(request.form.get('sepal width (cm)'))
#     petal_length = str(request.form.get('petal length (cm)'))
#     petal_width = str(request.form.get('petal width (cm)'))
    
# # prediction
#     result = model.predict(np.array([sepal_length,sepal_width,petal_length,petal_width],dtype=object).reshape(1,4))
#     result =  "The class of Iris Flower should be {} ".format(str(result[0]))

#     return render_template('index.html', result=result)
# if __name__ == '__main__':
#     app.run(debug=True)










