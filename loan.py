from flask import Flask,request,render_template
import pickle
import numpy as np

app = Flask(__name__)

def load_model():
    with open('Loan.pkl','rb') as file:
        data = pickle.load(file)
        
    return data

data = load_model()
model = data['model']
le_res_state = data['le_res_state']

@app.route('/')
def homepage():
    return render_template('loan.html')

@app.route('/detect',methods = ['POST'])
def make_detection():
    d1 = request.form['Work_Experience']
    d2 = request.form['House_Ownership']
    d3 = request.form['Vehicle_Ownership(car)']
    d4 = request.form['Residence_State']
    d5 = request.form['Years_in_Current_Residence']
    
    if  d2 == 'Rented':
        d2 = 2
    if d2 == 'Owned':
        d2 = 1
    if d2 == 'None':
        d2 = 0
        
    if d3 == 'Yes':
        d3 = 0
    if d3 == 'No':
        d3 = 1
        
    inp = np.array([[d1,d2,d3,d4,d5]])
    inp[:,3] = le_res_state.transform(inp[:,3])
    inp = inp.astype(int)
    
    prediction = model.predict(inp)
    
    if prediction == 1:
        text = "You're at risk of defaulting the loan"
    elif prediction == 0:
        text = "Not at the risk of defaulting the loan"
    
    return render_template('loan.html',pred = text)

if __name__ == '__main__':
    app.run(debug=True)