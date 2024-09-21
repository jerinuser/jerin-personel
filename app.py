from flask import Flask, render_template, request

app = Flask(__name__)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/submit', methods=['POST'])
def submit():
    data = request.form['field_name']
    # Process data
    return 'Data submitted!'

if __name__ == '__main__':
    app.run(debug=True)