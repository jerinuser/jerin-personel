from flask import Flask, render_template

app = Flask(__name__)

# Home route
@app.route('/')
def home():
    return "Welcome to the Home Page!"

# Hello route with a dynamic user name
@app.route('/hello/<name>')
def hello(name):
    return f"Hello, {name}!"

if __name__ == '__main__':
    app.run(
            debug=True,
            host='0.0.0.0',
            port=4004
            )
