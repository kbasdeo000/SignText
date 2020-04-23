from flask import Flask

# initialize a flask object
app = Flask(__name__)
app.config["DEBUG"] = True


# Home page route:
@app.route('/')
def home():
    return "home"


app.run()
