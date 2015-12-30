import flask
import flask.ext.sqlalchemy
import flask.ext.restless
import time
from datetime import datetime

# Create the Flask application and the Flask-SQLAlchemy object.
app = flask.Flask(__name__)
app.config['DEBUG'] = True
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///msband2.db'
db = flask.ext.sqlalchemy.SQLAlchemy(app)

# Create your Flask-SQLALchemy models as usual but with the following two
# (reasonable) restrictions:
#   1. They must have a primary key column of type sqlalchemy.Integer or
#      type sqlalchemy.Unicode.
#   2. They must have an __init__ method which accepts keyword arguments for
#      all columns (the constructor in flask.ext.sqlalchemy.SQLAlchemy.Model
#      supplies such a method, so you don't need to declare a new one).
class Measurement(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    timestamp = db.Column(db.DateTime)
    hr = db.Column(db.Float)
    gsr = db.Column(db.Float)
    temp = db.Column(db.Float)
    accx = db.Column(db.Float)
    accy = db.Column(db.Float)
    accz = db.Column(db.Float)
    state = db.Column(db.String(80))

    def __init__(self, timestamp, hr, gsr, temp, accx, accy, accz, state):
        self.timestamp = timestamp
        self.hr = hr
        self.gsr = gsr
        self.temp = temp
        self.accx = accx
        self.accy = accy
        self.accz = accz
        self.state = state

    def __repr__(self):
        return '<Sensor Readings: State:{0}, hr:{1}, gsr:{2}, temp:{3}, acc:{4}'.format(self.state, self.hr, self.gsr, self.temp, self.acc)

# Create the database tables.
db.create_all()

# Create the Flask-Restless API manager.
manager = flask.ext.restless.APIManager(app, flask_sqlalchemy_db=db)

# Create API endpoints, which will be available at /api/<tablename> by
# default. Allowed HTTP methods can be specified as well.
manager.create_api(Measurement, methods=['GET', 'POST', 'DELETE', 'DELETE'])

# Set up Endpoints
# Main Methods and Routes
@app.route('/insert_set', methods=['POST'])
def insert_set():
    try:
        json = flask.request.get_json()
        measurement = Measurement(datetime.fromtimestamp(json['timestamp']/1000),
                                float(json['hr']),
                                float(json['gsr']),
                                float(json['temp']),
                                float(json['accx']),
                                float(json['accy']),
                                float(json['accz']), str(json['state']))
        db.session.add(measurement)
        db.session.commit()
        return flask.jsonify(request='good'), 200
    except Exception, e:
        return flask.jsonify(error=str(e)), 500

@app.route('/')
def main():
    return "Main2.py Working!"

# start the flask loop
