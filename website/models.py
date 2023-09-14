#models = tables in database
from . import db
from flask_login import UserMixin
from sqlalchemy.sql import func

class User(db.Model, UserMixin):
    id = db.Column(db.Integer, primary_key=True)
    email = db.Column(db.String(150), unique=True)
    password = db.Column(db.String(150))
    first_name = db.Column(db.String(150))
    products = db.relationship('Product')

class Product(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    name = db.Column(db.String(150))
    date = db.Column(db.DateTime(timezone=True), default=func.now())
    user_id = db.Column(db.Integer, db.ForeignKey('user.id'))
    sentiments = db.relationship('Sentiment')

class Sentiment(db.Model):
    id =  db.Column(db.Integer, primary_key=True)
    tweet = db.Column(db.String(280))
    polarity = db.Column(db.String(1))
    product_id = db.Column(db.Integer, db.ForeignKey('product.id'))
    