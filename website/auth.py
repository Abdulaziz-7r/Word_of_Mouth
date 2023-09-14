from flask import Blueprint, render_template, request, flash, redirect, url_for
from website import views
from .models import User
from . import db
from werkzeug.security import generate_password_hash, check_password_hash
from flask_login import login_required, login_user, logout_user, current_user


auth = Blueprint('auth', __name__)

@auth.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        email = request.form.get('email')
        password = request.form.get('password')

        # performing query to select the user based on his email
        user = User.query.filter_by(email=email).first()

        # check if the email exist
        if user: 
            # check if the password is correct
            if check_password_hash(user.password, password):
                flash('Logged in successfully', category='success')
                login_user(user, remember=True)
                # the user will be redirect to main menu page
                return redirect(url_for('views.main_menu'))
            else:
                flash('Incorrect password, try again', category='error')
        else:
            flash('Email does not exist', category='error')

    return render_template('login.html', user=current_user)

@auth.route('/logout')
@login_required
def logout():
    logout_user()
    return redirect(url_for('auth.login'))

@auth.route('/sign-up', methods=['GET', 'POST'])
def sign_up():
    if request.method == 'POST':
        # Assigning the form data sent using POST method  
        email = request.form.get('email')
        first_name = request.form.get('firstName')
        password1 = request.form.get('password1')
        password2 = request.form.get('password2')

        # performing query to select the user based on his email
        user = User.query.filter_by(email=email).first()

        #Validating the entered account information
        if user:
             flash('Email already exist', category='error')
        elif len(email) < 4 :
            flash('Email must be greater than 3 characters', category='error')
        elif len(first_name) < 2 :
            flash('First name must be greater than 1 character', category='error')
        elif len(password1) < 7   :
            flash('Password must be greater than 6 characters.', category='error')
        elif password1 != password2 :
            flash('Password don\'t match.', category='error')
        else :
            # Creating a new user 
            new_user = User(email=email, first_name=first_name, password=generate_password_hash(password1, method='sha256'))
            db.session.add(new_user)
            db.session.commit()
            flash('Account created!', category='success')
            login_user(new_user, remember=True)
            # After creating a new user, the user will be redirect to main menu page
            return redirect(url_for('views.main_menu'))
    
    return render_template('sign_up.html', user=current_user)       