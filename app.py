import datetime
import folium
from flask import Flask, render_template, request, redirect, flash
from flask_login import LoginManager, login_user, UserMixin, logout_user
from werkzeug.security import generate_password_hash, check_password_hash
from flask_sqlalchemy import SQLAlchemy
import random
import json
from keras.models import load_model
import numpy as np
import pickle
from nltk.stem import WordNetLemmatizer
import nltk
nltk.download('popular')
lemmatizer = WordNetLemmatizer()


model = load_model('model.h5')
intents = json.loads(open('data.json').read())
words = pickle.load(open('texts.pkl', 'rb'))
classes = pickle.load(open('labels.pkl', 'rb'))


app = Flask(__name__)

# Database work
app.app_context().push()
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///brain_mentor.db'
app.config['SECRET_KEY'] = 'a0455de1e15d46ad995c0d40928916ef'
db = SQLAlchemy(app)
login_manager = LoginManager(app)
login_manager.login_view = 'index'


# Create a Model of User
class Patients(UserMixin, db.Model):
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(80), unique=True, nullable=False)
    email = db.Column(db.String(100), nullable=False, unique=True)
    password = db.Column(db.String(100), nullable=False)

    def __repr__(self):
        return '<User %r>' % self.username

# Create a Model of Blogs


class Blogs(db.Model):
    blog_id = db.Column(db.Integer, primary_key=True)
    blog_type = db.Column(db.String(80), nullable=False)
    tit_name = db.Column(db.String(80), nullable=False)
    author = db.Column(db.String(40), nullable=False)
    content = db.Column(db.Text(), nullable=False)
    pub_date = db.Column(db.DateTime(), nullable=False,
                         default=datetime.datetime.utcnow)

    def __repr__(self):
        return '<Blogs %r>' % self.tit_name

# User Loader Function


@login_manager.user_loader
def load_user(user_id):
    return Patients.query.get(int(user_id))


@app.route('/', methods=["GET", "POST"])
def index():
    if request.method == 'POST':
        username = request.form.get('username')
        email = request.form.get('email')
        password = request.form.get('password')
        action = request.form.get('action')

        existing_user = Patients.query.filter_by(username=username).first()

        if not existing_user:
            if action == "signup":
                # It's a signup attempt
                new_user = Patients(username=username,
                                    email=email, password=password)
                db.session.add(new_user)
                db.session.commit()
                flash('Account has been successfully created!', 'success')
                return redirect('/home')
            else:
                # It's a login attempt with non-existent credentials
                flash('Invalid credentials. Please try again.', 'danger')
        else:
            # If the username exists, it's a login attempt
            if existing_user.password == password and email == existing_user.email and existing_user.username == username:
                login_user(existing_user)
                flash('Login successful!', 'success')
                return redirect('/home')
            else:
                flash('Invalid credentials. Please try again.', 'danger')

    return render_template('index.html')


@app.route('/logout')
def logout():
    logout_user()
    return redirect('/')


@app.route('/home')
def Home():
    blogss = Blogs.query.all()
    return render_template('Home.html', blogs=blogss)


@app.route('/patient_signup')
def Patient_Signup():
    return render_template('Patient_Signup.html')


@app.route('/detection')
def Detection():
    return render_template('Detection.html')


@app.route('/doctors')
def Doctors():
    return render_template('Doctors.html')


@app.route('/appointment')
def Appointment():
    return render_template('Appointment.html')


@app.route('/patient_treat')
def Patient_Treat():
    return render_template('Patient_Treat.html')


# Locations

@app.route('/doctors_marker')
def Doctors_Markers():
    # Coordinates for the center of Islamabad, Pakistan
    pakistan_isb_coordinates = [33.700266560649226, 73.05169660795143]

    # Create a Folium map
    map = folium.Map(
        location=pakistan_isb_coordinates,
        tiles='OpenStreetMap',
        zoom_start=12,  # Adjust the zoom level as needed
    )

    # Add a marker to the map
    folium.Marker(
        location=pakistan_isb_coordinates,
        popup='Islamabad, Pakistan',
        tooltip='Islamabad, Pakistan',
        icon=folium.Icon(color='red',icon='location-dot',prefix='fa')
    ).add_to(map)

 # 1 

    # Add a marker to the map
    marker_location = [34.223299962163615, 72.263259854991]
    marker_popup_content = """
    <div>
        <h4 style="color:blue;">Brain & Spine Clinic</h4>
        <img src="https://media.istockphoto.com/id/177373093/photo/indian-male-doctor.jpg?s=612x612&w=0&k=20&c=5FkfKdCYERkAg65cQtdqeO_D0JMv6vrEdPw3mX1Lkfg=" alt="Image" style="width:100px;height:100px;">
        <p>Address: Clinic Address, City, Country</p>
        <p>Email: clinic@example.com</p>
        <p>Open Hours: Monday to Friday, 9:00 AM - 5:00 PM</p>
    </div>
    """

    folium.Marker(
        location=marker_location,
        tooltip='Brain & Spine Clinic',
        popup=folium.Popup(html=marker_popup_content, max_width=300),
         icon=folium.Icon(color='green',icon='user-doctor',prefix='fa')
    ).add_to(map)


# 2
    # Add a marker to the map
    marker_location = [33.71081684674506, 73.02100506625507]
    marker_popup_content = """
    <div>
        <h4 style="color:blue;">Dr. Alamgir Neuro Surgeon</h4>
        <img src="https://media.istockphoto.com/id/177373093/photo/indian-male-doctor.jpg?s=612x612&w=0&k=20&c=5FkfKdCYERkAg65cQtdqeO_D0JMv6vrEdPw3mX1Lkfg=" alt="Image" style="width:100px;height:100px;">
        <p>Address: Clinic Address, City, Country</p>
        <p>Email: clinic@example.com</p>
        <p>Open Hours: Monday to Friday, 9:00 AM - 5:00 PM</p>
    </div>
    """

    folium.Marker(
        location=marker_location,
        tooltip='Dr. Alamgir Neuro Surgeon',
        popup=folium.Popup(html=marker_popup_content, max_width=300),
         icon=folium.Icon(color='green',icon='user-doctor',prefix='fa')
    ).add_to(map)

    # 3
    # Add a marker to the map
    marker_location = [33.694266983374575, 72.99629625361848]
    marker_popup_content = """
    <div>
        <h4 style="color:blue;">Dr Akbar Khan</h4>
        <img src="https://media.istockphoto.com/id/177373093/photo/indian-male-doctor.jpg?s=612x612&w=0&k=20&c=5FkfKdCYERkAg65cQtdqeO_D0JMv6vrEdPw3mX1Lkfg=" alt="Image" style="width:100px;height:100px;">
        <p>Address: Clinic Address, City, Country</p>
        <p>Email: clinic@example.com</p>
        <p>Open Hours: Monday to Friday, 9:00 AM - 5:00 PM</p>
    </div>
    """

    folium.Marker(
        location=marker_location,
        tooltip='Dr Akbar Khan',
        popup=folium.Popup(html=marker_popup_content, max_width=300),
        icon=folium.Icon(color='green',icon='user-doctor',prefix='fa')
    ).add_to(map)


    # Add a marker to the map
    marker_location = [33.67596469823713, 73.06711812392459]
    marker_popup_content = """
    <div>
        <h4 style="color:blue;">Prof. Dr Inayat Ullah Khan</h4>
        <img src="https://media.istockphoto.com/id/177373093/photo/indian-male-doctor.jpg?s=612x612&w=0&k=20&c=5FkfKdCYERkAg65cQtdqeO_D0JMv6vrEdPw3mX1Lkfg=" alt="Image" style="width:100px;height:100px;">
        <p>Address: Clinic Address, City, Country</p>
        <p>Email: clinic@example.com</p>
        <p>Open Hours: Monday to Friday, 9:00 AM - 5:00 PM</p>
    </div>
    """

    folium.Marker(
        location=marker_location,
        tooltip='Prof. Dr Inayat Ullah Khan',
        popup=folium.Popup(html=marker_popup_content, max_width=300),
        icon=folium.Icon(color='green',icon='user-doctor',prefix='fa')
    ).add_to(map)


    # Add markers for hotel, hospital, restaurant, and museum
    markers = [
    {"location": [33.6844, 73.0479], "name": "Hotel ABC", "type": "hotel", "icon": "bed"},
    {"location": [33.6804, 73.0471], "name": "City Hospital", "type": "hospital", "icon": "hospital"},
    {"location": [33.6865, 73.0512], "name": "Fine Dining Restaurant", "type": "restaurant", "icon": "cutlery"},
    {"location": [33.6883, 73.0495], "name": "National Museum", "type": "museum", "icon": "university"},
]

    for marker in markers:
        folium.Marker(
        location=marker["location"],
        popup=folium.Popup(html=f"<b>{marker['name']}</b>", max_width=300),
        icon=folium.Icon(color='blue' if marker["type"] == "hospital" else 'green', icon=marker["icon"], prefix='fa')
    ).add_to(map)




    return map._repr_html_()









# Blog Post
@app.route('/blog', methods=["GET", "POST"])
def Blog():
    if request.method == 'POST':
        type = request.form.get('type')
        title = request.form.get('title')
        author = request.form.get('author')
        content = request.form.get('text')
        blog = Blogs(blog_type=type, tit_name=title,
                     author=author, content=content)
        db.session.add(blog)
        db.session.commit()
        flash('Post Sucessfully Created', 'success')
        return redirect('/home')
    return render_template('Blogs.html')

# Read Blog


@app.route("/blogs_deatil/<int:id>")
def blogs_details(id):
    blog = Blogs.query.get(id)
    return render_template('Blogs_Details.html', blog=blog)

# Delete Blog


@app.route("/delete/<int:id>")
def del_post(id):
    blog = Blogs.query.get(id)
    db.session.delete(blog)
    db.session.commit()
    flash('Your Post Has Been Sucessfully Deleted', 'success')
    return redirect('/home')

# Edit Blog


@app.route("/edit/<int:id>", methods=['GET', 'POST'])
def edit_post(id):
    blog = Blogs.query.get(id)
    if request.method == 'POST':
        blog.tit_name = request.form.get('title')
        blog.author = request.form.get('author')
        blog.content = request.form.get('text')
        db.session.commit()
        flash('Your Post Has Been Sucessfully Edit', 'success')
        return redirect('/home')
    return render_template('Edit_post.html', blog=blog)


def clean_up_sentence(sentence):
    # tokenize the pattern - split words into array
    sentence_words = nltk.word_tokenize(sentence)
    # stem each word - create short form for word
    sentence_words = [lemmatizer.lemmatize(
        word.lower()) for word in sentence_words]
    return sentence_words

# return bag of words array: 0 or 1 for each word in the bag that exists in the sentence


def bow(sentence, words, show_details=True):
    # tokenize the pattern
    sentence_words = clean_up_sentence(sentence)
    # bag of words - matrix of N words, vocabulary matrix
    bag = [0]*len(words)
    for s in sentence_words:
        for i, w in enumerate(words):
            if w == s:
                # assign 1 if current word is in the vocabulary position
                bag[i] = 1
                if show_details:
                    print("found in bag: %s" % w)
    return (np.array(bag))


def predict_class(sentence, model):
    # filter out predictions below a threshold
    p = bow(sentence, words, show_details=False)
    res = model.predict(np.array([p]))[0]
    ERROR_THRESHOLD = 0.25
    results = [[i, r] for i, r in enumerate(res) if r > ERROR_THRESHOLD]
    # sort by strength of probability
    results.sort(key=lambda x: x[1], reverse=True)
    return_list = []
    for r in results:
        return_list.append({"intent": classes[r[0]], "probability": str(r[1])})
    return return_list


def getResponse(ints, intents_json):
    tag = ints[0]['intent']
    list_of_intents = intents_json['intents']
    for i in list_of_intents:
        if (i['tag'] == tag):
            result = random.choice(i['responses'])
            break
    return result


def chatbot_response(msg):
    ints = predict_class(msg, model)
    res = getResponse(ints, intents)
    return res


@app.route("/get")
def get_bot_response():
    userText = request.args.get('msg')
    return chatbot_response(userText)


if __name__ == '__main__':
    app.run(debug=True)
