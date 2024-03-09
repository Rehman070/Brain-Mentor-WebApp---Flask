import datetime
import random
from flask import Flask, render_template, request, jsonify, redirect, flash, url_for,abort, session
from flask_login import LoginManager, login_user, UserMixin, logout_user
from flask_sqlalchemy import SQLAlchemy
from openai import ChatCompletion
import folium
import numpy as np
from dateutil import parser
from sqlalchemy import LargeBinary
import tensorflow as tf
import openai
from authlib.integrations.flask_client import OAuth
from flask_mail import Mail, Message
import json
import requests
from folium import plugins


app = Flask(__name__)

# Database work
app.app_context().push()
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///brain_mentor.db'
app.config['SECRET_KEY'] = 'a0455de1e15d46ad995c0d40928916ef'
db = SQLAlchemy(app)
login_manager = LoginManager(app)
login_manager.login_view = 'index'

# Set your OpenAI GPT-3 API key here
openai.api_key = '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'

# Provided intents data
intents_data = {"intents": [
        {"tag": "greeting",
         "patterns": ["Hi there", "How are you", "Is anyone there?","Hey","Hola", "Hello", "Good day"],
         "responses": ["Hello, thanks for asking", "Good to see you again", "Hi there, how can I help?"],
         "context": [""]
        },
        {"tag": "goodbye",
         "patterns": ["Bye", "See you later", "Goodbye", "Nice chatting to you, bye", "Till next time"],
         "responses": ["See you!", "Have a nice day", "Bye! Come back again soon."],
         "context": [""]
        },
        {"tag": "thanks",
         "patterns": ["Thanks", "Thank you", "That's helpful", "Awesome, thanks", "Thanks for helping me"],
         "responses": ["Happy to help!", "Any time!", "My pleasure"],
         "context": [""]
        },
        {"tag": "noanswer",
         "patterns": [],
         "responses": ["Sorry, can't understand you", "Please give me more info", "Not sure I understand"],
         "context": [""]
        },

        {"tag": "options",
         "patterns": ["How you could help me?"],
         "responses": ["Hello! I'm here to assist you with information related to brain tumors, treatments and medical professionals"],
         "context": [""]
        },
        {"tag": "options",
         "patterns": ["What are the common symptoms of a brain tumor?"],
         "responses": ["Common symptoms include headaches, seizures, changes in vision, and cognitive issues."],
         "context": [""]
        },

         {"tag": "options",
         "patterns": ["Can you explain the different types of brain tumors?"],
         "responses": ["Brain tumors can be primary & They are classified by location and cell type."],
         "context": [""]
        },

           {"tag": "options",
         "patterns": ["What are the available treatment options for brain tumors?"],
         "responses": ["Treatment may include surgery, radiation therapy, chemotherapy, and targeted therapy."],
         "context": [""]
        },

        {"tag": "doctors",
         "patterns": [" How can I find a neurosurgeon near me?","Can you provide any motivational resources related to tumor?"],
         "responses": ["You can find a neurosurgeon on doctor section","Sure! you can check our Blog section to read blogs and posts"],
         "context": ["doctors"]
        },

           {"tag": "tumors",
         "patterns": ["What are the four stages of brain tumor?"],
         "responses": ["Stage I: Low Grade (Benign) Stage II: Low to Intermediate Grade Stage III: High Grade (Malignant) Stage IV: High Grade (Malignant)"],
         "context": ["tumors"]
        },

        
           {"tag": "tumors",
         "patterns": ["Can you explain the difference between benign and malignant brain tumors?"],
         "responses": ["Benign tumors are non-cancerous and usually less aggressive, while malignant tumors are cancerous and can spread."],
         "context": ["tumors"]
        },

        
           {"tag": "tumors",
         "patterns": ["What should I do if I suspect a brain tumor?"],
         "responses": ["Seek medical attention immediately. Early detection is crucial for treatment."],
         "context": ["tumors"]
        },
        {"tag": "appointment",
         "patterns": ["What information do I need to provide when booking a medical appointment?"],
         "responses": [" Typically, you'll need your personal information, insurance details, and a description of your medical concern."],
         "context": ["appointment"]
        },

         {"tag": "appointment",
         "patterns": ["Can I request a specific date and time for my appointment?"],
         "responses": ["Yes, you can request a preferred date and time, and we'll check for availability."],
         "context": ["appointment"]
        },

         {"tag": "appointment",
         "patterns": ["Can I have a video consultation with a brain tumor specialist?"],
         "responses": ["Yes, we can help you schedule a video consultation with a qualified specialist."],
         "context": ["appointment"]
        }
   ]
}

# Function to handle known intents
def handle_known_intent(intent, patterns):
    return random.choice(intent["responses"])

# Function to generate a response using ChatGPT
def generate_response_with_chatgpt(messages):
    # Make the ChatGPT API call
    response = ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=messages,
    )

    # Extract and return the content of the model-generated response
    return response['choices'][0]['message']['content'].strip()

# Main function to handle user queries
def chatbot(user_query):
    user_query = user_query.lower()

    # Check if the user's query matches predefined patterns
    for intent_data in intents_data["intents"]:
        for pattern in intent_data["patterns"]:
            if pattern.lower() in user_query:
                return handle_known_intent(intent_data, intent_data["patterns"])

    # If no match found, generate a response using ChatGPT
    messages = [{'role': 'system', 'content': 'You are a helpful assistant.'}]

    if user_query:
        messages.append({'role': 'user', 'content': user_query})

    return generate_response_with_chatgpt(messages)

@app.route('/get_response', methods=['POST'])
def get_response():
    user_query = request.form['user_query']
    response = chatbot(user_query)
    return jsonify({'response': response})


# Create a Model of User
class Patients(UserMixin, db.Model):
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(80), unique=True, nullable=False)
    email = db.Column(db.String(100), nullable=False, unique=True)
    password = db.Column(db.String(100), nullable=False)

    def __repr__(self):
        return '<User %r>' % self.username

class Blogs(db.Model):
    blog_id = db.Column(db.Integer, primary_key=True)
    blog_type = db.Column(db.String(80), nullable=False)
    tit_name = db.Column(db.String(80), nullable=False)
    author = db.Column(db.String(40), nullable=False)
    content = db.Column(db.Text(), nullable=False)
    image_data = db.Column(LargeBinary, nullable=False)  # Add image field
    pub_date = db.Column(db.DateTime(), nullable=False,
                         default=datetime.datetime.utcnow)

    def __repr__(self):
        return '<Blogs %r>' % self.tit_name

# User Loader Function
@login_manager.user_loader
def load_user(user_id):
    return Patients.query.get(int(user_id))

# User Signup/Login Function
@app.route('/', methods=["GET", "POST"])
def index():
    if request.method == 'POST':
        username = request.form.get('username')
        email = request.form.get('email')
        password = request.form.get('password')
        action = request.form.get('action')

        existing_patient = Patients.query.filter_by(username=username).first()
        existing_doctor = web_user.query.filter_by(email=email).first()

        if action == "signup":
            if not existing_patient:
                # It's a patient signup attempt
                new_patient = Patients(username=username,
                                       email=email, password=password)
                db.session.add(new_patient)
                db.session.commit()
                flash(' Account has been successfully created!', 'success')
                return redirect('/home')
            else:
                flash(' Account already exists. Please log in.', 'warning')
        elif action == "login":
            if existing_patient:
                # It's a patient login attempt
                if existing_patient.password == password and email == existing_patient.email:
                    login_user(existing_patient)
                    flash(' Login successful!', 'success')
                    return redirect('/home')
                else:
                    flash(' Invalid credentials. Please try again.', 'danger')
            elif existing_doctor:
                # It's a doctor login attempt
                if existing_doctor.password == password and email == existing_doctor.email and existing_doctor.name == username:
                    flash(' Doctor login successful!', 'success')
                    return render_template('doctor_dashboard.html', doctor_id=existing_doctor.doctor_id)
                else:
                    flash(' Invalid doctor credentials. Please try again.', 'danger')
            else:
                flash(' No account found. Please sign up first.', 'warning')

    return render_template('index.html')

#Logout User
@app.route('/logout')
def logout():
    logout_user()
    session.pop("user", None)
    return redirect('/')

import base64
from flask import render_template


# Sign in with Google
@app.route('/home')
def Home():
    blogs = Blogs.query.all()

    # Convert images to Base64
    for blog in blogs:
        if blog.image_data:
            blog.base64_image = base64.b64encode(blog.image_data).decode('utf-8')
        else:
            blog.base64_image = None

    return render_template('Home.html', blogs=blogs, session=session.get("user"), pretty=json.dumps(session.get("user"), indent=4))


appConf = {
    "OAUTH2_CLIENT_ID": "552872142229-vufan07358j75rr54k9l2u4ibrmpt8dn.apps.googleusercontent.com",
    "OAUTH2_CLIENT_SECRET": "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",
    "OAUTH2_META_URL": "https://accounts.google.com/.well-known/openid-configuration",
    "FLASK_SECRET": "ALongRandomlyGeneratedString",
    "FLASK_PORT": 5000
}

app.secret_key = appConf.get("FLASK_SECRET")

# Flask-Mail configuration
app.config['MAIL_SERVER'] = 'smtp.gmail.com'
app.config['MAIL_PORT'] = 587
app.config['MAIL_USE_TLS'] = True
app.config['MAIL_USE_SSL'] = False
app.config['MAIL_USERNAME'] = 'noreplay'
app.config['MAIL_PASSWORD'] = '12345'  # Use App Password if 2-Step Verification is enabled
app.config['MAIL_DEFAULT_SENDER'] = 'noreplay'

mail = Mail(app)

oauth = OAuth(app)
oauth.register(
    "myApp",
    client_id=appConf.get("OAUTH2_CLIENT_ID"),
    client_secret=appConf.get("OAUTH2_CLIENT_SECRET"),
    client_kwargs={
        "scope": "openid profile email https://www.googleapis.com/auth/user.birthday.read https://www.googleapis.com/auth/user.gender.read",
    },
    server_metadata_url=f'{appConf.get("OAUTH2_META_URL")}',
)

@app.route("/signin-google")
def googleCallback():
    # fetch access token and id token using authorization code
    token = oauth.myApp.authorize_access_token()

    # google people API
    personDataUrl = "https://people.googleapis.com/v1/people/me?personFields=genders,birthdays,emailAddresses"
    personDataResponse = requests.get(personDataUrl, headers={
        "Authorization": f"Bearer {token['access_token']}"
    })

    if personDataResponse.status_code == 200:
        personData = personDataResponse.json()
        token["personData"] = personData

        # Send welcome email to the user
        email = personData.get('emailAddresses', [{}])[0].get('value', None)
        if email:
            send_welcome_email(email)

        # set complete user information in the session
        session["user"] = token
        flash(' Successfully logged in with Google!', 'success')  # Flash a success message
        return redirect(url_for("Home"))
    else:
        print(f"Error fetching Google People API data: {personDataResponse.text}")
        abort(500)  # You can handle the error in a more appropriate way

def send_welcome_email(email):
    msg = Message("BrainMentor: Your Path to Brain Health and Wellness",
                  recipients=[email])
    msg.body = "Welcome to BrainMentor! 🧠🌐 Our advanced web app uses Deep Learning and Image Processing to detect brain tumors swiftly. Beyond detection, doctor appointments, and a chatbot for valuable insights. Explore video consultations, physician finders, and more. Your journey to a healthier future starts here!"
    mail.send(msg)

@app.route("/google-login")
def googleLogin():
    if "user" in session:
        abort(404)
    return oauth.myApp.authorize_redirect(redirect_uri=url_for("googleCallback", _external=True))


@app.route('/patient_signup')
def Patient_Signup():
    return render_template('Patient_Signup.html')


#Brain Tumor Model Prediction
from flask import Flask, render_template, request
import tensorflow as tf
from PIL import Image
import numpy as np
from flask import Flask, render_template, request, url_for

# Load the pre-trained model
model_path = 'Brain Tumor Detection\RealCNN.h5'
model = tf.keras.models.load_model(model_path)

# Define class labels
class_labels = ['glioma', 'meningioma', 'notumor', 'pituitary']

# Function to preprocess the image
def preprocess_image(image_path):
    img = Image.open(image_path)
    img = img.resize((224, 224))
    img_array = tf.keras.preprocessing.image.img_to_array(img)
    img_array = tf.expand_dims(img_array, 0)
    return img_array

def predict_image(image_path):
    img_array = preprocess_image(image_path)
    predictions = model.predict(img_array)
    predicted_class = class_labels[np.argmax(predictions)]
    confidence = np.max(predictions)
    return predicted_class, confidence
@app.route('/detection')
def Detection():
    return render_template('Detection.html')

# Prediction
@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return render_template('Detection.html', message='No file part')

    file = request.files['file']

    if file.filename == '':
        return render_template('Detection.html', message='No selected file')

    try:
        image_path = f'static/tmp/{file.filename}'
        file.save(image_path)
        predicted_class, confidence = predict_image(image_path)

        # Use url_for to get the correct URL for the image
        uploaded_image = url_for('static', filename=f'tmp/{file.filename}')

        return render_template('Detection.html', prediction=predicted_class, confidence=confidence, uploaded_image=uploaded_image)

    except Exception as e:
        return render_template('Detection.html', message=f'Error: {str(e)}')



@app.route('/appointment')
def Appointment():
    return render_template('Appointment.html')


@app.route('/patient_treat')
def Patient_Treat():
    return render_template('Patient_Treat.html')

#Doctor Location & Markers
@app.route('/doctors_marker')
def Doctors_Markers():
    # Coordinates for the center of Islamabad, Pakistan
    pakistan_isb_coordinates = [33.700266560649226, 73.05169660795143]

    # Create a Folium map
    map = folium.Map(
        location=pakistan_isb_coordinates,
        tiles=None,
        zoom_start=13,  # Adjust the zoom level as needed
    )

    # Create a FeatureGroup layer to contain markers
    marker_layer = folium.FeatureGroup(name='Markers').add_to(map)

    # Add tile layers
    folium.TileLayer('CartodbPositron').add_to(map)
    folium.TileLayer('https://{s}.basemaps.cartocdn.com/dark_all/{z}/{x}/{y}{r}.png', name='CartoDBDarkMatter', attr="CartoDBDarkMatter").add_to(map)
    folium.TileLayer('https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}', name='Esri', attr="Esri").add_to(map)
    folium.TileLayer('OpenStreetMap').add_to(map)

    # Add layers control over the map
    folium.LayerControl().add_to(map)

    # Add search functionality to the FeatureGroup layer
    search = plugins.Search(layer=marker_layer, geom_type='Marker', search_zoom=15).add_to(map)

    # Define markers as in your original code
    # Add a marker to the map
    folium.Marker(
        location=pakistan_isb_coordinates,
        popup='Islamabad, Pakistan',
        tooltip='Islamabad, Pakistan',
        icon=folium.Icon(color='red', icon='location-dot', prefix='fa'),
        title='Islamabad'  # Add the location name as the title
    ).add_to(marker_layer)

    # Add markers for hospitals and doctors
    markers = [
        {"location": [34.223299962163615, 72.263259854991], "name": "Brain & Spine Clinic", "type": "doctor", "popup_content": """
    <div>
        <h4 style="color:blue;">Prof. Dr Inayat Ullah Khan</h4>
        <img src="https://media.istockphoto.com/id/177373093/photo/indian-male-doctor.jpg?s=612x612&w=0&k=20&c=5FkfKdCYERkAg65cQtdqeO_D0JMv6vrEdPw3mX1Lkfg=" alt="Image" style="width:100px;height:100px;">
        <p>Address: Clinic Address, City, Country</p>
        <p>Email: clinic@example.com</p>
        <p>Open Hours: Monday to Friday, 9:00 AM - 5:00 PM</p>
    </div>
    """},
        {"location": [33.71081684674506, 73.02100506625507], "name": "Dr. Alamgir Neuro Surgeon", "type": "doctor","popup_content": """
    <div>
        <h4 style="color:blue;">Prof. Dr Inayat Ullah Khan</h4>
        <img src="https://media.istockphoto.com/id/177373093/photo/indian-male-doctor.jpg?s=612x612&w=0&k=20&c=5FkfKdCYERkAg65cQtdqeO_D0JMv6vrEdPw3mX1Lkfg=" alt="Image" style="width:100px;height:100px;">
        <p>Address: Clinic Address, City, Country</p>
        <p>Email: clinic@example.com</p>
        <p>Open Hours: Monday to Friday, 9:00 AM - 5:00 PM</p>
    </div>
    """},
        {"location": [33.694266983374575, 72.99629625361848], "name": "Dr Akbar Khan", "type": "doctor","popup_content": """
    <div>
        <h4 style="color:blue;">Prof. Dr Inayat Ullah Khan</h4>
        <img src="https://media.istockphoto.com/id/177373093/photo/indian-male-doctor.jpg?s=612x612&w=0&k=20&c=5FkfKdCYERkAg65cQtdqeO_D0JMv6vrEdPw3mX1Lkfg=" alt="Image" style="width:100px;height:100px;">
        <p>Address: Clinic Address, City, Country</p>
        <p>Email: clinic@example.com</p>
        <p>Open Hours: Monday to Friday, 9:00 AM - 5:00 PM</p>
    </div>
    """},
        {"location": [33.67596469823713, 73.06711812392459], "name": "Prof. Dr Inayat Ullah Khan", "type": "doctor","popup_content": """
    <div>
        <h4 style="color:blue;">Prof. Dr Inayat Ullah Khan</h4>
        <img src="https://media.istockphoto.com/id/177373093/photo/indian-male-doctor.jpg?s=612x612&w=0&k=20&c=5FkfKdCYERkAg65cQtdqeO_D0JMv6vrEdPw3mX1Lkfg=" alt="Image" style="width:100px;height:100px;">
        <p>Address: Clinic Address, City, Country</p>
        <p>Email: clinic@example.com</p>
        <p>Open Hours: Monday to Friday, 9:00 AM - 5:00 PM</p>
    </div>
    """},
        {"location": [33.6844, 73.0479], "name": "Hotel ABC", "type": "hotel","popup_content": """
    <div>
        <h4 style="color:blue;">Prof. Dr Inayat Ullah Khan</h4>
        <img src="https://media.istockphoto.com/id/177373093/photo/indian-male-doctor.jpg?s=612x612&w=0&k=20&c=5FkfKdCYERkAg65cQtdqeO_D0JMv6vrEdPw3mX1Lkfg=" alt="Image" style="width:100px;height:100px;">
        <p>Address: Clinic Address, City, Country</p>
        <p>Email: clinic@example.com</p>
        <p>Open Hours: Monday to Friday, 9:00 AM - 5:00 PM</p>
    </div>
    """},
        {"location": [33.6804, 73.0471], "name": "City Hospital", "type": "hospital","popup_content": """
    <div>
        <h4 style="color:blue;">Prof. Dr Inayat Ullah Khan</h4>
        <img src="https://media.istockphoto.com/id/177373093/photo/indian-male-doctor.jpg?s=612x612&w=0&k=20&c=5FkfKdCYERkAg65cQtdqeO_D0JMv6vrEdPw3mX1Lkfg=" alt="Image" style="width:100px;height:100px;">
        <p>Address: Clinic Address, City, Country</p>
        <p>Email: clinic@example.com</p>
        <p>Open Hours: Monday to Friday, 9:00 AM - 5:00 PM</p>
    </div>
    """},
      
        {"location": [33.6883, 73.0495], "name": "National Museum", "type": "museum","popup_content": """
    <div>
        <h4 style="color:blue;">Prof. Dr Inayat Ullah Khan</h4>
        <img src="https://media.istockphoto.com/id/177373093/photo/indian-male-doctor.jpg?s=612x612&w=0&k=20&c=5FkfKdCYERkAg65cQtdqeO_D0JMv6vrEdPw3mX1Lkfg=" alt="Image" style="width:100px;height:100px;">
        <p>Address: Clinic Address, City, Country</p>
        <p>Email: clinic@example.com</p>
        <p>Open Hours: Monday to Friday, 9:00 AM - 5:00 PM</p>
    </div>
    """},
    {"location": [33.95126074394443, 71.43232422698537], "name": "Nawab Market", "type": "Market","popup_content": """
    <div>
        <h4 style="color:blue;">Prof. Dr Inayat Ullah Khan</h4>
        <img src="https://media.istockphoto.com/id/177373093/photo/indian-male-doctor.jpg?s=612x612&w=0&k=20&c=5FkfKdCYERkAg65cQtdqeO_D0JMv6vrEdPw3mX1Lkfg=" alt="Image" style="width:100px;height:100px;">
        <p>Address: Clinic Address, City, Country</p>
        <p>Email: clinic@example.com</p>
        <p>Open Hours: Monday to Friday, 9:00 AM - 5:00 PM</p>
    </div>
    """},
    ]

    for marker_info in markers:
        icon = 'user-doctor' if marker_info["type"] == "doctor" else 'hospital' if marker_info["type"] == "hospital" else 'university'
        
        # Add a marker to the map
        marker = folium.Marker(
            location=marker_info["location"],
            popup=folium.Popup(html=f"<b>{marker_info['name']}</b>{marker_info.get('popup_content', '')}", max_width=300),
            icon=folium.Icon(color='blue' if marker_info["type"] == "hospital" else 'green', icon=icon, prefix='fa'),
            title=marker_info['name']  # Add the location name as the title
        ).add_to(marker_layer)

    return map._repr_html_()

from flask import request, redirect, render_template, flash
from werkzeug.utils import secure_filename
import os
import datetime

UPLOAD_FOLDER = 'static/uploads'
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif'}

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Function to check if the file extension is allowed
def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/blog', methods=["GET", "POST"])
def Blog():
    if request.method == 'POST':
        type = request.form.get('type')
        title = request.form.get('title')
        author = request.form.get('author')
        content = request.form.get('content')
        
        # Check if the post request has the file part
        if 'image' not in request.files:
            flash(' No file part', 'error')
            return redirect(request.url)
        
        file = request.files['image']
        
        # If the user does not select a file, the browser submits an empty file without a filename
        if file.filename == '':
            flash( 'No selected file', 'error')
            return redirect(request.url)
        
        # Check if the file extension is allowed
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(file_path)
            
            # Read image file as binary data
            with open(file_path, 'rb') as f:
                image_data = f.read()
            
            # Debug statement to print the first few characters of image_data
            print("First few characters of image_data:", image_data[:20])  # Print the first 20 characters
            
            # Create the blog post with the image binary data
            blog = Blogs(blog_type=type, tit_name=title,
                         author=author, content=content, image_data=image_data)
            db.session.add(blog)
            db.session.commit()
            flash(' Blog Sucessfully Created', 'success')
            return redirect('/home')
        else:
            flash(' Invalid file format', 'error')
            return redirect(request.url)
    
    return render_template('Blogs.html')


# Routes
@app.route("/blogs_detail/<int:id>")
def blogs_detail(id):
    blog = Blogs.query.get(id)

    if blog.image_data:
        blog.base64_image = base64.b64encode(blog.image_data).decode('utf-8')
    else:
        blog.base64_image = None
    return render_template('View_Blog.html', blog=blog)

@app.route("/delete/<int:id>")
def del_post(id):
    blog = Blogs.query.get(id)
    db.session.delete(blog)
    db.session.commit()
    flash(' Your Post Has Been Successfully Deleted', 'success')
    return redirect('/home')

@app.route("/edit/<int:id>", methods=['GET', 'POST'])
def edit_post(id):
    blog = Blogs.query.get(id)
    if request.method == 'POST':
        blog.blog_type = request.form.get('type')  # Update the field name to match your model
        blog.tit_name = request.form.get('title')
        blog.author = request.form.get('author')
        blog.content = request.form.get('content')  # Update the field name to match your model
        if 'image' in request.files:
            file = request.files['image']
            if file.filename != '':
                if allowed_file(file.filename):
                    filename = secure_filename(file.filename)
                    file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
                    file.save(file_path)
                    with open(file_path, 'rb') as f:
                        image_data = f.read()
                    blog.image_data = image_data
                else:
                    flash('Invalid file format', 'error')
                    return redirect(request.url)
        db.session.commit()
        flash(' Your Post Has Been Successfully Update', 'success')
        return redirect('/home')
    return render_template('edit_post.html', blog=blog)



# Fetching doctors appointments informaton  from the database and passing json
@app.route('/get_appointments', methods=['POST'])
def get_appointments():
    doctor_id = request.json['doctor_id']
    appointments = patient_appointments.query.filter_by(doctor_id=doctor_id).all()
    appointments_list = []
    for ap in appointments:
        appointment_data = {
            'appointment_id': ap.appointment_id,
            'doctor_id': ap.doctor_id,
            'name': ap.name,
            'email': ap.email,
            'number': ap.number,
            'date': ap.date,
            'timeslot': ap.timeslot,
            'consultation':ap.consultation
        }
        appointments_list.append(appointment_data)

    return jsonify(appointments=appointments_list)


# Adding doctors info into the database
@app.route('/add_doctor', methods=["POST"])
def add_doctor():
    # Create a new doctor
    doctor_data = [
    {
        "name": "Dr. Aisha Khan",
        "designation": "Neurologist",
        "age": 42,
        "country": "Pakistan",
        "email": "aisha@example.com",
        "password": "neuro123"
    },
    {
        "name": "Dr. Hassan Ali",
        "designation": "Cardiologist",
        "age": 39,
        "country": "Pakistan",
        "email": "hassan@example.com",
        "password": "cardiodoc456"
    },
    {
        "name": "Dr. Fatima Ahmed",
        "designation": "Dermatologist",
        "age": 37,
        "country": "Pakistan",
        "email": "fatima@example.com",
        "password": "skinexpert789"
    },
    {
        "name": "Dr. Usman Khan",
        "designation": "Gastroenterologist",
        "age": 45,
        "country": "Pakistan",
        "email": "usman@example.com",
        "password": "gastro456"
    },
    {
        "name": "Dr. Sana Malik",
        "designation": "Pediatrician",
        "age": 41,
        "country": "Pakistan",
        "email": "sana@example.com",
        "password": "pediatrics789"
    },
    {
        "name": "Dr. Ali Raza",
        "designation": "Orthopedic Surgeon",
        "age": 48,
        "country": "Pakistan",
        "email": "ali@example.com",
        "password": "ortho123"
    },
    {
        "name": "Dr. Zainab Ahmed",
        "designation": "Ophthalmologist",
        "age": 43,
        "country": "Pakistan",
        "email": "zainab@example.com",
        "password": "eyeexpert456"
    },
    {
        "name": "Dr. Bilal Khan",
        "designation": "Psychiatrist",
        "age": 40,
        "country": "Pakistan",
        "email": "bilal@example.com",
        "password": "minddoc123"
    },
    {
        "name": "Dr. Sara Ali",
        "designation": "Obstetrician/Gynecologist",
        "age": 36,
        "country": "Pakistan",
        "email": "sara@example.com",
        "password": "obgyn456"
    }
]

    for obj in doctor_data:
        new_doctor = web_user(name=obj['name'], designation=obj['designation'], age=obj['age'], country=obj['country'], email=obj['email'], password=obj['password'])

        # Add the new doctor to the database session
        db.session.add(new_doctor)

        # Commit the session to save changes to the database
        db.session.commit()

    return "New doctor added successfully!"


# Fetching all doctors from database for doctor route and showing there info in cards
@app.route('/get_doctors')
def get_doctors():
    all_doctors = web_user.query.all()
    doctors_list = [{'doctor_id': doctor.doctor_id,'name': doctor.name, 'designation': doctor.designation, 'age': doctor.age, 'country': doctor.country} for doctor in all_doctors]

    return jsonify(doctors=doctors_list)

# Rendering doctors page
@app.route('/doctors')
def Doctors():
    all_doctors = web_user.query.all()
    return render_template('Doctors.html', doctors=all_doctors)


# Creating appointment for the patient
@app.route('/make_appointment', methods=["POST"])
def make_appointment():
    # Create a new doctor
    doctor_id = request.form['doctor_id']
    name = request.form['name']
    email = request.form['email']
    number =request.form['number']
    # Format date
    date = parser.parse(request.form['date']).strftime('%d/%m/%Y')

    # Format time slot
    time_slot = request.form['timeslot']

    # Map time slot values to appropriate format
    time_slot_mapping = {
        "9am-12pm": "9 AM to 12 PM",
        "1pm-4pm": "1 PM to 4 PM"
    }

    # Check if the time slot value is valid
    if time_slot in time_slot_mapping:
        time_slot = time_slot_mapping[time_slot]
    else:
        return "Invalid time slot"


    consultation = request.form['consultation']



    new_appointment = patient_appointments(name=name, doctor_id=doctor_id, email=email, number=number, date=date, timeslot=time_slot, consultation=consultation)

    # Add the new doctor to the database session
    db.session.add(new_appointment)

    # Commit the session to save changes to the database
    db.session.commit()

    return redirect(url_for("Appointment"))


# Define  database model for the doctor
class Doctor(db.Model):
    doctor_id = db.Column(db.Integer, primary_key=True, autoincrement=True, nullable=False)
    name = db.Column(db.String(80), nullable=False)
    designation = db.Column(db.String(120), nullable=False)
    age = db.Column(db.Integer, nullable=False)
    country = db.Column(db.String(80), nullable=False)
    email = db.Column(db.String(80), nullable=False)
    password = db.Column(db.String(80), nullable=False)

    def __repr__(self):
        return '<Doctors %r>' % self.name

# Define  database model for the doctor
class web_user(db.Model):
    doctor_id = db.Column(db.Integer, primary_key=True, autoincrement=True, nullable=False)
    name = db.Column(db.String(80), nullable=False)
    designation = db.Column(db.String(120), nullable=False)
    age = db.Column(db.Integer, nullable=False)
    country = db.Column(db.String(80), nullable=False)
    email = db.Column(db.String(80), nullable=False)
    password = db.Column(db.String(80), nullable=False)

    def __repr__(self):
        return '<web_user %r>' % self.name

# Define  database model for the appointments
class appointment(db.Model):
    appointment_id = db.Column(db.Integer, primary_key=True, autoincrement=True)
    doctor_id = db.Column(db.Integer, nullable=False)
    name = db.Column(db.String(120), nullable=False)
    email = db.Column(db.String(120), nullable=False)
    number = db.Column(db.String(50), nullable=False)
    date = db.Column(db.String(90), nullable=False)
    timeslot = db.Column(db.String(90), nullable=False)
    consultation = db.Column(db.String(190), nullable=False)


    def __repr__(self):
        return '<appointment %r>' % self.name

class patient_appointments(db.Model):
    appointment_id = db.Column(db.Integer, primary_key=True, autoincrement=True)
    doctor_id = db.Column(db.Integer, nullable=False)
    name = db.Column(db.String(120), nullable=False)
    email = db.Column(db.String(120), nullable=False)
    number = db.Column(db.String(50), nullable=False)
    date = db.Column(db.String(90), nullable=False)
    timeslot = db.Column(db.String(90), nullable=False)
    consultation = db.Column(db.String(190), nullable=False)


    def __repr__(self):
        return '<appointment %r>' % self.name

if __name__ == '__main__':
    # creating new tables if not previously has been made
    db.create_all()

    app.run(debug=True,host='0.0.0.0', port=appConf.get("FLASK_PORT"))
