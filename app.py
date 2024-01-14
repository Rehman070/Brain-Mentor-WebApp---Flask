import datetime
import folium
from flask import Flask, render_template, request, redirect, flash
from flask_login import LoginManager, login_user, UserMixin, logout_user
from werkzeug.security import generate_password_hash, check_password_hash
from flask_sqlalchemy import SQLAlchemy
import numpy as np
from flask import Flask, render_template, request, jsonify
from openai import ChatCompletion
import random
import openai

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

# Chatbot Code
# Provided intents data
intents_data = {"intents": [
    {"tag": "greeting",
     "patterns": ["Hi there", "How are you", "Is anyone there?", "Hey", "Hola", "Hello", "Good day"],
     "responses": ["Hello, thanks for asking", "Good to see you again", "Hi there, how can I help?"],
     "context": [""]},
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
     "patterns": [" How can I find a neurosurgeon near me?", "Can you provide any motivational resources related to tumor?"],
     "responses": ["You can find a neurosurgeon on doctor section",
                    "Sure! you can check our Blog section to read blogs and posts"],
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
    # ... other intents
]}

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
    # Creating a list of messages for ChatGPT, including the user's query
    messages = [{'role': 'system', 'content': 'You are a helpful assistant.'}]

    if user_query:
        messages.append({'role': 'user', 'content': user_query})

    # Call the new function to generate a response using ChatGPT
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

if __name__ == '__main__':
    app.run(debug=True)
