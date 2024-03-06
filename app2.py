# from flask import Flask, render_template, jsonify
#
# app = Flask(__name__)
#
# @app.route("/")
# def hello_career():
#   return render_template("home.html")
#
#
# if __name__ == "__main__":
#   app.run(host='0.0.0.0', debug=True)

#
# from flask import Flask, render_template, redirect, url_for
# from flask_wtf import FlaskForm
# from wtforms import FileField, SubmitField
# from werkzeug.utils import secure_filename
# import os
# from wtforms.validators import InputRequired
# import subprocess
#
# app = Flask(__name__)
# app.config['SECRET_KEY'] = 'supersecretkey'
# app.config['UPLOAD_FOLDER'] = 'static/files'
#
# class UploadFileForm(FlaskForm):
#     file = FileField("File", validators=[InputRequired()])
#     submit = SubmitField("Upload File")
#
# @app.route('/', methods=['GET',"POST"])
# @app.route('/home', methods=['GET',"POST"])
# def home():
#     form = UploadFileForm()
#     if form.validate_on_submit():
#         file = form.file.data # First grab the file
#         file_path = os.path.join(os.path.abspath(os.path.dirname(__file__)),app.config['UPLOAD_FOLDER'],secure_filename(file.filename))
#         file.save(file_path) # Then save the file
#         # Call main2.py with the path to the uploaded file as an argument
#         subprocess.Popen(['python', 'main2.py', file_path])
#         return "File has been uploaded."
#     return render_template('index.html', form=form)
#
# if __name__ == '__main__':
#     app.run(debug=True)







#
# from flask import Flask, render_template, redirect, url_for
# from flask_wtf import FlaskForm
# from wtforms import FileField, SubmitField
# from werkzeug.utils import secure_filename
# import os
# from wtforms.validators import InputRequired
# import subprocess
#
# app = Flask(__name__)
# app.config['SECRET_KEY'] = 'supersecretkey'
# app.config['UPLOAD_FOLDER'] = 'static/files'
#
# class UploadFileForm(FlaskForm):
#     file = FileField("File", validators=[InputRequired()])
#     submit = SubmitField("Upload File")
#
# @app.route('/', methods=['GET',"POST"])
# @app.route('/home', methods=['GET',"POST"])
# def home():
#     form = UploadFileForm()
#     if form.validate_on_submit():
#         file = form.file.data # First grab the file
#         file_path = os.path.join(os.path.abspath(os.path.dirname(__file__)),app.config['UPLOAD_FOLDER'],secure_filename(file.filename))
#         file.save(file_path) # Then save the file
#         # Call main.py with the path to the uploaded file as an argument
#         subprocess.Popen(['python', 'main.py', file_path])
#         return "File has been uploaded."
#     return render_template('index.html', form=form)
#
# if __name__ == '__main__':
#     app.run(debug=True)


from flask import Flask, render_template, redirect, url_for
from flask_wtf import FlaskForm
from wtforms import FileField, SubmitField
from werkzeug.utils import secure_filename
import os
from wtforms.validators import InputRequired
import subprocess
import main

app = Flask(__name__)
app.config['SECRET_KEY'] = 'supersecretkey'
app.config['UPLOAD_FOLDER'] = 'static/files'


class UploadFileForm(FlaskForm):
    file = FileField("File", validators=[InputRequired()])
    submit = SubmitField("Summarize")


@app.route('/', methods=['GET', "POST"])
@app.route('/home', methods=['GET', "POST"])
def home():
    form = UploadFileForm()
    if form.validate_on_submit():
        file = form.file.data  # First grab the file
        file_path = os.path.join(os.path.abspath(os.path.dirname(__file__)), app.config['UPLOAD_FOLDER'],
                                 secure_filename(file.filename))
        file.save(file_path)  # Then save the file

        # Fetch the summary from the second program
        # summary = get_summary(file_path)
        extracted_data = main.extract_text_from_pdf(file_path)
        print(extracted_data)
        # Redirect to a page where the summary will be displayed
        return render_template('summary.html', extracted_data=extracted_data)
    return render_template('index.html', form=form)

if __name__ == '__main__':
    app.run(debug=True)
