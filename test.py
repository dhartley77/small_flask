from flask import Flask, render_template, request, session,send_from_directory
import os
import numpy as np
import pickle
from werkzeug.utils import secure_filename
#import torch
import subprocess
from torchvision.models import inception_v3

# WSGI Application
# Defining upload folder path
UPLOAD_FOLDER = os.path.join('static')
# Define allowed files
ALLOWED_EXTENSIONS = {'pt'}

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.secret_key = 'super secret'


@app.route('/')
def home():
    return render_template('home2.html')

@app.route('/',  methods=("POST", "GET"))
def uploadFile():
    file_uploaded = False
    if request.method == 'POST':
        # Upload file flask
        uploaded_mdl = request.files['uploaded-file']
        if uploaded_mdl:
            # Extracting uploaded data file name
            mdl_filename = secure_filename(uploaded_mdl.filename)
            # Upload file to database (defined uploaded folder in static path)
            uploaded_mdl.save(os.path.join(app.config['UPLOAD_FOLDER'], mdl_filename))
            # Storing uploaded file path in flask session
            session['uploaded_mdl_file_path'] = os.path.join(app.config['UPLOAD_FOLDER'], mdl_filename)
            file_uploaded = True

        return render_template('home2.html', file_uploaded=file_uploaded)

@app.route('/result')
def Result():
    mdl_file_path = session.get('uploaded_mdl_file_path', None)
    result = os.path.join(app.root_path, mdl_file_path)
    if result.endswith('.pt'):  # Assuming it's a PyTorch model file
        # Load the input model
        #input_model = torch.load(result, map_location=torch.device('cpu'))
        # Load the testing model
        command = [
            "python",
            "entrypoint.py",
            "infer",
            "--model_filepath=result",
            "--result_filepath=output.txt",
            "--scratch_dirpath=scratch/",
            "--examples_dirpath=model/id-00000002/clean-example-data/",
            "--round_training_dataset_dirpath=/this/filepath/does/nothing",
            "--metaparameters_filepath=new_learned_parameters/metaparameters.json",
            "--schema_filepath=metaparameters_schema.json",
            "--learned_parameters_dirpath=new_learned_parameters/",
            "--scale_parameters_filepath=scale_params.npy"
        ]

    result_m = subprocess.run(command, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, encoding="utf-8")
    #, stdout=subprocess.PIPE,
    #breakpoint()
    output = result_m.stdout  # Captured standard output
    error = result_m.stderr  # Captured error output

    # Print the captured output and error
    #print("Output:", output)
    #print("Error:", error)

    #will need to read output.txt file for prediction text
    return render_template('result.html', prediction_text=output)



if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8080, debug=True)
