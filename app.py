
# import import_ipynb
from flask import Flask, render_template, request, jsonify, send_from_directory
from flask_cors import CORS
import os
import uuid
import threading
from style_transfer import run_style_transfer, vgg19, mean, std

app = Flask(__name__)
CORS(app)

# --- Configuration ---
UPLOAD_FOLDER = 'uploads'
GENERATED_FOLDER = 'generated'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['GENERATED_FOLDER'] = GENERATED_FOLDER

# Create directories if they don't exist
# if dir already exists -> no error due to exist_ok argument
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(GENERATED_FOLDER, exist_ok=True)


tasks = {}  #dict to track task status


def process_style_transfer(task_id, content_path, style_path, output_path, style_weight=1000000):
    #this function runs in a bg thread
    try:
        run_style_transfer(vgg19, mean, std, content_path, style_path, output_path, num_steps=500, style_weight=style_weight)
        tasks[task_id]['status'] = 'complete'
    except Exception as e:
        print(f'Error processing task {task_id}: {e}')
        tasks[task_id]['status'] = 'error'

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/generate', methods = ['POST'])
def generate():
    if 'content_image' not in request.files or 'style_image' not in request.files:
        return jsonify({'error': 'Missing image files'}), 400
    
    content_file = request.files['content_image']
    style_file = request.files['style_image']

    # save uploaded files
    content_filename = str(uuid.uuid4()) + os.path.splitext(content_file.filename)[1]
    style_filename = str(uuid.uuid4()) + os.path.splitext(style_file.filename)[1]

    content_path = os.path.join(app.config['UPLOAD_FOLDER'], content_filename)
    style_path = os.path.join(app.config['UPLOAD_FOLDER'], style_filename)
    content_file.save(content_path)
    style_file.save(style_path)

    # get style_weight from form (default to 1000000 if not provided)
    style_weight = request.form.get('style_weight', '1000000')
    try:
        style_weight = int(style_weight)
    except Exception:
        style_weight = 1000000

    # for output
    task_id = str(uuid.uuid4())
    ouput_filename = f'{task_id}.jpg'
    output_path = os.path.join(app.config['GENERATED_FOLDER'], ouput_filename)

    tasks[task_id] = {'status': 'processing', 'output_filename': ouput_filename}

    # start bg thread, pass style_weight
    thread = threading.Thread(target=process_style_transfer, args=(task_id, content_path, style_path, output_path, style_weight))
    thread.start()

    return jsonify({'task_id': task_id})


@app.route('/status/<task_id>', methods=['GET'])
def get_status(task_id):
    task = tasks.get(task_id)
    if not task:
        return jsonify({'error': 'Task not found'}), 404
    
    response = {'status': task['status']}
    if task['status'] == 'complete':
        response['image_url'] = f'/generated/{task['output_filename']}'

    return jsonify(response)


@app.route('/generated/<filename>')
def get_generated_image(filename):
    return send_from_directory(app.config['GENERATED_FOLDER'], filename)


if __name__ == '__main__':
    app.run(debug=True, threaded=True)




