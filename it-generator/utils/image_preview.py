'''
pip install Flask
export FLASK_APP=image_preview.py
cd /path/to/images/generated_images
sudo cp utils/image_preview.py /path/to/images/generated_images
flask run --host=0.0.0.0
'''
import os
from io import BytesIO
from glob import glob

from flask import Flask, Response, request, abort, render_template_string, send_from_directory
from PIL import Image

app = Flask(__name__)

WIDTH = 640
HEIGHT = 640

TEMPLATE = '''
<!DOCTYPE html>
<html>
    <head>
        <title></title>
        <meta charset="utf-8"/>
        <style>
        # body {
        #     margin: 0;
        #     background-color: #333;
        # }
        .res {
          position: relative;
          width: 280px;
          display: inline-block;
        }
        .image {
            display: inline-block;
            position: relative,
            margin: 2em auto;
            background-color: #444;
            box-shadow: 0 0 10px rgba(0,0,0,0.3);
        }
        img {
            display: inline-block;
        }
        </style>
    </head>
    <body>
        {% for image in images %}
        <div class="res">
        <a class="image" href="{{ image.src }}" style="width: {{ image.width }}px; height: {{ image.height }}px">
                <img src="{{ image.src }}?w={{ image.width }}&amp;h={{ image.height }}" width="{{ image.width }}" height="{{ image.height }}" />
            </a>
            <p width: 256px;>{{ image.src}}</p>
        </div>

        {% endfor %}
    </body>
'''


@app.route('/<path:filename>')
def image(filename):
    try:
        w = int(request.args['w'])
        h = int(request.args['h'])
    except (KeyError, ValueError):
        return send_from_directory('.', filename)

    try:
        im = Image.open(filename)
        im.thumbnail((w, h), Image.ANTIALIAS)
        io = BytesIO()
        im.save(io, format='JPEG')
        return Response(io.getvalue(), mimetype='image/jpeg')

    except IOError:
        abort(404)

    return send_from_directory('.', filename)


@app.route('/')
def index():
    images = []
    example_dir = './model0'
    filenames = [y for x in os.walk(example_dir)
                     for y in glob(os.path.join(x[0], '*.png'))]
    filenames_dict = {}
    for f in filenames:
        idx = f.find('/img')
        filenames_dict[int(f[idx:idx + f[idx:].find('.')][4:])] = f[idx:]

    gt_imgs_dir_name = 'imgs'
    gt_filenames_dict = {}
    dirs = [
        gt_imgs_dir_name,
        'model0',
        'model1',
        'model2',
    ]
    all_filenames = []
    for dir in dirs:
        if dir == gt_imgs_dir_name:
            all_filenames.append([y for x in os.walk(dir) for y in glob(os.path.join(x[0], '*.jpg'))])
        else:
            all_filenames.append([y for x in os.walk(dir) for y in glob(os.path.join(x[0], '*.png'))])

    vis_filenames = []
    for i in range(1, 500):  # zlen(filenames)+1):
        if i not in filenames_dict:
            continue
        flag = True
        for dir, filenames in zip(dirs, all_filenames):
            if dir + filenames_dict[i] not in filenames:
                flag = False
        if flag:
            for dir in dirs:
                if dir == gt_imgs_dir_name:
                    vis_filenames.append(dir + gt_filenames_dict[i])
                else:
                    vis_filenames.append(dir + filenames_dict[i])

    for filename in vis_filenames:
        im = Image.open(filename)
        w, h = im.size
        aspect = 1.0 * w / h
        if aspect > 1.0 * WIDTH / HEIGHT:
            width = min(w, WIDTH)
            height = width / aspect
        else:
            height = min(h, HEIGHT)
            width = height * aspect
        images.append({
            'width': int(width),
            'height': int(height),
            'src': filename
        })

    return render_template_string(TEMPLATE, **{
        'images': images
    })


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)