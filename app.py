from flask import Flask, render_template, request, redirect
import os
import numpy as numpy
import image_caption_bot as caption_bot

app = Flask(__name__)
 
@app.route('/')
def home():
    return render_template('index.html')

@app.route('/caption', methods=['POST'])
def submit():
    files = os.listdir('./static/')
    for f in files:
        if f != 'style.css':
            os.remove('./static/' + f)
            
    if request.method == 'POST':
        f = request.files['image']
        path = './static/{}'.format(f.filename)
        f.save(path)

        caption = caption_bot.predict_this_image(path)
        data = {
            'img_path': path,
            'caption': caption
        }
        return render_template('index.html', data=data)

if __name__ == '__main__':
    app.run(debug=True)
