from flask import Flask, render_template, request, send_file
from gtts import gTTS
import os
import subprocess

app = Flask(__name__)

# Convert text to speech
def text_to_speech(text, audio_path):
    tts = gTTS(text=text, lang='en')
    tts.save(audio_path)

# Lip-sync the image to the generated audio using Wav2Lip script
def lip_sync(image_path, audio_path, output_path):
    # Ensure the Wav2Lip directory is the correct path
    wav2lip_script = 'Wav2Lip/infer.py'
    model_path = 'models/wav2lip_gan.pth'
    
    command = [
        'python', wav2lip_script,
        '--checkpoint_path', model_path,
        '--face', image_path,
        '--audio', audio_path,
        '--output', output_path
    ]
    
    subprocess.run(command, check=True)

# Route for the home page
@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        # Get the text input from the user
        text = request.form['text_input']
        
        # Paths
        image_path = 'static/preview_target.webp'
        audio_path = 'static/input_audio.mp3'
        output_video_path = 'static/output_video.mp4'

        # Convert text to speech
        text_to_speech(text, audio_path)
        
        # Lip-sync the image
        lip_sync(image_path, audio_path, output_video_path)
        
        return render_template('avtar.html', video_generated=True, video_path=output_video_path)
    
    return render_template('avtar.html', video_generated=False)

# Serve the video
@app.route('/video')
def serve_video():
    return send_file('static/preview_target.webp', as_attachment=False)

if __name__ == '__main__':
    app.run(debug=True)
