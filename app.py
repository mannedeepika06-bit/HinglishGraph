from flask import Flask, render_template, request, jsonify
import spacy
import speech_recognition as sr
from pydub import AudioSegment
import networkx as nx
import plotly.graph_objects as go
import plotly.utils
import json
import os

app = Flask(__name__)

# Load spacy model
try:
    nlp = spacy.load("en_core_web_sm")
except OSError:
    import subprocess
    import sys
    subprocess.check_call([sys.executable, "-m", "spacy", "download", "en_core_web_sm"])
    nlp = spacy.load("en_core_web_sm")

r = sr.Recognizer()

@app.route('/', methods=['GET', 'POST'])
def index():
    return render_template('index.html')

@app.route('/process', methods=['POST'])
def process_speech():
    # 1. Check if file exists in request
    if 'audio' not in request.files:
        return jsonify({'error': 'No audio file found in request'})
    
    audio_file = request.files['audio']
    if audio_file.filename == '':
        return jsonify({'error': 'No file selected'})

    try:
        # 2. SAVE the file properly to disk first
        temp_input_path = "temp_input.wav"
        audio_file.save(temp_input_path)

        # 3. Convert/Format audio using pydub
        audio = AudioSegment.from_file(temp_input_path)
        audio = audio.set_frame_rate(16000).set_channels(1)
        
        temp_exported_path = 'processed_temp.wav'
        audio.export(temp_exported_path, format='wav')

        # 4. Speech Recognition
        with sr.AudioFile(temp_exported_path) as source:
            audio_data = r.record(source)
            try:
                # recognize_google supports Telugu if you add language='te-IN'
                text = r.recognize_google(audio_data)
            except sr.UnknownValueError:
                text = "Could not understand audio"
            except sr.RequestError:
                text = "Speech service down"

        # 5. NLP & Knowledge Graph Logic
        doc = nlp(text)
        G = nx.DiGraph()
        for ent in doc.ents:
            G.add_node(ent.text, label=ent.label_)
        for token in doc:
            for child in token.children:
                G.add_edge(token.text, child.text)

        # 6. Create Plotly Graph
        pos = nx.spring_layout(G)
        node_x = [pos[node][0] for node in G.nodes()]
        node_y = [pos[node][1] for node in G.nodes()]
        
        node_trace = go.Scatter(
            x=node_x, y=node_y, mode='markers+text', 
            text=[node for node in G.nodes()],
            marker=dict(size=20)
        )
        
        fig = go.Figure(data=[node_trace])
        graph_json = json.dumps(fig, cls=plotly.utils.PlotlyJSONEncoder)

        # Clean up files
        os.remove(temp_input_path)
        os.remove(temp_exported_path)

        return jsonify({
            'text': text,
            'entities': [(ent.text, ent.label_) for ent in doc.ents],
            'graph': graph_json
        })

    except Exception as e:
        return jsonify({'error': str(e)})

if __name__ == '__main__':
    port = int(os.environ.get("PORT", 5000))
    app.run(host='0.0.0.0', port=port)
