from flask import Flask, render_template, request, jsonify
import spacy
import speech_recognition as sr
from pydub import AudioSegment
import networkx as nx
import plotly.graph_objects as go
import plotly.utils
import json
from flask import send_from_directory

app = Flask(__name__)

# 🚀 AUTO-FIX spaCy for Render (CRITICAL!)
try:
    nlp = spacy.load("en_core_web_sm")
except OSError:
    print("📥 Downloading English spaCy model...")
    import subprocess
    import sys
    subprocess.check_call([sys.executable, "-m", "spacy", "download", "en_core_web_sm"])
    nlp = spacy.load("en_core_web_sm")
    print("✅ spaCy model ready!")

r = sr.Recognizer()

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/process', methods=['POST'])
def process_speech():
    if 'audio' not in request.files:
        return jsonify({'error': 'No audio file'})
    
    audio_file = request.files['audio']
    audio = AudioSegment.from_file(audio_file)
    audio = audio.set_frame_rate(16000).set_channels(1)
    
    with open('temp.wav', 'wb') as f:
        audio.export(f, format='wav')
    
    with sr.AudioFile('temp.wav') as source:
        audio_data = r.record(source)
        try:
            text = r.recognize_google(audio_data)
        except:
            text = "Speech recognition failed"
    
    # Process with spaCy
    doc = nlp(text)
    
    # Build knowledge graph
    G = nx.DiGraph()
    for ent in doc.ents:
        G.add_node(ent.text, label=ent.label_)
    for token in doc:
        for child in token.children:
            G.add_edge(token.text, child.text)
    
    # Plotly graph
    pos = nx.spring_layout(G)
    edge_x = []
    edge_y = []
    for edge in G.edges():
        x0, y0 = pos[edge[0]]
        x1, y1 = pos[edge[1]]
        edge_x.extend([x0, x1, None])
        edge_y.extend([y0, y1, None])
    
    edge_trace = go.Scatter(x=edge_x, y=edge_y, line=dict(width=1, color='#888'), hoverinfo='none', mode='lines')
    
    node_x = [pos[node][0] for node in G.nodes()]
    node_y = [pos[node][1] for node in G.nodes()]
    node_text = [f"{node}<br>Label: {G.nodes[node]['label']}" for node in G.nodes()]
    
    node_trace = go.Scatter(x=node_x, y=node_y, mode='markers+text', hoverinfo='text', text=node_text,
                           marker=dict(size=20, line_width=2))
    
    fig = go.Figure(data=[edge_trace, node_trace], layout=go.Layout(showlegend=False, hovermode='closest', margin=dict(b=20,l=5,r=5,t=40)))
    
    graph_json = json.dumps(fig, cls=plotly.utils.PlotlyJSONEncoder)
    
    return jsonify({
        'text': text,
        'entities': [(ent.text, ent.label_) for ent in doc.ents],
        'graph': graph_json
    })

@app.route('/uploads/<filename>')
def uploaded_file(filename):
    return send_from_directory('uploads', filename)

if __name__ == '__main__':
    app.run(debug=True)
