from flask import Flask, render_template, request, jsonify
import spacy
import speech_recognition as sr
from pydub import AudioSegment
import networkx as nx
import plotly.graph_objects as go
import plotly.utils
import json
import os
from collections import Counter

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
    if 'audio' not in request.files:
        return jsonify({'error': 'No audio file found in request'})
    
    audio_file = request.files['audio']
    if audio_file.filename == '':
        return jsonify({'error': 'No file selected'})

    try:
        temp_input_path = "temp_input.wav"
        audio_file.save(temp_input_path)

        audio = AudioSegment.from_file(temp_input_path)
        audio = audio.set_frame_rate(16000).set_channels(1)
        
        temp_exported_path = 'processed_temp.wav'
        audio.export(temp_exported_path, format='wav')

        with sr.AudioFile(temp_exported_path) as source:
            audio_data = r.record(source)
            try:
                text = r.recognize_google(audio_data)
            except sr.UnknownValueError:
                text = "Could not understand audio"
            except sr.RequestError:
                text = "Speech service down"

        # NLP & Knowledge Graph Logic
        doc = nlp(text)
        G = nx.DiGraph()
        
        if not doc.ents:
            for token in doc:
                if not token.is_punct:
                    G.add_node(token.text)
        else:
            for ent in doc.ents:
                G.add_node(ent.text, label=ent.label_)

        for token in doc:
            for child in token.children:
                if not token.is_punct and not child.is_punct:
                    G.add_edge(token.text, child.text)

        # --- NEW ANALYTICS LOGIC ---
        total_concepts = G.number_of_nodes()
        total_connections = G.number_of_edges()
        
        # Calculate Main Hub (Node with highest degree)
        if total_concepts > 0:
            degrees = dict(G.degree())
            main_hub = max(degrees, key=degrees.get)
        else:
            main_hub = "None"

        # Calculate Top Keywords
        words = [token.text.lower() for token in doc if not token.is_stop and not token.is_punct]
        top_keywords = [item[0] for item in Counter(words).most_common(5)]
        # ---------------------------

        # Create Plotly Graph
        pos = nx.spring_layout(G)
        
        edge_x = []
        edge_y = []
        for edge in G.edges():
            x0, y0 = pos[edge[0]]
            x1, y1 = pos[edge[1]]
            edge_x.extend([x0, x1, None])
            edge_y.extend([y0, y1, None])

        edge_trace = go.Scatter(
            x=edge_x, y=edge_y,
            line=dict(width=2, color='#888'),
            hoverinfo='none',
            mode='lines'
        )

        node_x = [pos[node][0] for node in G.nodes()]
        node_y = [pos[node][1] for node in G.nodes()]
        
        node_trace = go.Scatter(
            x=node_x, y=node_y,
            mode='markers+text',
            text=[node for node in G.nodes()],
            textposition="top center",
            marker=dict(
                size=25,
                color='mediumpurple',
                line=dict(width=2, color='white')
            )
        )
        
        # UPDATED LAYOUT: Gridlines enabled
        fig = go.Figure(data=[edge_trace, node_trace],
             layout=go.Layout(
                showlegend=False,
                hovermode='closest',
                margin=dict(b=0, l=0, r=0, t=0),
                xaxis=dict(
                    showgrid=True, 
                    gridcolor='rgba(255,255,255,0.1)', 
                    zeroline=False, 
                    showticklabels=False
                ),
                yaxis=dict(
                    showgrid=True, 
                    gridcolor='rgba(255,255,255,0.1)', 
                    zeroline=False, 
                    showticklabels=False
                ),
                plot_bgcolor='rgba(0,0,0,0)',
                paper_bgcolor='rgba(0,0,0,0)'
            )
        )
        
        graph_json = json.dumps(fig, cls=plotly.utils.PlotlyJSONEncoder)

        os.remove(temp_input_path)
        os.remove(temp_exported_path)

        # UPDATED JSON RESPONSE: Including new metrics
        return jsonify({
            'text': text,
            'entities': [(ent.text, ent.label_) for ent in doc.ents],
            'graph': graph_json,
            'total_concepts': total_concepts,
            'total_connections': total_connections,
            'main_hub': main_hub,
            'top_keywords': top_keywords
        })

    except Exception as e:
        return jsonify({'error': str(e)})

if __name__ == '__main__':
    port = int(os.environ.get("PORT", 5000))
    app.run(host='0.0.0.0', port=port)
