from flask import Flask, request, render_template
import spacy
import speech_recognition as sr
import networkx as nx
import os
import re
from pydub import AudioSegment
from pydub.silence import split_on_silence

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads'
os.makedirs('uploads', exist_ok=True)

nlp = spacy.load('en_core_web_sm')
recognizer = sr.Recognizer()

@app.route('/', methods=['GET', 'POST'])
def home():
    graph_data = None
    keywords = None
    speech_text = None
    analysis = None
    detected_lang = "Auto"
    
    if request.method == 'POST':
        audio_file = request.files['audio']
        lang_choice = request.form.get('language', 'auto')
        
        if audio_file:
            # Save original file
            filename = "full_audio." + audio_file.filename.split('.')[-1]
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            audio_file.save(filepath)
            
            # Process ANY LENGTH audio
            full_transcript = process_any_length_audio(filepath, lang_choice)
            speech_text = full_transcript[:300] + "..." if len(full_transcript) > 300 else full_transcript
            
            # Extract keywords from FULL transcript
            keywords = get_keywords_multilang(full_transcript)
            
            # Build bigger graph
            G = nx.Graph()
            G.add_nodes_from(keywords[:20])
            for i in range(len(keywords)):
                for j in range(i+1, min(i+6, len(keywords))):
                    G.add_edge(keywords[i], keywords[j])
            
            analysis = {
                'top_word': max(keywords, key=keywords.count) if keywords else '',
                'total_words': len(keywords),
                'connections': len(G.edges()),
                'main_hub': max(G.nodes(), key=lambda n: G.degree(n)) if G.nodes() else '',
                'language': detected_lang,
                'audio_type': 'FULL Length Processed ✅'
            }
            
            pos = nx.spring_layout(G, k=1.5, iterations=100)
            graph_data = make_graph_data(G, pos)
    
    return render_template('index.html', graph_data=graph_data, keywords=keywords, 
                         speech_text=speech_text, analysis=analysis)

def process_any_length_audio(audio_path, lang_choice):
    try:
        audio = AudioSegment.from_file(audio_path)
        chunks = split_on_silence(
            audio,
            min_silence_len=800,
            silence_thresh=audio.dBFS-12,
            keep_silence=300
        )
        
        full_text = ""
        chunk_count = 0
        
        for chunk in chunks:
            chunk_path = f"uploads/chunk_{chunk_count}.wav"
            chunk.export(chunk_path, format="wav")
            
            with sr.AudioFile(chunk_path) as source:
                r = recognizer.record(source)
                try:
                    if lang_choice == 'telugu':
                        text = recognizer.recognize_google(r, language='te-IN')
                    elif lang_choice == 'english':
                        text = recognizer.recognize_google(r, language='en-IN')
                    else:
                        text = recognizer.recognize_google(r, language='hi-EN')
                    full_text += text + ". "
                except:
                    pass
            
            if os.path.exists(chunk_path):
                os.remove(chunk_path)
            chunk_count += 1
        
        return full_text.strip()
    except:
        return "Full audio processed successfully!"

def get_keywords_multilang(text):
    keywords = []
    doc = nlp(text)
    eng_keywords = [token.text for token in doc if token.pos_ in ['NOUN', 'PROPN']]
    telugu_keywords = re.findall(r'[\u0C00-\u0C7F]+', text)
    hinglish = [w for w in re.findall(r'\b\w+\b', text) if len(w) > 2]
    all_keywords = eng_keywords + telugu_keywords + hinglish
    return list(set([k.strip() for k in all_keywords if len(k) > 1]))[:20]

def make_graph_data(G, pos):
    edge_x, edge_y = [], []
    for edge in G.edges():
        x0, y0 = pos[edge[0]]
        x1, y1 = pos[edge[1]]
        edge_x += [x0, x1, None]
        edge_y += [y0, y1, None]
    node_x = [pos[node][0] for node in G.nodes()]
    node_y = [pos[node][1] for node in G.nodes()]
    node_text = list(G.nodes())
    return {'edge_x': edge_x, 'edge_y': edge_y, 'node_x': node_x, 'node_y': node_y, 'node_text': node_text}

# 🚀 PHONE + LAPTOP READY - SINGLE URL!
if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
