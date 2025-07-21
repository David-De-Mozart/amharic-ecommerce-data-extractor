from fastapi import FastAPI
from pydantic import BaseModel
import spacy

app = FastAPI()
nlp = spacy.load("models/distilbert")

class TextRequest(BaseModel):
    text: str

@app.post("/extract-entities")
def extract_entities(request: TextRequest):
    doc = nlp(request.text)
    return {
        "entities": [
            {"text": ent.text, "label": ent.label_, "start": ent.start_char, "end": ent.end_char}
            for ent in doc.ents
        ]
    }

@app.get("/")
def demo_ui():
    return """
    <html><body>
    <h1>Amharic Entity Extractor</h1>
    <form action="/extract-entities" method="post">
        <textarea name="text" rows="5" cols="50"></textarea><br>
        <input type="submit" value="Extract Entities">
    </form>
    </body></html>
    """