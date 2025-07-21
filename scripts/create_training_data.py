import spacy
from spacy.tokens import DocBin, Span
from spacy.util import get_words_and_spaces

def create_training_data():
    nlp = spacy.blank("am")
    db = DocBin()
    
    # Sample 1 with entities - added spaces to control tokenization
    text1 = "በ አዲስ አበባ ቦሌ ላይ እንጀራ ማሽን በ 5000 ብር ይሸጣል"
    words = text1.split()
    spaces = [True] * (len(words) - 1) + [False]
    doc1 = spacy.tokens.Doc(nlp.vocab, words=words, spaces=spaces)
    
    # Create spans with exact token indices
    ents = [
        Span(doc1, 1, 3, "LOC"),      # አዲስ አበባ
        Span(doc1, 3, 4, "LOC"),       # ቦሌ
        Span(doc1, 5, 7, "PRODUCT"),   # እንጀራ ማሽን
        Span(doc1, 8, 10, "PRICE")     # 5000 ብር
    ]
    doc1.ents = ents
    db.add(doc1)
    
    # Sample 2 with entities
    text2 = "ለልጆች እግር ኳስ በ 250 ብር በ መስቀል ላይ"
    words = text2.split()
    spaces = [True] * (len(words) - 1) + [False]
    doc2 = spacy.tokens.Doc(nlp.vocab, words=words, spaces=spaces)
    
    ents = [
        Span(doc2, 0, 1, "PRODUCT"),   # ለልጆች
        Span(doc2, 1, 3, "PRODUCT"),   # እግር ኳስ
        Span(doc2, 4, 6, "PRICE"),      # 250 ብር
        Span(doc2, 7, 8, "LOC")        # መስቀል
    ]
    doc2.ents = ents
    db.add(doc2)
    
    # Save to disk
    db.to_disk("data/labeled/train.spacy")
    print("✅ Created training data with entities")
    
    # Print tokens for verification
    print("\nTokens for text1:")
    for token in doc1:
        print(f"'{token.text}'", end=' ')
    
    print("\n\nTokens for text2:")
    for token in doc2:
        print(f"'{token.text}'", end=' ')

if __name__ == "__main__":
    create_training_data()