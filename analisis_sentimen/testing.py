import joblib
import re
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory

# Load model and vectorizer with error handling
try:
    model = joblib.load('logistic_model.joblib')
    vectorizer = joblib.load('tfidf_vectorizer.joblib')
    print("[INFO] Model and vectorizer loaded successfully")
    print(f"[DEBUG] Model classes: {model.classes_}")
except Exception as e:
    print(f"[ERROR] Failed to load model: {str(e)}")
    exit()

# Initialize stemmer
factory = StemmerFactory()
stemmer = factory.create_stemmer()

def preprocess(text):
    """Consistent text preprocessing"""
    text = text.lower()
    text = re.sub(r'[^\w\s]', '', text)  # Remove punctuation
    text = re.sub(r'\d+', '', text)       # Remove numbers
    return stemmer.stem(text)

def predict(text):
    """Robust prediction function with full error handling"""
    try:
        # Preprocess input
        cleaned_text = preprocess(text)
        print(f"[DEBUG] Cleaned Text: {cleaned_text}")
        
        # Vectorize
        text_vector = vectorizer.transform([cleaned_text])
        print(f"[DEBUG] Vector Shape: {text_vector.shape}")
        
        # Get prediction
        predicted_class = model.predict(text_vector)[0]
        proba = model.predict_proba(text_vector)[0]
        
        # Handle class labels properly
        if hasattr(model, 'classes_'):
            class_mapping = {
                0: "Negatif",
                1: "Positif"
            }
            # Ensure we're mapping correctly
            if len(model.classes_) > 1:
                class_mapping = {
                    model.classes_[0]: "Negatif",
                    model.classes_[1]: "Positif"
                }
        else:
            class_mapping = {0: "Negatif", 1: "Positif"}
        
        # Get confidence scores
        confidence_pos = round(proba[1] * 100, 2)
        confidence_neg = round(proba[0] * 100, 2)
        
        return {
            "text": text,
            "cleaned_text": cleaned_text,
            "prediction": class_mapping.get(predicted_class, "Unknown"),
            "confidence": max(confidence_pos, confidence_neg),
            "confidence_positif": confidence_pos,
            "confidence_negatif": confidence_neg,
            "raw_prediction": int(predicted_class),
            "probabilities": {
                "negatif": float(proba[0]),
                "positif": float(proba[1])
            }
        }
        
    except Exception as e:
        return {
            "error": str(e),
            "input_text": text,
            "prediction": "Error",
            "confidence": 0
        }

if __name__ == "__main__":
    print("\n=== APLIKASI ANALISIS SENTIMEN ===")
    print(f"[INFO] Model loaded with {len(vectorizer.get_feature_names_out())} features")
    
    test_cases = [
        "aplikasinya bagus",
        "aplikasinya jelek, banyak bug dan error",
        "user interface sangat membantu"
    ]
    
    for text in test_cases:
        print("\n" + "="*50)
        print(f"Memproses: '{text}'")
        result = predict(text)
        
        if 'error' in result:
            print(f"[ERROR] {result['error']}")
        else:
            print(f"\nHasil: {result['prediction']} (Confidence: {result['confidence']}%)")
            print(f"Detail: Positif={result['confidence_positif']}%, Negatif={result['confidence_negatif']}%")
            print(f"\nDebug Info:")
            print(f"Cleaned: {result['cleaned_text']}")
            print(f"Raw Pred: {result['raw_prediction']}")
            print(f"Probs: {result['probabilities']}")
        
    print("\n" + "="*50)
    print("Masukkan teks secara manual (ketik 'exit' untuk keluar)")
    
    while True:
        user_input = input("\nInput teks: ").strip()
        if user_input.lower() == 'exit':
            break
            
        result = predict(user_input)
        
        if 'error' in result:
            print(f"[ERROR] {result['error']}")
        else:
            print(f"\nHasil: {result['prediction']}")
            print(f"Keyakinan: {result['confidence']}%")
            print(f"Detail: Positif={result['confidence_positif']}%, Negatif={result['confidence_negatif']}%")