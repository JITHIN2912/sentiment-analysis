import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, classification_report

# Page configuration
st.set_page_config(
    page_title="Sentiment Analysis",
    layout="centered",
    page_icon="💬"
)

# Load and display image
st.image("sa.webp", use_container_width=True)

# Custom CSS for dark theme
st.markdown("""
    <style>
    .stApp {
        background-color: #121212;
        color: #e0e0e0;
    }
    .stButton > button {
        background-color: #1de9b6;
        color: #000;
        border-radius: 8px;
        font-weight: bold;
    }
    .stTextArea textarea {
        background-color: #2c2c2c;
        color: #f1f1f1;
        border-radius: 8px;
    }
    h1, h2, h3 {
        color: #1de9b6;
    }
    .bottom-right {
        position: fixed;
        bottom: 10px;
        right: 15px;
        background-color: #263238;
        padding: 10px 16px;
        border-radius: 10px;
        font-size: 13px;
        color: #80cbc4;
        box-shadow: 0 2px 6px rgba(0,0,0,0.5);
    }
    </style>
""", unsafe_allow_html=True)

# App title
st.markdown("<h1 style='text-align: center;'>💬 Sentiment Analysis </h1>", unsafe_allow_html=True)

# File uploader
st.markdown("### 📥 Upload your dataset")
uploaded_file = st.file_uploader("Upload a CSV file with 'selected_text' and 'sentiment' columns", type=["csv"])

# Main logic
if uploaded_file is not None:
    try:
        df = pd.read_csv(uploaded_file, on_bad_lines='skip', encoding='utf-8')

        if not ('selected_text' in df.columns and 'sentiment' in df.columns):
            st.error("❌ CSV must contain 'selected_text' and 'sentiment' columns.")
        else:
            df = df.dropna(subset=['selected_text', 'sentiment'])

            # Encode labels
            label_enc = LabelEncoder()
            df['label'] = label_enc.fit_transform(df['sentiment'])

            # Split data
            X_train, X_test, y_train, y_test = train_test_split(
                df['selected_text'], df['label'], test_size=0.2, random_state=42, stratify=df['label']
            )

            # TF-IDF
            tfidf = TfidfVectorizer(stop_words='english', max_features=5000)
            X_train_tfidf = tfidf.fit_transform(X_train)
            X_test_tfidf = tfidf.transform(X_test)

            # Naive Bayes model
            model = MultinomialNB()
            model.fit(X_train_tfidf, y_train)
            y_pred = model.predict(X_test_tfidf)

            # Metrics
            acc = accuracy_score(y_test, y_pred)
            precision, recall, f1, _ = precision_recall_fscore_support(y_test, y_pred, average='weighted')

            # Classification report
            with st.expander("📋 Full Classification Report"):
                st.text(classification_report(y_test, y_pred, target_names=label_enc.classes_))

            # User input for prediction
            st.markdown("### 🧠 Try It Out")
            user_text = st.text_area("Enter text for sentiment prediction:")

            if st.button("🔮 Predict"):
                if user_text.strip() == "":
                    st.warning("⚠️ Please enter some text.")
                else:
                    vec = tfidf.transform([user_text])
                    pred = model.predict(vec)[0]
                    sentiment = label_enc.inverse_transform([pred])[0]
                    st.success(f"✅ Predicted Sentiment: **{sentiment}**")

            # Accuracy block in bottom right
            st.markdown(
                f"""
                <div class="bottom-right">
                    <b>Model Evaluation:</b><br>
                    Accuracy: {acc:.4f}<br>
                    Precision: {precision:.4f}<br>
                    Recall: {recall:.4f}<br>
                    F1 Score: {f1:.4f}
                </div>
                """,
                unsafe_allow_html=True
            )

    except Exception as e:
        st.error(f"❌ Error reading file or training model: {e}")
else:
    st.info("📂 Please upload a CSV file to continue.")
