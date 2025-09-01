# Disaster Tweet Classification using DistilBERT (NLP & Deep Learning)

This project applies **Natural Language Processing (NLP)** and **Deep Learning with Transformers** to classify tweets as **disaster-related** or **not disaster-related**. By fine-tuning a pretrained **DistilBERT** model on labeled Twitter data, the project demonstrates how transfer learning can be used to solve real-world text classification problems.  

---

## 📌 Problem Statement
Disasters generate a massive amount of information on platforms like Twitter. Identifying whether a tweet refers to a real disaster or not is critical for emergency services, governments, and organizations responding to crises.  

This project leverages **state-of-the-art transformer models** to accurately classify tweets, improving both speed and reliability of disaster response systems.  

---

## ⚙️ Tech Stack
- **Programming:** Python  
- **Libraries/Frameworks:** TensorFlow, Keras, Hugging Face Transformers, Pandas, NumPy, Matplotlib, Seaborn  
- **Model:** DistilBERT (transformer-based pretrained model)  
- **Metrics:** Accuracy, F1-Score, Confusion Matrix  

---

## 🔄 Workflow
1. **Data Preprocessing & EDA**  
   - Explored tweet length, token distribution, and class imbalance  
   - Tokenized text, applied truncation/padding, and generated attention masks  

2. **Model Development**  
   - Loaded pretrained **DistilBERT** from Hugging Face  
   - Added classification head for binary classification  
   - Trained with Adam optimizer, cross-entropy loss, and dropout regularization  

3. **Fine-Tuning**  
   - Fine-tuned transformer weights on labeled tweet dataset  
   - Adjusted sequence length (max_length=160) and batch size (32)  

4. **Evaluation**  
   - Compared baseline vs. transformer models  
   - Evaluated with F1-score, accuracy, and confusion matrix  

5. **Results**  
   - Achieved **XX% accuracy** and **YY% F1-score** (replace with your actual numbers)  
   - Demonstrated clear improvement of transformer-based models over classical baselines  

---

## 📊 Results & Insights
- Transformer-based models significantly outperform traditional NLP approaches in detecting disaster-related tweets  
- **F1-score** is a more reliable metric than accuracy due to class imbalance  
- Pretrained language representations (DistilBERT) transfer well to domain-specific text  

---

## 🚀 How to Run
```bash
# Clone the repo
git clone https://github.com/yourusername/disaster-tweet-classification.git
cd disaster-tweet-classification

# Install dependencies
pip

