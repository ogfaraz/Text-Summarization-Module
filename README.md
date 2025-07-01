# Text-Summarization-Module
Core Module in my FInal Year Project: a sequence-to-sequence (Seq2Seq) encoder-decoder model using LSTM layers for abstractive text summarization


---


# 🧠 Seq2Seq Text Summarization Module

This repository contains the implementation of a Sequence-to-Sequence (Seq2Seq) Encoder-Decoder model for **abstractive text summarization**. It is developed using Bi-directional LSTM for the encoder and unidirectional LSTM for the decoder, with GloVe word embeddings for semantic representation.

> 📘 This model is part of the **Final Year Project (FYP)** titled **"Sense Stream: Emotion-Driven Media Analysis and Speech Generation"**, which integrates emotion detection, toxicity classification, summarization, and expressive TTS.

---

## 📌 Overview

This module is focused on summarizing long-form text documents (e.g., news articles, transcripts) into concise, human-readable summaries using deep learning-based natural language generation.

### Key Features:
- Preprocessing of text data with tokenization and padding
- Integration of pretrained **GloVe embeddings**
- **Bidirectional LSTM encoder** for improved context retention
- **LSTM decoder** with teacher forcing
- Trained on the **CNN/DailyMail** dataset
- Evaluation using Rogue scores

---

## 📁 Files

- `seq2seq-enc-dec.ipynb`: Jupyter notebook containing the full pipeline including preprocessing, model architecture, training, and evaluation.
- `glove.6B.100d.txt`: GloVe embedding file (download separately from [GloVe](https://nlp.stanford.edu/projects/glove/)).

---

## 🧠 Model Architecture


Input Sequence → Embedding Layer → Bidirectional LSTM (Encoder)
                                         ↓
                        Decoder Input → LSTM (Decoder) → Dense (Softmax)


---

## ⚙️ Dependencies

* Python 3.x
* TensorFlow / Keras
* NumPy
* NLTK
* Scikit-learn
* tqdm

Install dependencies with:

```bash
pip install -r requirements.txt
```

---

## 📊 Dataset

The model is trained on the [CNN/DailyMail dataset](https://huggingface.co/datasets/cnn_dailymail), which contains thousands of news articles paired with human-written summaries.

---

## 🚀 How to Run

1. Download and unzip GloVe vectors (100d).
2. Update paths in the notebook if necessary.
3. Open and execute `seq2seq-enc-dec.ipynb` in Jupyter or Google Colab.

---


## 🔍 Applications

* Educational content summarization
* News summarization engines
* Assistive tech for low-literacy users
* Video/media moderation (as part of **Sense Stream**)

---

## 👩‍💻 Author

* **Faraz Ahmad Khan**
* Final Year Project - 2025
* Department of Computer Science
* *Bahria University*

---

## 📜 License

This project is licensed under the MIT License.

---

## ⭐ Acknowledgements

* [Stanford GloVe Embeddings](https://nlp.stanford.edu/projects/glove/)
* [CNN/DailyMail Dataset](https://huggingface.co/datasets/cnn_dailymail)
* TensorFlow & Keras documentation

```
