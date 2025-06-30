# 🇲🇰 Macedonian Sentiment Analyzer (Rule-Based)

This project implements a **rule-based sentiment analysis system** for the Macedonian language using curated lexicons and handcrafted linguistic rules. It is designed for low-resource settings, leveraging linguistic knowledge rather than large annotated datasets.

> Based on sentiment lexicons and modifier rules adapted from the works of Jovanoski et al. (2015) and Jahić & Vičič (2023).

---

## 📁 Project Structure
- `Initial Translated Lexicon/`
  - `MK_POSITIVE.txt`
  - `MK_NEGATIVE.txt`
  - `MK_AnAwords_intensifiers.txt` – Intensifiers
  - `MK_AnAwords_diminishers.txt` – Diminishers
  - `МК_AnAwords_polarityShifters.txt` – Polarity shifters
  - `MK_AnAwords_stopwords.txt` – Macedonian stopwords
- `Merged Lexicon/`
  - `MK_POSITIVE_merged.txt`
  - `MK_NEGATIVE_merged.txt`
- `sentimentAnalyzer.py` – Main rule-based sentiment classifier  
- `test_data.mo` – Annotated Macedonian Twitter dataset (Exact one from the Jovanoski et al paper) 
- `README.md` 

## 🚀 What This Project Does

- ✅ Tokenizes and lemmatizes Macedonian tweets using `classla`
- ✅ Removes stopwords and normalizes repeated characters
- ✅ Scores each token using:
  - Positive and negative lexicons
  - Intensifiers and diminishers (e.g., doubling or halving impact)
  - Polarity shifters (negation window-based reversal)
- ✅ Computes normalized sentiment score per tweet
- ✅ Classifies tweets into **positive**, **negative**, or **neutral**
- ✅ Evaluates performance across multiple `ε` threshold values

---

## 📦 Requirements

Install dependencies:

```bash
pip install classla scikit-learn
```
Also download the Macedonian model for classla:

```python
import classla
classla.download('mk')
```
🔧 Running the Analyzer
Run the sentiment analyzer with:
```bash
python sentimentAnalyzer.py
```
