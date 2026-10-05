# OLX Cars Content-Based Recommendation System

[![Python](https://img.shields.io/badge/Python-3.13%2B-blue.svg)](https://www.python.org/)
[![scikit-learn](https://img.shields.io/badge/scikit-learn-NLP-orange.svg)](https://scikit-learn.org/)
[![Streamlit](https://img.shields.io/badge/Streamlit-App-red.svg)](https://streamlit.io/)

This repository uses the OLX cars dataset to build a content-based recommendation system based on car features and descriptions using TF-IDF and cosine similarity[cite: 18].

---

## Project Workflow
1. **Data Preprocessing**: Loading dataset (`OLX_cars_dataset00.csv`)[cite: 18] and filling missing values in feature, description, make, and model columns[cite: 18].
2. **Metadata Merging (Soup Creation)**: Combining make, model, car name, fuel type, transmission, and car features into a single descriptive soup string[cite: 18].
3. **Text Vectorization**: Transforming soup texts into numerical feature matrices using `TfidfVectorizer`[cite: 18].
4. **Similarity Computation**: Calculating cosine similarity between all entries using `linear_kernel`[cite: 18].
5. **Recommendation Engine**: Mapping car names to index values and retrieving top similar car listings with details like price and year[cite: 18].
6. **Web Application**: Interactive deployment interface built with Streamlit.

---

## Getting Started & Installation

1. Clone the repository:
   ```bash
   git clone [https://github.com/YOUR_USERNAME/olx-cars-recommendation.git](https://github.com/YOUR_USERNAME/olx-cars-recommendation.git)
   cd olx-cars-recommendation
