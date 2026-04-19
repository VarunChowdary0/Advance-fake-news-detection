# Explainable Fake News Detection Using BiLSTM-Attention with LIME and SHAP Analysis

---

## Abstract

The proliferation of fake news has emerged as a significant threat to the integrity of information dissemination, particularly on social media platforms. Traditional deep learning approaches for fake news detection often function as black-box models, lacking transparency in their decision-making process. In this paper, we propose a comprehensive approach that integrates Bidirectional Long Short-Term Memory (BiLSTM) networks with attention mechanisms for fake news detection, enhanced with multiple Explainable AI (XAI) techniques including LIME, SHAP, and attention visualization. Our model utilizes FastText word embeddings to capture semantic and morphological information from text. We evaluate our approach on a balanced dataset of fake and real news articles, achieving competitive classification performance. Additionally, we implement bias detection probes to ensure model reliability. The explainability components provide human-interpretable insights into the model's predictions, fostering trust and accountability in automated fake news detection systems.

**Keywords:** Fake News Detection, Explainable AI, BiLSTM, Attention Mechanism, LIME, SHAP, Deep Learning, Natural Language Processing, FastText Embeddings

---

## I. Introduction

The rise of social media platforms has dramatically increased the volume of information sharing, enabling rapid dissemination of both legitimate news and misinformation. Facebook, Twitter, YouTube, and other platforms serve as primary sources of news for millions of users globally [1]. However, this democratization of information has also facilitated the spread of fake news—incorrect or deceptive information that purports to be newsworthy [2].

The spread of unverified false information can have serious repercussions, including harming the credibility of news ecosystems, damaging reputations, inciting public panic, and undermining societal stability [3]. Fake news can influence public opinion on various subjects, divert attention from critical issues, and has been linked to conflicts in many countries [4].

Traditional approaches to fake news detection often rely solely on content-based features, employing machine learning classifiers on linguistic patterns. While effective to some extent, these methods face limitations in capturing complex contextual relationships within text [5]. Deep learning approaches, particularly recurrent neural networks, have shown promise in addressing these limitations by automatically learning hierarchical feature representations [6].

However, neural network models are inherently black-box in nature, presenting significant challenges regarding interpretability, accountability, transparency, and trust in the models' decision-making processes [7]. This opacity is particularly problematic in fake news detection, where understanding why a piece of content is classified as fake is crucial for validation, bias detection, and user trust.

Explainable AI (XAI) techniques have emerged to address these challenges, providing clear insights into model predictions and enhancing understanding among stakeholders [8]. Various XAI approaches, including Local Interpretable Model-agnostic Explanations (LIME) and SHapley Additive exPlanations (SHAP), have been successfully applied to explain predictions in text classification tasks [9].

In this paper, we propose a comprehensive fake news detection system that combines the powerful sequence modeling capabilities of BiLSTM networks with attention mechanisms, enhanced by multiple explainability techniques. Our contributions are as follows:

- We develop a BiLSTM-Attention model with FastText embeddings for fake news classification that captures both forward and backward contextual dependencies
- We implement three complementary XAI techniques (Attention Visualization, LIME, and SHAP) to provide multi-perspective explanations of model predictions
- We introduce bias detection probes to identify potential model biases toward political names or sensational content
- We provide comprehensive evaluation including classification metrics, calibration analysis, and error analysis

The rest of the paper is structured as follows: Section II reviews related work on fake news detection and explainable AI. Section III describes our proposed methodology. Section IV details the dataset and preprocessing. Section V presents experimental results. Section VI discusses the explainability analysis. Section VII concludes the paper with future directions.

---

## II. Related Work

### A. Traditional Machine Learning Approaches

Traditional machine learning models such as Logistic Regression, Support Vector Machines (SVM), Decision Trees, Random Forest, and Naive Bayes have been widely used for fake news detection [10]. These approaches typically rely on handcrafted features including linguistic patterns, sentiment scores, and stylometric features [11].

SVM, Naive Bayes, and passive-aggressive classifiers have been employed with Term Frequency-Inverse Document Frequency (TF-IDF) features, achieving accuracies around 95% [12]. However, these methods face limitations in feature engineering, scalability, handling imbalanced data, and contextual understanding [13].

### B. Deep Learning Approaches

Neural network models have demonstrated superior capability in extracting complex features compared to traditional ML approaches [14]. Convolutional Neural Networks (CNNs) have been applied to fake news detection based on discriminatory content characteristics, achieving accuracy of 98.36% [15].

Recurrent Neural Networks (RNNs), particularly Long Short-Term Memory (LSTM) networks, have shown effectiveness in capturing sequential dependencies in text [16]. Bidirectional LSTMs (BiLSTMs) extend this capability by processing sequences in both forward and backward directions, capturing richer contextual information [17].

The attention mechanism, introduced by Bahdanau et al. [18], allows models to focus on relevant parts of the input when making predictions. This mechanism has been successfully integrated with RNN architectures for various NLP tasks, providing both improved performance and inherent interpretability through attention weights [19].

### C. Transformer-Based Models

Transformer-based models like BERT, RoBERTa, and XLNet have achieved state-of-the-art performance in various NLP tasks including fake news detection [20]. These models leverage self-attention mechanisms and pre-training on large corpora to capture deep contextual representations [21].

However, transformer models are computationally expensive and require significant resources for fine-tuning [22]. Additionally, their complex architecture makes interpretation challenging despite the attention mechanism.

### D. Explainable AI in Fake News Detection

The black-box nature of deep learning models has motivated the application of XAI techniques to fake news detection. LIME has been used to explain BERT model predictions by highlighting influential features [23]. SHAP has been applied to reveal key features in LSTM and BERT predictions [24].

Attention visualization provides an intrinsic form of explainability by showing which input tokens the model focuses on during prediction [25]. However, the faithfulness of attention weights as explanations remains debated in the literature [26].

Our work differs from existing approaches by combining multiple complementary XAI techniques to provide comprehensive explanations, along with bias detection mechanisms to ensure model reliability.

---

## III. Proposed Methodology

### A. System Architecture Overview

Our proposed system follows an end-to-end pipeline for fake news detection with integrated explainability. Figure 1 illustrates the overall architecture.

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                         SYSTEM ARCHITECTURE                                  │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  ┌──────────┐    ┌──────────────┐    ┌─────────────┐    ┌───────────────┐  │
│  │  Input   │───▶│ Preprocessing │───▶│  FastText   │───▶│    BiLSTM     │  │
│  │   Text   │    │   Pipeline    │    │  Embedding  │    │    Layer      │  │
│  └──────────┘    └──────────────┘    └─────────────┘    └───────┬───────┘  │
│                                                                  │          │
│                                                                  ▼          │
│  ┌──────────────────────────────────────────────────────────────────────┐  │
│  │                        Attention Layer                                │  │
│  │              (Bahdanau-style Attention Mechanism)                     │  │
│  └───────────────────────────────┬──────────────────────────────────────┘  │
│                                  │                                          │
│                                  ▼                                          │
│  ┌──────────────┐    ┌──────────────┐    ┌──────────────────────────────┐  │
│  │    Dense     │───▶│   Dropout    │───▶│      Output Layer            │  │
│  │    Layer     │    │    (0.5)     │    │   (Sigmoid - Binary)         │  │
│  └──────────────┘    └──────────────┘    └──────────────┬───────────────┘  │
│                                                          │                  │
│                                                          ▼                  │
│  ┌──────────────────────────────────────────────────────────────────────┐  │
│  │                    EXPLAINABILITY MODULE                              │  │
│  │  ┌────────────────┐  ┌────────────────┐  ┌────────────────────────┐  │  │
│  │  │   Attention    │  │      LIME      │  │         SHAP           │  │  │
│  │  │ Visualization  │  │   Explainer    │  │      Explainer         │  │  │
│  │  └────────────────┘  └────────────────┘  └────────────────────────┘  │  │
│  └──────────────────────────────────────────────────────────────────────┘  │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

**Figure 1: System Architecture Overview**

### B. Text Preprocessing

The preprocessing pipeline transforms raw text into a format suitable for model input:

1. **Text Cleaning**: Remove URLs, HTML tags, and special characters; retain only alphabetic characters; convert to lowercase; normalize whitespace

2. **Tokenization**: Convert text to sequences of integer tokens using Keras Tokenizer with vocabulary size of 50,000

3. **Padding**: Pad or truncate sequences to fixed length of 400 tokens

  The preprocessing function is defined as:

```
preprocess(text) = normalize(remove_special(remove_html(remove_urls(lowercase(text)))))
```

### C. FastText Word Embeddings

We employ FastText embeddings [27] to represent words as dense vectors. FastText extends Word2Vec by representing words as bags of character n-grams, enabling:

- **Morphological awareness**: Capturing subword information for better handling of rare words and morphological variations
- **Out-of-vocabulary handling**: Generating representations for unseen words based on character n-grams

FastText model configuration:
- Vector dimensionality: 300
- Context window: 5
- Minimum word count: 2
- Training algorithm: Skip-gram
- Training epochs: 10

The embedding matrix E ∈ ℝ^(V×d) is constructed by mapping each word in vocabulary V to its FastText vector of dimension d=300.

### D. BiLSTM Layer

Bidirectional LSTM processes the input sequence in both forward and backward directions, capturing contextual dependencies from both past and future tokens [28].

For input sequence x = (x₁, x₂, ..., xₜ):

**Forward LSTM:**
$$\overrightarrow{h_t} = LSTM(x_t, \overrightarrow{h_{t-1}})$$

**Backward LSTM:**
$$\overleftarrow{h_t} = LSTM(x_t, \overleftarrow{h_{t+1}})$$

**Concatenated output:**
$$h_t = [\overrightarrow{h_t}; \overleftarrow{h_t}]$$

Configuration: 64 units per direction (128 total), return sequences enabled for attention.

### E. Attention Mechanism

We implement Bahdanau-style (additive) attention [18] to compute a weighted sum of BiLSTM outputs, focusing on the most relevant tokens for classification.

**Score computation:**
$$e_t = v^T \tanh(W_h \cdot h_t + b)$$

**Attention weights (softmax normalization):**
$$\alpha_t = \frac{\exp(e_t)}{\sum_{k=1}^{T} \exp(e_k)}$$

**Context vector:**
$$c = \sum_{t=1}^{T} \alpha_t \cdot h_t$$

The attention weights α provide interpretable importance scores for each input token.

### F. Classification Layers

The context vector from attention is passed through:

1. **Dense Layer**: 64 units with ReLU activation and L2 regularization (λ=1e-4)
2. **Dropout Layer**: Rate 0.5 for regularization
3. **Output Layer**: Single unit with sigmoid activation for binary classification

**Loss Function:** Binary Cross-Entropy
$$\mathcal{L} = -\frac{1}{N}\sum_{i=1}^{N}[y_i \log(\hat{y}_i) + (1-y_i)\log(1-\hat{y}_i)]$$

**Optimizer:** Adam with default parameters

### G. Model Training

Training configuration:
- Batch size: 32
- Maximum epochs: 10
- Callbacks:
  - Early Stopping (patience=3, monitor=val_loss)
  - Model Checkpoint (save best model)
  - ReduceLROnPlateau (factor=0.1, patience=2)

---

## IV. Dataset and Preprocessing

### A. Dataset Description

We utilize a balanced dataset comprising fake and real news articles. The dataset is constructed by combining:
- **Fake.csv**: Collection of fake news articles (label=1)
- **True.csv**: Collection of authentic news articles (label=0)

### B. Dataset Statistics

| Metric | Value |
|--------|-------|
| Total Samples | Varies based on source |
| Fake News | 50% |
| Real News | 50% |
| Average Text Length | ~400 tokens |
| Vocabulary Size | 50,000 |
### C. Data Split

| Split | Percentage | Purpose |
|-------|------------|---------|
| Training | 70% | Model training |
| Validation | 15% | Hyperparameter tuning |
| Test | 15% | Final evaluation |

### D. Preprocessing Pipeline

```
┌────────────────┐    ┌────────────────┐    ┌────────────────┐
│  Load Datasets │───▶│ Combine & Label│───▶│  Text Cleaning │
└────────────────┘    └────────────────┘    └───────┬────────┘
                                                    │
                                                    ▼
┌────────────────┐    ┌────────────────┐    ┌────────────────┐
│ Save Tokenized │◀───│    Padding     │◀───│  Tokenization  │
│     Arrays     │    │   (len=400)    │    │                │
└────────────────┘    └────────────────┘    └────────────────┘
```

**Figure 2: Data Preprocessing Pipeline**

---

## V. Experimental Results

### A. Experimental Setup

**Hardware Configuration:**
- GPU: NVIDIA GPU with CUDA support
- RAM: 16GB+
- Storage: SSD for faster data loading

**Software Environment:**
- Python 3.8+
- TensorFlow 2.x
- Gensim (FastText)
- scikit-learn
- LIME, SHAP libraries

### B. Evaluation Metrics

We evaluate model performance using:

1. **Precision**: $P = \frac{TP}{TP + FP}$

2. **Recall**: $R = \frac{TP}{TP + FN}$

3. **F1-Score**: $F1 = 2 \cdot \frac{P \cdot R}{P + R}$

4. **ROC-AUC**: Area under the Receiver Operating Characteristic curve

5. **Calibration**: Reliability of predicted probabilities

### C. Classification Results

| Metric | Score |
|--------|-------|
| Precision | TBD |
| Recall | TBD |
| F1-Score | TBD |
| ROC-AUC | TBD |
| Accuracy | TBD |

*Note: Fill in actual results from your model evaluation*

### D. Confusion Matrix Analysis

```
                    Predicted
                 Fake    Real
Actual  Fake  [  TP   |   FN  ]
        Real  [  FP   |   TN  ]
```

The confusion matrix reveals:
- True Positives (TP): Correctly identified fake news
- True Negatives (TN): Correctly identified real news
- False Positives (FP): Real news misclassified as fake
- False Negatives (FN): Fake news misclassified as real

### E. Comparison with Baseline Methods

| Model | Precision | Recall | F1-Score |
|-------|-----------|--------|----------|
| Logistic Regression + TF-IDF | - | - | - |
| SVM + TF-IDF | - | - | - |
| LSTM | - | - | - |
| BiLSTM | - | - | - |
| **BiLSTM + Attention (Ours)** | - | - | - |

*Note: Fill in comparative results*

### F. Ablation Study

| Configuration | F1-Score |
|--------------|----------|
| BiLSTM only | - |
| BiLSTM + Attention | - |
| BiLSTM + Attention + Dropout | - |
| Full Model | - |

---

## VI. Explainability Analysis

### A. Attention Visualization

The attention mechanism provides inherent interpretability by assigning importance weights to input tokens. We visualize these weights to understand which words the model focuses on during prediction.

**Advantages:**
- Fast computation (no additional model calls)
- Direct insight into model's internal focus
- Useful for identifying potential biases

**Implementation:**
We extract attention weights from the trained model using an intermediate layer model and visualize them as bar charts showing token importance.

### B. LIME (Local Interpretable Model-agnostic Explanations)

LIME [29] explains individual predictions by:
1. Perturbing the input text (removing/modifying words)
2. Observing prediction changes
3. Fitting a local linear model
4. Identifying top influential features

**Configuration:**
- Number of features: 10
- Perturbation method: Word removal
- Output: Interactive HTML visualization

**Sample LIME Explanation:**

| Word | Contribution to FAKE |
|------|---------------------|
| "shocking" | +0.15 |
| "revealed" | +0.12 |
| "sources" | -0.08 |
| "according" | -0.05 |

### C. SHAP (SHapley Additive exPlanations)

SHAP [30] provides theoretically-grounded feature attributions based on game-theoretic Shapley values:

$$\phi_i = \sum_{S \subseteq N \setminus \{i\}} \frac{|S|!(|N|-|S|-1)!}{|N|!}[f(S \cup \{i\}) - f(S)]$$

**Configuration:**
- Explainer: KernelExplainer (model-agnostic)
- Background samples: 100
- Number of samples: 100

**Output:** Summary plots showing global feature importance across dataset.

### D. Comparison of XAI Methods

| Method | Type | Speed | Scope | Faithfulness |
|--------|------|-------|-------|--------------|
| Attention | Internal | Very Fast | Local | Model-specific |
| LIME | Perturbation | Moderate | Local | Approximation |
| SHAP | Game-theoretic | Slow | Local + Global | Theoretically grounded |

### E. Bias Detection

We implement bias probes to detect if the model relies on superficial features:

**Probe Categories:**
1. Political names in neutral context
2. Sensational language patterns
3. Generic control sentences
4. Adversarial mixing

**Bias Detection Results:**

| Probe Type | Mean Prediction | Bias Flag |
|------------|-----------------|-----------|
| Political names | - | - |
| Sensational (true) | - | - |
| Sensational (fake) | - | - |
| Generic | - | - |

A difference > 0.25 between political-name and generic sentences indicates potential bias.

---

## VII. Conclusion and Future Work

### A. Conclusion

In this paper, we presented a comprehensive approach to fake news detection that combines the sequence modeling power of BiLSTM networks with attention mechanisms, enhanced by multiple explainability techniques. Our system addresses the critical need for transparency in automated fake news detection by providing:

1. **Accurate Classification**: The BiLSTM-Attention model effectively captures contextual dependencies in news text for reliable fake/real classification

2. **Multi-perspective Explainability**: Three complementary XAI techniques (Attention, LIME, SHAP) provide diverse insights into model predictions

3. **Bias Detection**: Probing mechanisms help identify and mitigate potential model biases

4. **Practical Deployment Readiness**: The modular architecture supports easy integration into production systems

### B. Limitations

- Computational overhead of BiLSTM compared to simpler models
- SHAP explanations are slow for large datasets
- Model may require fine-tuning for different news domains
- Limited to English language text

### C. Future Work

1. **Transformer Integration**: Explore BERT-based models with attention visualization for improved performance

2. **Multi-modal Detection**: Incorporate image and video analysis for comprehensive fake news detection

3. **Real-time Deployment**: Optimize model for real-time inference in social media monitoring

4. **Cross-domain Generalization**: Develop domain adaptation techniques for robust performance across news categories

5. **Multilingual Support**: Extend the system to support multiple languages

6. **User Study**: Conduct human evaluation of explanation quality and usefulness

---

## Acknowledgments

[Add acknowledgments for advisors, funding sources, computing resources, etc.]

---

## References

[1] K. Shu, A. Sliva, S. Wang, J. Tang, and H. Liu, "Fake news detection on social media: A data mining perspective," *ACM SIGKDD Explorations Newsletter*, vol. 19, no. 1, pp. 22-36, 2017.

[2] S. Malliga et al., "Overview of the shared task on fake news detection from social media text," in *Proc. Third Workshop on Speech and Language Technologies for Dravidian Languages*, 2023, pp. 59-63.

[3] A. Yee, "Post-truth politics & fake news in asia," *Global Asia*, vol. 12, no. 2, pp. 66-71, 2017.

[4] K. Babacan and M. S. Tam, "The information warfare role of social media: Fake news in the Russia-Ukraine war," *Erciyes İletişim Dergisi*, no. 3, pp. 75-92, 2022.

[5] N. Ruchansky, S. Seo, and Y. Liu, "CSI: A hybrid deep model for fake news detection," in *Proc. 2017 ACM Conference on Information and Knowledge Management*, 2017, pp. 797-806.

[6] C. Janiesch, P. Zschech, and K. Heinrich, "Machine learning and deep learning," *Electronic Markets*, vol. 31, no. 3, pp. 685-695, 2021.

[7] A. B. Arrieta et al., "Explainable artificial intelligence (XAI): Concepts, taxonomies, opportunities and challenges toward responsible AI," *Information Fusion*, vol. 58, pp. 82-115, 2020.

[8] A. S. Madhav and A. K. Tyagi, "Explainable artificial intelligence (XAI): Connecting artificial decision-making and human trust in autonomous vehicles," in *Proc. Third International Conference on Computing, Communications, and Cyber-Security*, Springer, 2022, pp. 123-136.

[9] M. Szczepański, M. Pawlicki, R. Kozik, and M. Choraś, "New explainability method for BERT-based model in fake news detection," *Scientific Reports*, vol. 11, no. 1, p. 23705, 2021.

[10] J. Shaikh and R. Patil, "Fake news detection using machine learning," in *2020 IEEE International Symposium on Sustainable Energy, Signal Processing and Cyber Security*, IEEE, 2020, pp. 1-5.

[11] M. Potthast, J. Kiesel, K. Reinartz, J. Bevendorff, and B. Stein, "A stylometric inquiry into hyperpartisan and fake news," *arXiv preprint arXiv:1702.05638*, 2017.

[12] S. Pandey, S. Prabhakaran, N. S. Reddy, and D. Acharya, "Fake news detection from online media using machine learning classifiers," in *Journal of Physics: Conference Series*, vol. 2161, IOP Publishing, 2022, p. 012027.

[13] M. Sch ̈utz, A. Schindler, M. Siegel, and K. Nazemi, "Automatic fake news detection with pre-trained transformer models," in *Pattern Recognition. ICPR International Workshops and Challenges*, Springer, 2021, pp. 627-641.

[14] R. K. Kaliyar, A. Goswami, P. Narang, and S. Sinha, "FNDNet–a deep convolutional neural network for fake news detection," *Cognitive Systems Research*, vol. 61, pp. 32-44, 2020.

[15] A. De, D. Bandyopadhyay, B. Gain, and A. Ekbal, "A transformer-based approach to multilingual fake news detection in low-resource languages," *Transactions on Asian and Low-Resource Language Information Processing*, vol. 21, no. 1, pp. 1-20, 2021.

[16] S. Hochreiter and J. Schmidhuber, "Long short-term memory," *Neural Computation*, vol. 9, no. 8, pp. 1735-1780, 1997.

[17] M. Schuster and K. K. Paliwal, "Bidirectional recurrent neural networks," *IEEE Transactions on Signal Processing*, vol. 45, no. 11, pp. 2673-2681, 1997.

[18] D. Bahdanau, K. Cho, and Y. Bengio, "Neural machine translation by jointly learning to align and translate," *arXiv preprint arXiv:1409.0473*, 2014.

[19] A. Vaswani et al., "Attention is all you need," in *Advances in Neural Information Processing Systems*, 2017, pp. 5998-6008.

[20] J. Devlin, M.-W. Chang, K. Lee, and K. Toutanova, "BERT: Pre-training of deep bidirectional transformers for language understanding," *arXiv preprint arXiv:1810.04805*, 2018.

[21] Y. Liu et al., "RoBERTa: A robustly optimized BERT pretraining approach," *arXiv preprint arXiv:1907.11692*, 2019.

[22] A. Conneau et al., "Unsupervised cross-lingual representation learning at scale," *arXiv preprint arXiv:1911.02116*, 2019.

[23] E. Hashmi et al., "Advancing fake news detection: Hybrid deep learning with FastText and explainable AI," *IEEE Access*, 2024.

[24] V. Dua, A. Rajpal, S. Rajpal, M. Agarwal, and N. Kumar, "I-FLASH: Interpretable fake news detector using LIME and SHAP," *Wireless Personal Communications*, vol. 131, no. 4, pp. 2841-2874, 2023.

[25] S. Jain and B. C. Wallace, "Attention is not explanation," in *Proc. 2019 Conference of the North American Chapter of the Association for Computational Linguistics*, 2019, pp. 3543-3556.

[26] S. Wiegreffe and Y. Pinter, "Attention is not not explanation," in *Proc. 2019 Conference on Empirical Methods in Natural Language Processing*, 2019, pp. 11-20.

[27] P. Bojanowski, E. Grave, A. Joulin, and T. Mikolov, "Enriching word vectors with subword information," *Transactions of the Association for Computational Linguistics*, vol. 5, pp. 135-146, 2017.

[28] A. Graves and J. Schmidhuber, "Framewise phoneme classification with bidirectional LSTM and other neural network architectures," *Neural Networks*, vol. 18, no. 5-6, pp. 602-610, 2005.

[29] M. T. Ribeiro, S. Singh, and C. Guestrin, ""Why should I trust you?": Explaining the predictions of any classifier," in *Proc. 22nd ACM SIGKDD International Conference on Knowledge Discovery and Data Mining*, 2016, pp. 1135-1144.

[30] S. M. Lundberg and S.-I. Lee, "A unified approach to interpreting model predictions," in *Advances in Neural Information Processing Systems*, vol. 30, 2017.

---

## Appendix A: Model Hyperparameters

| Parameter | Value |
|-----------|-------|
| Embedding Dimension | 300 |
| BiLSTM Units | 64 per direction |
| Dense Layer Units | 64 |
| Dropout Rate | 0.5 |
| L2 Regularization | 1e-4 |
| Batch Size | 32 |
| Max Epochs | 10 |
| Sequence Length | 400 |
| Vocabulary Size | 50,000 |
| Optimizer | Adam |
| Loss Function | Binary Cross-Entropy |

---

## Appendix B: Code Availability

The implementation code is organized as follows:

| Directory | Contents |
|-----------|----------|
| `tools/` | Data preprocessing scripts |
| `models/` | Model training and artifacts |
| `layers/` | Custom Keras layers (Attention) |
| `explainability/` | LIME, SHAP, Attention visualizers |
| `evaluation/` | Metrics and bias detection |

---

*Manuscript received [DATE]; revised [DATE]; accepted [DATE].*
