# KNOW-NET: A Knowledge-Aware Cross-Attention Network with NER for Fake News Classification

[![IEEE](https://img.shields.io/badge/IEEE-QPAIN%202026-blue)](https://doi.org/10.1109/QPAIN69676.2026.11546260)
[![Python](https://img.shields.io/badge/Python-3.9%2B-blue)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-1.13%2B-orange)](https://pytorch.org/)
[![License: All Rights Reserved](https://img.shields.io/badge/License-All%20Rights%20Reserved-red.svg)](#license)

Official implementation of **KNOW-NET**, published at the *2026 IEEE 2nd International Conference on Quantum Photonics, Artificial Intelligence, and Networking (QPAIN)*, held 16 to 18 April 2026 in Chittagong, Bangladesh.

**DOI:** [10.1109/QPAIN69676.2026.11546260](https://doi.org/10.1109/QPAIN69676.2026.11546260)

## Authors

- Mahfuza Maisha, Department of Computer Science and Engineering, International Islamic University Chittagong (maisha.mahfuza13@gmail.com)
- Umme Kawsher, Department of Computer Science and Engineering, International Islamic University Chittagong
- Zinnia Sultana, Department of Computer Science and Engineering, International Islamic University Chittagong

## Abstract

The proliferation of online misinformation requires robust detection systems, yet existing knowledge graph based methods often use simplistic fusion that limits text knowledge interaction. KNOW-NET addresses this by integrating RoBERTa text encoding with DBpedia embeddings via dynamic cross-attention, and by employing multi-task learning with Named Entity Recognition (NER) supervision for regularization and entity understanding. On the Kaggle Fake News dataset, KNOW-NET achieves state-of-the-art performance of 0.9855 accuracy and 0.9855 F1-score, a 1.41 percent absolute improvement over DEAP-FAKED (p < 0.0001). The model operates on news titles only, maintains balanced error rates, and achieves real-time inference at 15 ms per sample.

## Key Contributions

- **Dynamic Cross-Attention Fusion**: Replaces the static concatenation used in prior knowledge-graph based methods, allowing each text token to dynamically attend to relevant knowledge graph entities.
- **Multi-Task NER Supervision**: Jointly optimizes fake news classification and named entity recognition within a single training objective, improving regularization and entity-aware reasoning.
- **Title-Only, Real-Time Design**: Operates on news titles alone using DBpedia, achieving 15 ms per-sample inference suitable for social media monitoring.
- **Comprehensive Statistical Validation**: Includes paired t-tests, McNemar's test, Cohen's d effect size, confidence intervals, and ablation studies isolating the contribution of each component.

## Architecture Overview

```
Raw News Titles
      |
Text Cleaning -> Bias Mitigation -> Named Entity Recognition (spaCy)
      |
Entity Linking (DBpedia) -> Knowledge Embedding -> Stratified Split
      |
RoBERTa Text Encoder (first 6 layers frozen)      DBpedia Entity Embeddings
      |                                                    |
      +-------------------- Cross-Attention Fusion --------+
                                  |
                    Shared Fused Representation
                       /                    \
        Classification Head              NER Head
      (Dense + Sigmoid)              (Token-level IOB2)
                       \                    /
                    Multi-Task Loss (Eq. 4)
```

Full mathematical formulation, training algorithm, and diagrams are provided in the paper (Section III).

## Results Summary

| Model | Accuracy | F1-Score | Architecture |
|---|---|---|---|
| **KNOW-NET** | **0.9855** | **0.9855** | RoBERTa + Cross-Attention + KG |
| SentRoBERTa | 0.9847 | 0.9847 | RoBERTa |
| DEAP-FAKED | 0.9714 | 0.9714 | BiLSTM + KG |
| EntWiki-BiLSTM | 0.9714 | 0.9714 | BiLSTM + Wikipedia |
| StackedBiLSTM | 0.9660 | 0.9660 | BiLSTM |
| ExtraTree | 0.9611 | 0.9611 | Tree-based |
| LSTM | 0.5080 | 0.3369 | LSTM |

Statistical significance: paired t-test t = 4.1759, p = 3.1 x 10^-5; McNemar's test p = 3.8 x 10^-5; Cohen's d = 0.0816; 95 percent CI [0.0075, 0.0208].

Ablation study: removing cross-attention causes the largest drop (-1.05 percent), followed by removing KG embeddings (-1.55 percent) and removing the NER head (-0.55 percent), confirming that all three components contribute meaningfully to performance.

Full results, confusion matrices, and architectural comparisons against 2024-2025 knowledge-aware methods are available in Sections IV and V of the paper.

## Repository Structure

```
.
├── Implementations/
│   ├── KnowNet_Data Preprocessing and Ablation Study.ipynb
│   ├── Base Model Implementation (DeepFake).ipynb
│   ├── KnowNet_OtherBaselines.ipynb
│   └── KnowNet Implementation with all Visualizations and Statistical Analysis.ipynb
├── LICENSE
└── README.md
```

### Notebook Descriptions

- **KnowNet_Data Preprocessing and Ablation Study.ipynb**: End-to-end preprocessing pipeline, including text cleaning, bias term removal, spaCy-based NER, DBpedia entity linking, stratified train/validation/test splitting, and ablation study configurations.
- **Base Model Implementation (DeepFake).ipynb**: Reimplementation of the DEAP-FAKED baseline using DBpedia embeddings under identical preprocessing and splits.
- **KnowNet_OtherBaselines.ipynb**: Implementation of the remaining baseline models, including ExtraTreeClassifier, LSTM, SentRoBERTa, StackedBiLSTM, and EntWiki-StackedBiLSTM.
- **KnowNet Implementation with all Visualizations and Statistical Analysis.ipynb**: Full KNOW-NET model implementation, along with statistical significance testing, effect size analysis, and result visualizations.

## Dataset

This project uses the [Kaggle Fake News dataset](https://www.kaggle.com/c/fakenews/overview) enriched with DBpedia knowledge graph embeddings, spaCy-derived NER labels, and the preprocessing steps described in the paper.

The processed dataset files exceed GitHub's size limits and are therefore hosted on Kaggle instead of this repository:

**Dataset link:** [https://www.kaggle.com/datasets/mahfuzamaisha13/fake-news-dataset-for-knownet](https://www.kaggle.com/datasets/mahfuzamaisha13/fake-news-dataset-for-knownet)

If you are unable to access the dataset from the link above, please email **maisha.mahfuza13@gmail.com** and it will be shared with you directly.

## Requirements

The notebooks were developed and tested on an NVIDIA Tesla T4 GPU (Google Colab). Core dependencies include:

```
torch
torchvision
torchaudio
transformers
datasets
spacy
en_core_web_md (spaCy model)
pandas
numpy
scikit-learn
scipy
statsmodels
matplotlib
seaborn
tqdm
requests
```

Install the core dependencies with:

```bash
pip install torch torchvision torchaudio transformers datasets spacy pandas numpy scikit-learn scipy statsmodels matplotlib seaborn tqdm requests
python -m spacy download en_core_web_md
```

## Usage

1. Download the preprocessed dataset from the Kaggle link above, or request it by email.
2. Run **KnowNet_Data Preprocessing and Ablation Study.ipynb** to reproduce the preprocessing pipeline and ablation configurations, or place the already-preprocessed files in the expected paths.
3. Run **Base Model Implementation (DeepFake).ipynb** and **KnowNet_OtherBaselines.ipynb** to reproduce the baseline results.
4. Run **KnowNet Implementation with all Visualizations and Statistical Analysis.ipynb** to train KNOW-NET and reproduce the reported performance, statistical tests, and figures.

## Citation

If you use this work, please cite:

```bibtex
@INPROCEEDINGS{knownet2026,
  author={Maisha, Mahfuza and Kawsher, Umme and Sultana, Zinnia},
  booktitle={2026 IEEE 2nd International Conference on Quantum Photonics, Artificial Intelligence, and Networking (QPAIN)},
  title={KNOW-NET: A Knowledge-Aware Cross-Attention Network with NER for Fake News Classification},
  year={2026},
  address={Chittagong, Bangladesh},
  doi={10.1109/QPAIN69676.2026.11546260}
}
```

## License

Copyright (c) 2026 Mahfuza Maisha, Umme Kawsher, Zinnia Sultana. All rights reserved.

This repository is shared for reference and academic transparency purposes only. No license is granted to use, copy, modify, distribute, or create derivative works from this code without prior written permission from the main author. For permission requests, please contact maisha.mahfuza13@gmail.com.

## Contact

For questions, collaboration inquiries, or dataset access, please contact:

**Mahfuza Maisha**

Email: maisha.mahfuza13@gmail.com

GitHub: [C221213-Maisha](https://github.com/C221213-Maisha)
