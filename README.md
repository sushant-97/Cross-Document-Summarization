# Cross-Document-Summarization
Given a cluster of documents related to specific topic, task is to generate summary capturing relevant information with the help of GNN representations.

## Project Overview

This project focuses on extracting and summarizing information from a set of documents that discuss the same topic. Our goal is to condense this information into a coherent summary that captures the essential elements of the discussed topic. This is particularly useful in scenarios where decision-makers need quick insights from large volumes of text.

## Objectives

- **Identify Key Information**: Extract key details and information from multiple documents.
- **Summarization**: Generate concise summaries that accurately reflect the content of the original texts.
- **Knowledge Transfer**: Apply insights from one document to enhance the summarization of others.
- **Model Integration**: Explore effective strategies for merging embeddings from Language Model (LM) and Graph Neural Network (GNN).

## Challenges

- **Labelled Data Scarcity**: One of the main challenges is the scarcity of labeled datasets for training summarization models.
- **Quality of Summaries**: Ensuring that the generated summaries are accurate and retain the essential information from the original documents.

## Methodology

### Data Collection

Documents related to the chosen topic are collected and preprocessed for further analysis.

### Summarization Using Pretrained Models

We employ the pretrained model `Facebook/bart-large-cnn` from Hugging Face for initial document summarization. This model provides a strong baseline with its ability to generate high-quality summaries.

### Multi-Document Summarization

Summaries generated from the `Facebook/bart-large-cnn` model serve as a pseudo ground truth. We further refine these summaries to ensure they cover all relevant information across multiple documents.

### Merging LLM and GNN Embeddings

To enhance the summarization process, we experiment with merging embeddings from Language Models (LM) and Graph Neural Networks (GNN). This involves:
- Generating embeddings for each document using both LM and GNN.
- Combining these embeddings using concatenation or more sophisticated fusion techniques to capture both semantic and structural information.

## Getting Started

To set up this project locally, follow these steps:

```bash
git clone https://github.com/your-repository.git
cd your-repository
pip install -r requirements.txt


## Methodology
![Methodology](Methodology.jpeg)

## Qualitative Results
![Results](Results.jpeg)

## Limitations of Future Scope
![Limitations of the work](Limitations_and_Future_Scope.jpeg)
