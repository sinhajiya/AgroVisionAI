# AgroVisionAI: Multimodal Plant Disease Diagnosis
A multimodal AI system that analyzes crop leaf images, describes visual symptoms, retrieves agricultural knowledge from expert documents, and generates an interpretable disease diagnosis with treatment recommendations.
This project combines computer vision, image captioning, retrieval-augmented generation, and LLM reasoning into a single end-to-end pipeline.

## Core pipeline

### CLIP: Rough similarity prediction

CLIP is used for zero-shot classification, giving coarse label guesses based on visual similarity.
It provides an approximate idea of what the leaf might resemble, even without disease-specific training.

### BLIP: Symptom description

BLIP (Bootstrapped Language-Image Pretraining) generates a natural-language caption of the leaf image.

It produces human-like descriptions such as:
`
“a green leaf with yellow lesions and circular brown spots”
`
. Unlike CLIP, BLIP doesn’t guess a label, it describes the visual evidence.

### LLM Reasoning: Interprets symptoms

The LLM uses the BLIP caption, optional CLIP guesses, and retrieved knowledge to infer the most likely disease.
It can interpret patterns, match symptoms, incorporate context, and produce a structured diagnosis and suggested actions.

### RAG: Expert agricultural knowledge

RAG retrieves relevant content from agricultural PDFs (ICAR, FAO, crop protection guides).
This grounds the LLM’s reasoning in factual treatment methods and disease descriptions.

## Dataset
Plant Village dataset dowloaded from kaggle is being used here. 

Link: https://www.kaggle.com/datasets/soumiknafiul/plantvillage-dataset-labeled

### About dataset _(from kaggle)_
This dataset is recreated using offline augmentation form the original one. Healthy and diseased images of five crops are taken into consideration which are apple, corn, grape, potato and tomato containing 31,397 images. The images of 5 crops species were classified into 25 classes. Among these 25 classes, 20 classes contain diseased images and 5 classes contain healthy images. Segmented and gray-scale images are generated for each image and then the dataset is partitioned into three portions for color, gray-scale and segmented images. Overall, the dataset contains a total of 94,191 images. 
