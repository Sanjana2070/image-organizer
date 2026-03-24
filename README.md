# Image Organizer

## Introduction

Image Organizer is a computer vision pipeline for structuring large collections of images automatically. It processes raw images and organizes them based on visual content, primarily focusing on faces.

---

## Problem Statement

Managing large image collections manually is inefficient and error-prone. Images often lack proper organization, making it difficult to:

- Identify people across images  
- Group similar images  
- Separate useful images from irrelevant ones  

There is a need for an automated system that can analyze visual content and organize images into meaningful groups.

---

## Overview

This project builds an end-to-end pipeline that:

1. Detects faces in images  
2. Generates embeddings for each detected face  
3. Groups similar faces using clustering  
4. Separates images with no detectable faces  

The pipeline produces structured outputs along with metadata for reproducibility and further processing.

---

## Solution

### Key Features

- Face detection and extraction from images  
- Embedding generation for similarity comparison  
- Clustering of faces into identity groups  
- Separation of images without faces  
- Structured metadata generation  

---

## Usage

1. Add images to:
input_images/

2. Run the pipeline:
python main.py

3. Output will be generated in:
output.py


---

## Output Description 

- **metadata/**  
  JSON files linking images, faces, and embeddings  

- **clusters/**  
  Grouped faces representing similar identities  

- **no_faces/**  
  Images where no faces were detected  

---

## Future Work

- CLIP-based categorization of non-face images  
- UI for browsing clusters and categories  
- Duplicate image detection  