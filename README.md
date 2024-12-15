# Automaton of Eerie Dreams

This project delves into the eerie and dream-like qualities of **Natsume Sōseki’s _Ten Nights of Dreams_** through the lens of machine learning and AI-driven visual composition. Using embeddings derived from text, we create a dynamic representation of the unsettling yet poetic atmosphere of Sōseki's work.

## Project Overview
The AI system leverages a **BERT model** trained on _Ten Nights of Dreams_ to generate high-dimensional numerical embeddings. These embeddings encapsulate the semantic and emotional nuances of the text. 

### Key Steps:
1. **Text Embedding Generation:**
   - The BERT model processes each night’s text and outputs embeddings (numerical representations of the text).

2. **Embedding Utilization:**
   - Embedding values are used to manipulate parameters of a visual composition.
   - Each dream’s unique pattern continuously updates the visuals in real-time.

3. **Dynamic Visual Composition:**
   - Built in **TouchDesigner**, the visuals transform dynamically based on embedding-driven parameters.
   - This interplay creates an evocative, eerie atmosphere reflective of Sōseki’s dreamscapes.

### Example:
- **1st Night Embeddings:**
  ```json
  {"1st Night": [-0.06213264912366867, 0.026785440742969513, 0.308075487613678, -0.2862982153892517, …]}
  ```
- These embedding values control the weight, texture, and movement within the visual network.

## Future Improvements
- Extend to additional literary works to explore how different texts translate to dynamic visuals.
- Experiment with other transformer models for richer semantic representations.
- Introduce interactivity, allowing users to adjust visuals based on input.

## Acknowledgments
- Inspired by **Natsume Sōseki** and **Refik Anadol**.
