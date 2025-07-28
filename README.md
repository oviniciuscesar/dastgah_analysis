# Comparative Computational Analysis of Persian Modal Music

This repository contains a Python script (`analysis.py`) designed to perform a detailed and comparative musicological analysis of musical performances within the Persian modal system (Dastgah). The main objective is to quantify the fidelity of a musical performance (whether human or AI-generated) to a theoretical model, evaluating scale structure, tuning accuracy, and the syntax of melodic patterns.

The methodology was developed based on Audio Signal Processing and Music Information Retrieval (MIR) techniques, informed by approaches described in the academic literature on the subject.

## Methodology

The script implements a multi-phase, hybrid analysis pipeline:

1. **Preprocessing and Note Extraction:**
* The pitch contour of each audio track is extracted using the **pYIN** algorithm from the `librosa` library.

* Musical notes are segmented from the pitch contour using a robust **attack detection** (note onsets) approach.
* Each note segment is validated based on **minimum duration** and **pitch stability** criteria to ensure that only consistent musical events are analyzed.

2. **Phase 2: Structural and Pitch Analysis:**
* **Quantization:** The *N* valid notes excluded in the performance are mapped to the 7 theoretical classes of the reference Dastgah (in this case, Shur).
* **Fuzzification:** Inspired by Abdoli's (2011) approach, each note class (both from the performance and theory) is modeled as a **Type 2 Interval Fuzzy Set (IT2FS)**, which mathematically represents the note and its pitch uncertainty.
* **Similarity Calculation:** The **Jaccard Similarity Measure (JSM)** is used to calculate the similarity between performed and theoretical notes. A final score is calculated using a **Fuzzy Weighted Average (FWA),** which considers both the theoretical importance of each note and its frequency of occurrence in performance.

3. **Phase 3: Syntactic and Melodic Pattern Analysis:**
* Based on the concept of "skeletal melodic models" by Darabi et al. (2006), this phase analyzes the **sequence** of notes.
* The script quantifies the occurrence of specific and idiomatic melodic patterns, such as the articulation of the second degree in the tonic, calculating the "Articulation Rate" to assess understanding of musical "grammar."

## Academic References
* **Abdoli, S. (2011).** *IRANIAN TRADITIONAL MUSIC DASTGAH CLASSIFICATION*. ISMIR 2011.
* **Darabi, N., et al. (2006).** *RECOGNITION OF DASTGAH AND MAQAM FOR PERSIAN MUSIC...*.
* **Ebrat, D., et al. (2022).** *IRANIAN MODAL MUSIC (DASTGAH) DETECTION USING DEEP NEURAL NETWORKS*.
* **Babiracki, C. M., & Nettl, B. (1987).** *Internal Interrelationships in Persian Classical Music: The Dastgah of Shur...*. Asian Music.
