This Repository contains the code for 2 modules: 
1. Resume Plagiarism Detector
2. Resume AI text Detector

"plagiarism_detector.py" uses functions from "common_utils.py" and "plagiarism_heuristics.py"

"resumes_loader.py" uses functions from "common_utils.py"

"ai_detector.py" can be used in 2 modes (infer, train). The model used is a HuggingFace model microsoft/deberta-v3-small which was trained on 5000+ resumes of Human and AI from kaggle datasets.This module also uses functions from "common_utils.py".
