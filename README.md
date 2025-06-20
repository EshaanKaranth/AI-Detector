This Repository contains the code for 2 modules: 
1. Resume Plagiarism Detector
2. Resume AI text Detector

"plagiarism_detector.py" uses functions from "common_utils.py" and "plagiarism_heuristics.py"

"resumes_loader.py" uses functions from "common_utils.py"

"ai_detector.py" can be used in 2 modes (infer, train). The model used is a HuggingFace model microsoft/deberta-v3-small which was trained on 5000+ resumes of Human and AI from kaggle datasets.This module also uses functions from "common_utils.py".

All the requirements can be downloaded from the "requirements.txt" file

Libre Office needs to be installed in the system for running headless soffice doc/docx to pdf conversion

<pre><code>The recommended directory structure:

AI Detector
    |
    |---ai_detector.py
    |---plagiarism_detector.py
    |---plagiarism_heuristics.py
    |---resumes_loader.py
    |---common_utils.py
    |
    |---.env
    |---kaggle_data
    |---resumes
    |---test
    |---results
    |	    |
    |    	|---ai_results.json
    |	    |---plagiarism_results.json
    |
    |---processing
    |	      |
    |	      |---count.json
    |	      |---failed_files.json
    |	      |---processed_files.json
    |
    |---logs
    |---aienv
    |---ai_resume_detector_optimized
    |---.gitignore
    |---README.md
    |---requirements.txt
</code></pre>

