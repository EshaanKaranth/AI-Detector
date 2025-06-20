 # AI Resume Detector

## The Repository contains both functionalities:
       1.Resume Plagiarism Detector
       2.Resume AI text Detector

* **plagiarism_detector.py** uses functions from **common_utils.py** and **plagiarism_heuristics.py**

* **resumes_loader.py** uses functions from **common_utils.py**

* **ai_detector.py** can be used in 2 modes (infer, train). The model used is a HuggingFace model **microsoft/deberta-v3-small** which was trained on 5000+ resumes of Human and AI from kaggle datasets.This module also uses functions from **common_utils.py**

* All the requirements can be downloaded from the *requirements.txt* file

* **Libre Office** needs to be installed in the system for running headless *soffice* doc/docx to pdf conversion

* **Elron/bleurt-base-512** model needs to be downloaded from HugginFace Models for generating Bleurt scores.

<pre><code>
The recommended directory structure:

AI Detector
├── ai_detector.py    
├── plagiarism_detector.py
├── plagiarism_heuristics.py            # contains all the NLP functions which can be modified according to the situation
├── resumes_loader.py
├── common_utils.py                     # Shared utility functions
├── .env                                # requires Qdrant url, api key 
├── kaggle_data/                        # .csv files
├── resumes/                            # Locally stored Resume files for reference (used for contact info extraction when plagiarism is flagged, i.e between input file and source file)
├── test/                               # input files
├── results/                            # Output results
│ ├── ai_results.json                   
│ └── plagiarism_results.json
├── processing/                         # all these files will be created automatically when the program runs.
│ ├── count.json
│ ├── failed_files.json
│ └── processed_files.json
├── logs/                               # contains log files created during run time, handled by RotatingFileHandler
├── aienv/                              # project virtual environment
├── ai_resume_detector_optimized/       # Fine tuned Model
├── .gitignore                           
├── README.md
└── requirements.txt </code></pre>


<code><pre>
## **The Datasets used:**

Human Resumes :
https://www.kaggle.com/datasets/snehaanbhawal/resume-dataset
https://www.kaggle.com/datasets/gauravduttakiit/resume-dataset
https://www.kaggle.com/datasets/jillanisofttech/updated-resume-dataset
https://www.kaggle.com/datasets/serkanp/resume-screening-dataset
https://www.kaggle.com/datasets/leenardeshmukh/curriculum-vitae
https://www.kaggle.com/datasets/wahib04/multilabel-resume-dataset
https://www.kaggle.com/datasets/chingkuangkam/resume-text-classification-dataset

AI Resumes:
https://www.kaggle.com/datasets/ramzybakir/ai-generated-resume-dataset
https://www.kaggle.com/datasets/jithinjagadeesh/resume-dataset
https://www.kaggle.com/datasets/mdtalhask/ai-powered-resume-screening-dataset-2025
https://www.kaggle.com/datasets/sohrabbahari/ai-resume-screening-dataset
https://huggingface.co/datasets/InferencePrince555/Resume-Dataset


[few datasets may require special functions to process into .csv format]

</code></pre>




