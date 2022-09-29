# ILD
In this study, we used deep learning algorithms that integrated clinical history with chest CT scans to diagnose five types of ILD, including usual interstitial pneumonia (UIP), chronic hypersensitivity pneumonitis (CHP), nonspecific interstitial pneumonia (NSIP), sarcoidosis, and “other” ILD. We also used these algorithms to determine a patient’s 3-year survival rate. Among 449 patients with a consensus diagnosis collected from 230 medical centers in the United States, from 09/2014 - 04/2021, 132 (29.4 %) were confirmed as UIP, and 22 (9.4%) deceased patients. In a test set of 128 patients from an independent center, the deep learning algorithm achieved an area under the curve of 0.828 in diagnosing a UIP pattern, outperforming a senior thoracic radiologist (p<0.05), two senior general radiologists (p<0.001), and a senior pulmonologist (p<0.001). 

![alt text](https://github.com/lzl199704/ILD/blob/main/util/Figure3.png)

The Transformer framework with the inputs of longitudinal CT scans and clinical follow-ups also improved the detection of patients with a low 3-year survival rate, achieving a sensitivity of 72.7% and a negative predictive value of 94.6% with only three false negatives. Thus, when clinical history and associated CT scans are available, the proposed deep learning system can help clinicians diagnose and classify ILD patients and, importantly, dynamically predict disease progression and prognosis.

![alt text](https://github.com/lzl199704/ILD/blob/main/util/f4_ild_300dpi.png)

# Code
The ILD_classification.py in the code folder is the code for the joint model.

For the survival analysis, the transformer model was developed to predict survival chance. The survival_analysis_transformer.py in the code folder can be used for this application.

# Citation
If you find code helpful, cite paper:

