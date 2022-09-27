# ILD
For the accurate diagnosis of interstitial lung disease (ILD), a consensus of radiologic, pathological, and clinical findings is vital. Management of ILD also requires thorough follow-up computed tomography (CT) studies and lung function tests to assess disease progression, severity, and response to treatment. However, accurate classification of ILD subtypes can be challenging, especially for those not accustomed to reading chest CTs regularly. Dynamic models to predict patient survival rates based on longitudinal data are challenging to create due to disease complexity, variation, and irregular visit intervals.

In this study, we used deep learning algorithms that integrated clinical history with chest CT scans to diagnose five types of ILD, including usual interstitial pneumonia (UIP), chronic hypersensitivity pneumonitis (CHP), nonspecific interstitial pneumonia (NSIP), sarcoidosis, and “other” ILD. We also used these algorithms to determine a patient’s 3-year survival rate. Among 449 patients with a consensus diagnosis collected from 230 medical centers in the United States, from 09/2014 - 04/2021, 132 (29.4 %) were confirmed as UIP, and 22 (9.4%) deceased patients. In a test set of 128 patients from an independent center, the deep learning algorithm achieved an area under the curve of 0.828 in diagnosing a UIP pattern, outperforming a senior thoracic radiologist (p<0.05), two senior general radiologists (p<0.001), and a senior pulmonologist (p<0.001). The Transformer framework with the inputs of longitudinal CT scans and clinical follow-ups also improved the detection of patients with a low 3-year survival rate, achieving a sensitivity of 72.7% and a negative predictive value of 94.6% with only three false negatives. Thus, when clinical history and associated CT scans are available, the proposed deep learning system can help clinicians diagnose and classify ILD patients and, importantly, dynamically predict disease progression and prognosis.

# Code
There are three models for ILD classification, MLP, CT_CNN, and joint model. The MLP model only utilizes clinical information, CT_CNN model only uses CT images as inputs, and the joint model combines both information as inputs. 
![alt text](https://user-images.githubusercontent.com/106784487/192621173-8ac594b3-5023-4287-814e-130e985917ba.png)
For the survival analysis, the transformer model was developed to predict survival chance. 

# Citation
If you find code helpful, cite paper:

