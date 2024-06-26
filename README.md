# Named Entity Recognition with BERT on medical abstracts

<p align="center"><img src="paper_figures/Prediction_Results.png" width="800px"/></p>

## Introduction
Named Entity Recognition is the task of classifying word sequences in pre-defined classes. This work focuses on using BERT architectures to classify important medical classes like e.g. "Diagnosis", "Symptoms" and so on within medical abstracts for rare diseases.

For additional details, please see my paper:  
"[Segmentation of medical records with natural
language processing tools](https://github.com/flo-stilz/NER_BT/blob/main/paper_figures/NER_with_BERT_for_medical_abstracts.pdf)"
The project is by [Florian Stilz](https://flo-stilz.github.io/) and supervised by [Ole Winther](https://olewinther.github.io/)
from the [Technical University of Denmark](https://www.dtu.dk/english). 

## Dataset

The final dataset contains 186 medical abstracts with
109 Gaucher abstracts and 76 Fabry abstracts, respectively. [Prodigy](https://prodi.gy/) was used for annotating the data.

## Results

The final results display the average scores of 5 training runs with the standard deviation denoted next to it.

<table>
    <col>
    <col>
    <tr>
        <th>Model</th>
        <th>F1-Score</th>
    </tr>
    <tr>
        <td>BioBERT</td>
        <td>70.07±1.30%</td>
    </tr>
    <tr>
        <td>Bio+ClinicalBERT</td>
        <td>68.97±1.45%</td>
    </tr>
    <tr>
        <td>BERT_LARGE</td>
        <td>68.90±0.76%</td>
    </tr>
    <tr>
        <td>DistilBERT</td>
        <td>68.23±1.21%</td>
    </tr>
    <tr>
        <td>en_vectors_web_lg (CNN)</td>
        <td>68.07±1.73%</td>
    </tr>
    <tr>
        <td>BERT_BASE</td>
        <td>67.92±1.22%</td>
    </tr>

</table>
