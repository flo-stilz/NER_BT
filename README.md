# Named Entity Recognition with BERT on medical abstracts

## Introduction
Named Entity Recognition is the task of classifying word sequences in pre-defined classes. This work focuses on using BERT architectures to classify important medical classes like e.g. "Diagnosis", "Symptoms" and so on within medical abstracts for rare diseases.

The project is by [Florian Stilz](https://github.com/flo-stilz/)
from the [Technical University of Denmark](https://www.dtu.dk/english). 

## Results

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
