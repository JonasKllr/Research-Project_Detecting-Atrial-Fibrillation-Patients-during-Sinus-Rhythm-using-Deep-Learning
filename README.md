# Research Project: Detecting Atrial Fibrillation Patients during Sinus Rhythm using Deep Learning

This repository contains the code used in a research project during my master's program.


## Abstract

Atrial Fibrillation (AF) is one of the most common arrhythmias that can lead to a stroke and heart failure when untreated.
Therefore, automatically detecting AF in its early paroxysmal stages can initiate medical treatment to prevent AF to progress in its more frequent stages.
A deep learning approach was applied to detect AF in Sinus Rhythm (SR) Electrocardiograms (ECGs) and was trained with the Computers in Cardiology Challenge 2001 (CinC 2001) dataset [CinC2001].
To lower the inﬂuence of the size of the dataset, a transfer learning approach was investigated by pretraining the deep learning model with the PhysioNet/Computing in Cardiology Challenge 2021 (CinC 2021) dataset [CinC2021] to estimate a subject’s age from SR ECGs.
The pretrained model for age estimation produced a Mean Absolute Error (MAE) of 11.50 years on a separate test data split.
The model trained without transfer learning achieved an accuracy of 53.00 % and the model trained with transfer learning achieved an accuracy of 53.52 % on a separate test data slit.
In a further experiment including less recordings per subject in the training data set the model produced an accuracy of 62.07 %.
The results from the age estimation model showed that the chosen deep learning approach is able to learn subtle featuresfrom SR ECGs recordings.
However, the models could not reliably distinguish between AF and non-AF SR ECGs.
The size of the CinC 2001 dataset was identiﬁed as the main limitation of the approach.

## Concept

The CinC 2001 dataset consists of 50 ECG recordings which have a duration of 30 minutes.
Each recording contains two unknown leads.
For the model's training process, the recordings were split up into 10 second segments with the aim to lower the models complexity and to increase the dataset's size.
Furthermore the ECG signals were preprocessed to remove outliers and segments with signal clipping.
Finally, a Butterworth [butterworth] bandpass filter was applied to eliminate noise.

The datasets PTB-XL [PTB-XL], Chapman-Shaoxing [Chapman-Shaoxing] and Ningbo [Ningbo] were used from the CinC 2021 dataset.
The recordings were preprocessed with the same methods the CinC 2001 dataset was preprocessed.
Furthermore, two leads per 10 second recording were chosen ranomly to match the structure of the CinC 2001 ECG recordings.

|<img src="./img/concept.png" alt="drawing" width="700"/>|
|:--:|
|Conceptual steps conducted in the project.|




The model's training was done in three steps:

**1. Step:**
A Convolutional Neural Network (CNN) was trained on the CinC 2001 dataset to distinguish between ECGs from patients with and without AF.
The model's hyperparameters were determined in a grid search.
The best performing model was saved.

**2. Step:**
The best performing model from the previous step was now trained on the CinC 2021 dataset with the aim to estimate a patients age from an ECG recording.
The best performing model was saved.

**3. Step:**
The layers of the model saved in the previous step were now frozen.
One by one, the layers were made trainable and the model was re-trianed with the CinC 2001 dataset with the original goal to distinguish between patients with and without AF.

The best performing models found in step one and step three were then compared to investigate the influence of transfer learning.


## Bibliography 

- [CinC2001] George Moody et al. “Predicting the onset of paroxysmal atrial ﬁbrillation: The Computers in Cardiology Challenge 2001”. In: Computers in Cardiology 2001. Vol. 28 (Cat. No. 01CH37287). IEEE. 2001, pp. 113–116.
- [CinC2021] Matthew A Reyna et al. “Will two do? Varying dimensions in electrocardiography: the PhysioNet/Computing in Cardiology Challenge 2021”. In: 2021 Computing in Cardiology (CinC). Vol. 48. IEEE. 2021, pp. 1–4.
- [butterworth] Stephen Butterworth et al. “On the theory of ﬁlter ampliﬁers”. In: Wireless Engineer 7.6 (1930), pp. 536–541.