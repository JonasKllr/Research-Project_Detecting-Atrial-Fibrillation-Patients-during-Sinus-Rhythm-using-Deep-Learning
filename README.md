# Research Project: Detecting Atrial Fibrillation Patients during Sinus Rhythm using Deep Learning

This repository contains the code used in a research project during my master's program.


## Abstract
Atrial Fibrillation (AF) is one of the most common arrhythmias that can lead to a stroke and heart failure when untreated.
Therefore, automatically detecting AF in its early paroxysmal stages can initiate medical treatment to prevent AF to progress in its more frequent stages.
A deep learning approach was applied to detect AF in Sinus Rhythm (SR) Electrocardiograms (ECGs) and was trained with the the Computers in Cardiology Challenge 2001 (CinC 2001) dataset.
To lower the inﬂuence of the size of the dataset, a transfer learning approach was investigated by pretraining the deep learning model with the the PhysioNet/Computing in Cardiology Challenge 2021 (CinC 2021) dataset to estimate a subject’s age from SR ECGs.
The pretrained model for age estimation produced a Mean Absolute Error (MAE) of 11.50 years on a separate test data split.
The model trained without transfer learning achieved an accuracy of 53.00 % and the model trained with transfer learning achieved an accuracy of 53.52 % on a separate test data slit.
In a further experiment including less recordings per subject in the training data set the model produced an accuracy of 62.07 %.
The results from the age estimation model showed that the chosen deep learning approach is able to learn subtle featuresfrom SR ECGs recordings.
However, the models could not reliably distinguish between AF and non-AF SR ECGs.
The size of the CinC 2001 dataset was identiﬁed as the main limitation of the approach.