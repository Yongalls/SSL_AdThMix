# Adaptive Threshold MixMatch
With a great amount of labeled data, the role of unlabeled samples in semi-supervised learning often becomes insignificant. Because proper usage of unlabeled data is crucial for semi-supervised learning to be applicable in real-world problems, we have to refrain from training steps to be dominated by labeled data. To utilize unlabeled data more elegantly, we devised an advanced scheduling algorithm for managing pseudo labels in intermediate steps. Our algorithm filters the pseudo label by its confidence depending on the model’s current learning phase. As a result, it increased the model’s performance as well as adding more robustness where there are few labeled data. We gained ~% additional top 1 accuracy compared to MixMatch. Our research also can be combined with other SSL methods, since it’s compact and well applicable to other domains. 

## MixMatch_basic
Contains baseline codes for MixMatch. 

## Fixed_Threshold
Contains MixMatch with using fixed threshold for pseudo labels. 

## Adaptive_Threshold
Contains Adoptive Threshold version of MixMatch, which is the final result of our research.

## Experiment_codes
Contains exact codes which is used for our data in presentation and report. 
Specific description about session number and experiments are in "Experiment Data.pdf"

## Etc 
This folder contains implemented SSL algorithms from various papers but not used in final version
Currently, we've implemented below models and functions for our training data.

 - MixMatch
 - Mean Teacher
 - FixMatch
 - SimCLR
 - RotNet
 - RandAugment

Also, it contains some baseline code fragments and some integrated versions of above codes that have been used in intermeditate steps of our project

