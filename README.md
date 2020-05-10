# Adaptive Threshold MixMatch
With a great amount of labeled data, the role of unlabeled samples in semi-supervised learning often becomes insignificant. Because proper usage of unlabeled data is crucial for semi-supervised learning to be applicable in real-world problems, we have to refrain from training steps to be dominated by labeled data. To utilize unlabeled data more elegantly, we devised an advanced scheduling algorithm for managing pseudo labels in intermediate steps. Our algorithm filters the pseudo label by its confidence depending on the model’s current learning phase. As a result, it increased the model’s performance as well as adding more robustness where there are few labeled data. We gained 1.17% additional top 1 accuracy compared to MixMatch. Our research also can be combined with other SSL methods, since it’s compact and well applicable to other domains. 

## Overall Structure
Our GitHub repo is divided into 3 parts. Please keep in mind that all codes in here are for NSML environment, not in local machines.  
First, MixMatch_basic, Fixed_Threshold, Adaptive_Threshold contains finalized version of each source code that we used in our environments. These codes have proper argument settings that can be used to further research. Detailed instructions can be found in each directory. 
<br><br>
Second, Experiment_codes contains exact source codes and some informations about experiements that was conducted in NSML. Each folder has session name on it, and exact files and configurations are inside each folder. This is to reproduce results that are in our presentation and paper. However, due to randomization in validation, exact accuracy might differ a bit. 
<br><br>
Finally, etc folder contains remaining codes that are used in our research, but are not related with our final paper. Some implementation of SSL papers, code fragments, intermediate version of our implementation might be included in the folder. 
<br><br>

## MixMatch_basic
Contains baseline codes for MixMatch. 

## Fixed_Threshold
Contains MixMatch with using fixed threshold for pseudo labels. 

## Adaptive_Threshold
Contains Adoptive Threshold version of MixMatch, which is the final result of our research.

## Experiment_codes
Contains exact codes which is used for our data in presentation and report. 
Specific description about session number and experiments are in "Experiment_codes/Experiment Data.pdf"

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

