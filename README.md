# **conditional denoising diffusion probabilistic model (CDDPM)**
## Running screenshots show
- **Workflow of the proposed method**
  <img src="img/Workflow of the proposed method.jpg" width="400" />
***
## Paper Support
- Original information: A Missing Well-Logs Imputation Method Based on Conditional Denoising Diffusion Probabilistic Models.
- Recruitment Journal: SPE JOURNAL
- Original DOI: https://doi.org/10.2118/219452-PA
***
## Description of the project
This GitHub repository is for the work "A Missing Well-Logs Imputation Method Based on Conditional Denoising Diffusion Probabilistic Models" published in the SPE Journal. This repository aims to help readers utilize modern generative models to solve the missing well-logs problem in petroleum engineering. The repository contains code that processes the data and some parts of the model components. The design of the denoising model is not included in this repository, but the code for this denoising model is available upon request.
***
## Functions of the project
Datapreprocessing.ipynb is used for preprocessing the data. Users need to download the original data from https://www.kgs.ku.edu/PRS/petroDB.html and save the obtained data in the corresponding folder.
main_train.py is used to train the diffusion model. 
Model/diffusion is for the forward and reverse process for the diffusion process.
ProcessedDataset/WellLogDataLoader.py is for the dataloader construction. 
***
## The operating environment of the project
-	Python
-	Pytorch_lighning
-	Numpy
-	Pandas
-	Torch
-	Scikit-learn
***
