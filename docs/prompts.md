# Course Project

This is a course project focusing on machine learning in computer vision.

## The General Idea

I got the idea from the paper [94% on CIFAR-10 in 3.29 Seconds on a Single GPU](2404.00498v2.pdf), which aims to accelerate research and reduce the cost of experiments, and introduces training methods for CIFAR-10 which reach 94% accuracy in 3.29 seconds.

The method provided works on a CNN model for the image classification task. Our research wants to know whether these method works for transformer architecture.

## Detail Implementation

The article provide several methods to speed up the trainning:

- Frozen patch-whitening initialization
- Identity (Dirac) initializtion
- Optimization techniques, such as scalebias and Lookahead optimizer
- Alternating flip

For transformer, we can also apply these trainning methods. Build the baseline first, then do the research experiments. Write a paper on your findings and final results, explaining the reason of the result showing.

## File Structure

All project files should be stored in the related folder, keeping the workspace clean and tidy.

- `datasets/`: keep all datasets used in this project.
- `docs/`: documentation files and prompts.
- `experiments/`: python codes written for experiments, as well as its outputs, logs and result records. Saperate codes of different tests in different folders and store the outputs in a folder named `outputs` in the folder for that experiment.
- `external/`: downloaded codebase from other researches, inluding the codebase `external/Cifar10CNN`.
- `reports/`: write final reports in LaTeX in this folder.

Use conda `vision` environment. Feel free to install required packages
