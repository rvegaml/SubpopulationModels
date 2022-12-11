# Subpopulation models for infectious diseases
This repository contains the code to reproduce the results of the paper *Heterogeneous modeling of COVID-19 using latent subpopulations*

## Dictionary-based approaches
The jupyter notebook **Dictionary_Example.ipynb** contains an example on how to build a dictionary using Gaussian dictionaries, and some guidelines in how to construct general dictionaries using any type of curves. It also explains how to use the functions for learning the coefficients associated with the entires of such dictionary.

## Mixture of M-curves approaches
The jupyter notebook **Mixture_Example.ipynb** contains a description on how to use the code for finding a mixture of M-curves. It contains helper functions to compute a mixture of M-Gaussians and a mixture of M-SIR models.

## Modelling task
The jupyter notebook **Modelling_Demo.ipynb** has the code for fitting the dictionary and mixture of M-curves approaches to the data. It is possible to use this code to model the pandemic in the 189 countries available in the [COVID-19 Data Repository by the Center for Systems Science and Engineering (CSSE) at Johns Hopkins University](https://github.com/CSSEGISandData/COVID-19)

## Prediction task
The jupyter notebook **Forecasting_Demo.ipynb** has the code for predicting the number of people infected 1-to-4 weeks in advance. It uses the same dataset as the modelling task.

The file **create_predictions_results.py** contains the code for running the experiment for all the 189 countries available in the dataset.
