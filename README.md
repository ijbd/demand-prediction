Electricity Demand Regression with Machine Learning

## Table of Contents
* [Overview](#overview)
* [Requirements](#requirements)
* [Setup](#setup)
* [File Descriptions](#file-descriptions)

## Overview

The goal of this project is to predict hourly, balancing authority-scale demand from local meteorological conditions (i.e. hourly temperature). 

Our modeling approach is to break the regression into two steps. First, we predict *daily* peak demand from daily peak temperature using an artificial neural network. Next, we downscale daily values to hourly based on historical demand profiles (method TBD).

## Requirements

The following libraries are used in this project:
* **keras-tuner** version: 1.1.2
* **pandas** version: 1.4.2
* **tensorflow** version: 2.8.0
* **sklearn** version:0.0

## Setup

To install the required libraries in a virtual environment:

		pip install keras-tuner==1.1.2
		pip install pandas==1.4.2
		pip install tensorflow==2.8.0
		pip install sklearn

To run the ANN pipeline on the University of Michigan ARC Great Lakes computing cluster, the project directory should be changed in the `default_config.json` to a reasonable location (such Turbo Research Storage). The configuration file contains a parameter template for future models. It defines data filepaths, the hyperparameter search space, and other pipeline inputs.

## File Descriptions


