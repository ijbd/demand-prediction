## Overview

The goal of this project is to build an hourly, balancing authority-scale demand regression model agains local temperature to predict long term electricity demand

## Regression

Our approach is to separate the overarching model into two steps. First, we predict *daily* peak temperature from daily peak demand using an artificial neural network. Next, we downscale daily values to hourly based on historical demand profiles (method TBD).
