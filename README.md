# Tidepool Data Science Modeling

[![Build Status](https://travis-ci.com/tidepool-org/data-science-model-tools.svg?branch=master)](https://travis-ci.com/tidepool-org/data-science-model-tools)

# Metabolism Modeling

#### -- Project Status: [Active]

## Project Intro/Objective
The purpose of this project is providing metabolism modeling
tools as a stand alone library for use in other Tidepool
data science projects.

### Methods Used
* Time Series

### Technologies
* Python3

## Project Description
This code was born out of the iCGM risk analysis plan for Loop
FDA submission. Since it will likely be useful to have modeled
carb/insulin/etc. in a variety of contexts it has been refactored
and put in this repo for easier use and testing.

## Needs of this project


## Getting Started

1. Clone this repo (for help see the github [tutorial](https://help.github.com/articles/cloning-a-repository/)).
2. Import the model of interest and run. Example:
```
from src.models.simple_metabolism_model import SimpleMetabolismModel

smm = SimpleMetabolismModel(
        insulin_sensitivity_factor=isf,
        carb_insulin_ratio=cir,
        insulin_model_name="palerm",
        carb_model_name="cescon",
      )

(
    net_change_in_bg_smm,
    t_5min_smm,
    carb_amount_smm,
    insulin_amount_smm,
    iob_5min_smm,
) = smm.run(num_hours=8, carb_amount=0.0, insulin_amount=1.0)
```



## Available Models

![Insulin Models](reports/figures/Insulin_Models_Plot_2020-04-24_12:10-PM_v0.1_Simulated.png?raw=True)
![Carb Models](reports/figures/Carb_Models_Plot_2020-04-24_12:10-PM_v0.1_Simulated.png?raw=True)


## Team Members

**[Ed Nykaza](https://github.com/ed-nykaza)**

**[Cameron Summers](https://github.com/scaubrey)**

**[Jason Meno](https://github.com/jameno)**


