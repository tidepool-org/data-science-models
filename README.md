# Tidepool Data Science Modeling

[![Build Status](https://travis-ci.com/tidepool-org/data-science-models.svg?branch=jam/refactor-icgm-sensor-generator)](https://travis-ci.com/tidepool-org/data-science-model-tools)

# Metabolism Modeling

#### -- Project Status: [Active]

## Project Intro/Objective
The purpose of this project is providing metabolism modeling
tools as a stand alone library for use in other Tidepool
data science projects.

### Methods Used
* Time Series
* Pharmacokinetic/Pharmacodynamic Modeling
* Physiological Systems Modeling

### Technologies
* Python3

## Project Description
This code was born out of the iCGM risk analysis plan for Loop
FDA submission. Since it will likely be useful to have modeled
carb/insulin/etc. in a variety of contexts it has been refactored
and put in this repo for easier use and testing.

The library provides validated mathematical models for:

* Insulin absorption and action (Palerm model)
* Carbohydrate absorption and glucose appearance (Cescon model)
* Physical activity effects on glucose (Aiello model)
* Type 2 diabetes pancreatic insulin production (Type2Insulin model)

## Needs of this project


## Getting Started

1. Clone this repo (for help see the github [tutorial](https://help.github.com/articles/cloning-a-repository/)).
2. Import the model of interest and run. 

### Basic Usage Example
```from tidepool_data_science_models.models.simple_metabolism_model import SimpleMetabolismModel

# Initialize model with patient parameters
smm = SimpleMetabolismModel(
    insulin_sensitivity_factor=50,  # mg/dL per unit insulin
    carb_insulin_ratio=12,          # grams carbs per unit insulin
    insulin_model_name="palerm",
    carb_model_name="cescon",
    pa_model_name="aiello"          # Physical activity model
)

# Simulate insulin and carb response
net_change_in_bg, t_5min, insulin_amount, iob_5min = smm.run(
    num_hours=8, 
    carb_amount=45.0,     # grams
    insulin_amount=3.5,   # units
    heart_rate=0          # beats per minute (0 = no activity)
)
```
### Physical Activity Modeling Example
```# Model the glucose-lowering effect of physical activity
smm = SimpleMetabolismModel(
    insulin_sensitivity_factor=50,
    carb_insulin_ratio=12,
    insulin_model_name="palerm",
    carb_model_name="cescon",
    pa_model_name="aiello",
    w_hr=1.0,             # Heart rate coefficient 
    a=-0.002462,          # Activity coefficient
    tau=0.9989,           # Time constant
    n=28                  # Delay parameter (10-second steps)
)

# Simulate moderate exercise (120 bpm heart rate)
net_change_in_bg, t_5min, insulin_amount, iob_5min = smm.run(
    num_hours=4,
    carb_amount=0,
    insulin_amount=0,
    heart_rate=120        # Moderate exercise heart rate
)
```
### Type 2 Diabetes Modeling Example
```# Model pancreatic insulin production in Type 2 diabetes
smm = SimpleMetabolismModel(
    insulin_sensitivity_factor=30,     # Reduced ISF typical in T2D
    carb_insulin_ratio=8,              # Lower CIR typical in T2D
    glucose_sensitivity_factor=0.01,   # Pancreatic insulin response
    basal_blood_glucose=100,           # Target glucose level
    insulin_production_rate=0.02,      # Basal insulin production
    type2_insulin_model_name="t2_insulin"
)

# Simulate pancreatic response to elevated glucose
net_change_in_bg, t_5min, insulin_amount, iob_5min, endogenous_insulin = smm.run(
    num_hours=6,
    carb_amount=60,       # Large carb load
    insulin_amount=0,     # No external insulin
    blood_glucose=180     # Elevated starting glucose
)
```
## Available Models
### Insulin Model
**Palerm Model:** Biexponential insulin absorption model validated for rapid-acting insulin analogs
* Based on pharmacokinetic studies
* Used in FDA Tidepool Loop submission
* Configurable for different insulin types
### Carbohydrate Model
**Cescon Model:** Carbohydrate absorption with configurable absorption time
* Models delayed gastric emptying
* Supports absorption times from 25 minutes to ~10 hours
* Accounts for meal composition effects
### Physical Activity Models
**Aiello Model:** Heart rate-based glucose reduction model
* Models acute glucose-lowering effects of exercise
* Based on heart rate as proxy for exercise intensity
* Includes physiological delays and time constants
* Heart rate must be between 72-200 bpm
### Type 2 Diabetes Models
**Type 2 Insulin Model:** Pancreatic insulin production model
* Glucose-dependent insulin secretion
* Configurable basal insulin production
* Models residual beta-cell function
* Use with appropriate glucose sensitivity factors

## Installation Requirements
```pip install numpy```
## Testing 
Run the test suite to ensure model integrity:
```python -m pytest tests/ -v```
## References
References

* Palerm, C. C. (2009). Physiologic insulin delivery with insulin feedback: A control systems perspective
* Cescon, M., et al. (2009). Linear modeling and prediction in diabetes physiology
* Aiello, E. M., et al. (2020). Physical activity and exercise modeling for glucose regulation
* Hovorka, R., et al. (2004). Nonlinear model predictive control of glucose concentration in subjects with type 1 diabetes
