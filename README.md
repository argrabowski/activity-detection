# Human Activity Detection

This project focuses on human activity detection using mobile sensor data. It includes MATLAB scripts for processing, analyzing, and visualizing sensor data, as well as machine learning models for activity prediction.

## Overview

- **Data Processing:** MATLAB scripts for loading raw sensor data, converting it into tables, and saving the data into MAT files.

- **Feature Extraction:** MATLAB functions for extracting various features from sensor data, such as mean, standard deviation, entropy, etc.

- **Activity Prediction:** MATLAB scripts containing machine learning models for predicting human activities based on sensor data.

- **Visualization:** MATLAB scripts for visualizing sensor data and model predictions.

## Usage

1. **Data Preparation:**
   - Run `saveSensorDataAsMATFiles.m` to load and save raw sensor data.
  
2. **Feature Extraction:**
   - Utilize functions like `Wmean`, `Wstd`, `Wentropy`, etc., for feature extraction.
   - Example: `features = Wmean(rawSensorData.total_acc_x);`

3. **Activity Prediction:**
   - Load a pre-trained model or train a new one using your processed data.
   - Use scripts like `plotActivityResults.m` or `plotModelResults.m` for model evaluation and visualization.

4. **Visualization:**
   - Visualize raw sensor data using `plotRawSensorData.m`.
   - Explore predictions and actual activities with `plotActivityResults.m` or `plotModelResults.m`.

## Dependencies

- MATLAB R2022a or later
