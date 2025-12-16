# Just in Reach!

Modeling player reachability to better analyze passing opportunities and player performance.

## Overview

The goal of this project is to develop a novel approach for evaluating raw passing outcomes using a player reachability model. The model estimates the probability that a player can reach any location on the field given their observed motion history and a specified time horizon corresponding to ball airtime. This repository contains the code required to build the model, process tracking data, compute play-level metrics, and generate data-guided visualizations.

## Project Structure

### Python Modules

- `model.py`  
  Defines the machine learning model used to compute player reachability heatmaps.

- `play_class.py`  
  Defines the `Play` class, which stores model inputs, outputs, and play-level metadata.

- `helper_functions.py`  
  Utilities for data loading, preprocessing, and dictionary construction for tracking and supplemental data.

- `visualization_functions.py`  
  Visualization utilities for field animations, reachability heatmaps, open score plots, and video composition.

- `animate_play.py`  
  Script for animating individual plays using tracking data and model outputs.

### Jupyter Notebooks

- `model_train.ipynb`  
  End-to-end workflow for model training, validation, and evaluation.

- `play_visualizer.ipynb`  
  Interactive notebook for exploring individual plays and generating visual outputs.

### Data & Outputs

- `data/`  
  Raw player tracking data, play-level metadata, and supplemental datasets used for modeling and visualization.

- `model_params/`  
  Saved model weights, configuration files, and training artifacts.

- `metrics/`  
  Exported processed datasets and evaluation metrics.

- `animations/`  
  Generated play-level field animations exported as video files.

- `plots/`  
  Generated open score vs. time plots, including both animated and static figures.

- `outputs/`  
  Final composed videos combining broadcast highlights, field animations, and analytical plots.

## Authors
- Corwin Phung
  
- Derrick Ushko

*M.S. Engineering (Data Science) students at UCLA*

## Acknowledgements

This project uses player tracking data provided by the **NFL Big Data Bowl**.
