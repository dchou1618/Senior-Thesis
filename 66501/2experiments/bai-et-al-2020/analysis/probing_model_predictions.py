import argparse
import numpy as np

# predict_and_save_results_mstgcn - under lib/utils.py

# Pseudocode

# def probe_predictions(model_params, model, train_data, test_data):
#     initialized = model(model_params)
#     initialized.train(train_data)
#     predicted_data = initialized.predict(test_data)
#     Predicted average speeds: predicted_data[:,<detector_num>,2]
#     Plot differences between predicted and actual data
