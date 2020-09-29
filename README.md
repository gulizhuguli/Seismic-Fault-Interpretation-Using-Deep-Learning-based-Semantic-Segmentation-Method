# Seismic-Fault-Interpretation-Using-Deep-Learning-based-Semantic-Segmentation-Method
Seismic fault detection uses a simplified Semantic Segmentation Network(VGG 16) with HDC and ASPP.
This a workflow that uses a convolutional neural network–based method of semantic segmentation to interpret faults by using a small training set. 
The steps to implement this process are as follows：
1. Use the programs in the folder "train_sample_selection" to generate samples.
2. Use F3_hdc+aspp_output and F3_largefov to achieve model training and prediction.
3. Use the programs in the folder "post_processing" to refine the prediction result.
