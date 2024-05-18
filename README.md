# Chinese numbers classification
This project concerns a comprehensive workflow for classifying images of Chinese numbers using various neural network architectures, including fully connected neural networks (FCNNs) and convolutional neural networks (CNNs).
The dataset can be found on [Kaggle](https://www.kaggle.com/) at this [link](https://www.kaggle.com/datasets/gpreda/chinese-mnist)

### Models summary and results

The dataset is split into three subsets:
- training set (64%)
- validation set (20%)
- test set (16%)

The following callbacks are implemented for all the models:
- **Learning Rate Scheduler** dynamically adjusts the learning rate throughout training, allowing for smoother convergence by gradually decreasing the learning rate over epochs. The learning rate decay formula used is given by: `learning_rate = initial_learning_rate * (0.99 ** (epoch + NO_EPOCHS))` where `initial_learning_rate` is the initial learning rate, `epoch` is the current epoch number, and `NO_EPOCHS` is the total number of epochs. This formula ensures that the learning rate decreases by a factor of 0.99 after each epoch, promoting stable training progress.
- **Early Stopping** monitors the validation loss and halts training if no improvement is observed after a specified number of epochs, preventing the model from overfitting to the training data.
- **Model Checkpoint** callback saves the best-performing model during training based on a chosen metric, ensuring that the model with the highest validation accuracy is retained for future use.

These callbacks collectively contribute to optimizing model performance and generalization while managing training dynamics effectively.

The model results are reported in the following, with figures of the training progress and accuracy score on the test set.

- **Simple FCNN** (Test accuracy: 69.37%): Basic fully connected network for initial classification.
  ![fcnn_1](https://github.com/Andre1411/chinese_classification/assets/107708093/d1f04432-27c1-4316-bbd2-f464ea6382c7)
- **Advanced FCNN** (Test accuracy: 84.30%): Adds regularization, dropout, and batch normalization to improve performance.
  ![fcnn_2](https://github.com/Andre1411/chinese_classification/assets/107708093/f81bab93-13e4-473a-ab62-a0b8b05e7724)
- **Simple CNN** (Test accuracy: 97.33%): Basic convolutional network for spatial feature extraction.
![covnn_1](https://github.com/Andre1411/chinese_classification/assets/107708093/0a65ed80-1650-4f5f-98ad-7b6dad54988b)
- **Advanced CNN** (Test accuracy: 97.53%): Enhanced with regularization, dropout, and batch normalization.
![convnn_2](https://github.com/Andre1411/chinese_classification/assets/107708093/1948d70b-229b-48ba-8fcb-faaa0dfc62ac)
- **CNN with Attention** (Test accuracy: 98.83%): Uses SE blocks for better feature focusing.
![convnn_3](https://github.com/Andre1411/chinese_classification/assets/107708093/7b1eff21-58c9-4298-8b22-09314d74821e)

### Conclusion
This workflow demonstrates different approaches to image classification using neural networks, emphasizing the importance of model architecture and regularization techniques to achieve better performance.

