# NeuraNer: Deep Learning for Named Entity Recognition

This project involves building a deep learning model to tackle the task of Named Entity Recognition, that is the task of identifying and categorizing key information (entities) in text. Some of the most-common entities are person, location, organization, time.

1) I built a Bi-directional LSTM network with the architecture as this: Embeddingdim 100, NumLSTMlayers 1, LSTMhiddendim 256, LSTMDropout 0.33, Linearoutputdim 12 in part-1 of the project. A custom vocabulary was built from the conll dataset words with words which are greater than a threshold getting included in the vocabulary.
2) The previous Bi-LSTM network was supplied with Glove embeddings to add to the vocabulary additionally for improved performance by boosting the accuracy of NER prediction by almost more than 10% in part-2 of the project.

The report in the repo above explains the code to help understand how the model is built and how is it functioning.


 
