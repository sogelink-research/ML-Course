# Improvements

To try:

- Nothing

Already tried/done:

- [x] Use residual connections ==> Seems worse than concatenating and using convolutions
- [x] ResNet ==> Found U-Net instead and took inspiration from it
- [x] Try with SGD instead of Adam ==> Adam seems better
- [x] Look at the final activation function ==> Removing it helped for correct loss computation
- [x] Allow selection of the area ==> Now possible with a bounding box
- [x] Evaluation of the results (Accuracy, F1 score, etc.) ==> Overall and image per image evaluation
- [x] Give higher level control over the model
- [x] gebruiksdoel: if NULL then it's probably a small building ==> not always true but added a filter to remove on condition (gebruiksdoel is NULL) and (area < 30 m²)
- [x] Improve visualisation ==> Used the visualisation for the tree segmentation project
- [x] Give the possibility to use a pretrained model ==> Now possible
- [x] Add the metrics to the visualisation during training
- [x] Add explanations for each parameter ==> Added in the notebook
- [x] Look whether the computation of the loss is accurate when averaging ==> Seems alright
- [x] Fully allow for interrupting the training but have everything written on disk
- [x] Better naming for the output folders with the input folders names ==> Used the data folder name and also harmonised "metrics" and "output"
- [x] Provide a website to decide on areas
- [x] Give the possibility to use multiple areas for training
- [x] Create a final test dataset for everyone to evaluate their final model ==> Selected 25 areas of 1 km²

Cancelled:

- [-] Performance for CPU ==> Use Google Colab instead
  - [-] Look at compiling
  - [-] Make a smaller model
  - [-] Use a smaller dataset?
  - [-] Use a smaller image size?
- [-] Try dropout
- [-] Focal loss (try different combinations)
- [ ] Pretrain models on two different datasets (one with many buildings, one with few buildings)
