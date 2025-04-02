# Improvements

To try:

- [ ] Focal loss (try different combinations)
- [ ] Try dropout
- [ ] Pretrain models on two different datasets (one with many buildings, one with few buildings)

Already tried/done:

- [x] Use residual connections ==> Seems worse than concatenating and using convolutions
- [x] ResNet ==> Found U-Net instead and took inspiration from it
- [x] Try with SGD instead of Adam ==> Adam seems better
- [x] Look at the final activation function ==> Removing it helped for correct loss computation
- [x] Allow selection of the area ==> Now possible with a bounding box
- [x] Evaluation of the results (Accuracy, F1 score, etc.) ==> Overall and image per image evaluation
- [x] Give higher level control over the model
- [x] gebruiksdoel: if NULL then it's probably a small building ==> not always true but added a filter to remove on condition (gebruiksdoel is NULL) and (area < 30 m²>)
- [x] Improve visualisation ==> Used the visualisation for the tree segmentation project
- [x] Give the possibility to use a pretrained model ==> Now possible

Cancelled:

- [-] Performance for CPU ==> Use Google Colab instead
  - [-] Look at compiling
  - [-] Make a smaller model
  - [-] Use a smaller dataset?
  - [-] Use a smaller image size?
