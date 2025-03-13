# Improvements

To try:

- [ ] Focal loss (try different combinations)
- [ ] Try dropout
- [ ] Performance for CPU:
  - [ ] Look at compiling
  - [ ] Make a smaller model
  - [ ] Use a smaller dataset?
  - [ ] Use a smaller image size?
- [ ] Pretrain models on two different datasets (one with many buildings, one with few buildings)
- [ ] gebruiksdoel: if NULL then it's probably a small building -> not always true but if we add an area threshold like 30 m², it could be good
- [ ] Evaluation of the results (Accuracy, F1 score, etc.)

Already tried:

- [x] Use residual connections ==> Seems worse than concatenating and using convolutions
- [x] ResNet ==> Found U-Net instead and took inspiration from it
- [x] Try with SGD instead of Adam ==> Adam seems better
- [x] Look at the final activation function ==> Removing it helped for correct loss computation
