# winstars-ai-test-2024

# Data preparation step
- Dataset contains images of ships (or no ships), shape 768 x 768 x 3. 
Too large for kaggle to keep descent amount of images in RAM, so resized it to 128 x 128 x 3
- Train and validation datasets contain 9000 and 1000 images respectively

# Training
- Loss: BCE + Dice_loss
- Other metrics observed: Accuracy, Dice_coefficient, Dice_loss
- Learning: 20 epochs, Adam(lr=0.0005)

# Results
0.56 dice-score on validation data
