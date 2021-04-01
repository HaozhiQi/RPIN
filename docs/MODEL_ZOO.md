# Model Zoo

This file documents a collection of models reported in our paper: [Learning Long-term Visual Dynamics with Region Proposal Interaction Networks](https://arxiv.org/abs/2008.02265). Links to the pretrained models are provided as well.


## Prediction

We use T=5 for PHYRE, T=15 for ShapeStacks, and T=20 for training SimB/RealB:

| Dataset         | Error @ [0, T] | Error @ [T, 2 * T] | download
| :-------------: | :------------: | :----------------: | :---:
| PHYRE (Within)  |      1.308     |       11.060       | [model](https://drive.google.com/file/d/10g0U00-pv2dRH2PjfrSi1jlnF4OewrX4/view?usp=sharing)
| PHYRE (Cross)   |      3.262     |       15.185       | [model](https://drive.google.com/file/d/1WLA5w3944Cz2CAmFpZK6uJv_V4UmTEfV/view?usp=sharing)
| ShapeStacks (3) |      1.027     |        4.728       | [model](https://drive.google.com/file/d/1VufPAnn2uSeAe1I9KA-NctpvGTjuLscX/view?usp=sharing)
| ShapeStacks (4) |      5.753     |       23.459       | [model](https://drive.google.com/file/d/1VufPAnn2uSeAe1I9KA-NctpvGTjuLscX/view?usp=sharing)
| RealB           |      0.296     |        2.336       | [model](https://drive.google.com/file/d/1w8Id8UYfQcYhc3nh2_I6Qnkt56Qr29xS/view?usp=sharing)
| SimB            |      2.55      |        25.77       | [model](https://drive.google.com/file/d/1vbJWlLCdT6GqTqry61TB3eEOGtDg9q-J/view?usp=sharing)

## Planning

### Within-Task Generalization

| Fold ID | AUCCESS | download
| :-----: | :-----: | :------:
| 0 | 85.49 | [model](https://drive.google.com/file/d/1ho6ndZH7BlwNfAyOSqVlS__zYlgpZMhy/view?usp=sharing)
| 1 | 86.57 | [model](https://drive.google.com/file/d/1_qRWsHam9EmbEBR1Aj0lm5CGec13dFhg/view?usp=sharing)
| 2 | 85.58 | [model](https://drive.google.com/file/d/1EpLzuKew4r4GnaOU8QsOxdUO61tAKE1d/view?usp=sharing)
| 3 | 84.11 | [model](https://drive.google.com/file/d/1YGQuoGQ68z0ZpIOFm0yqYTjVQdqQKjIf/view?usp=sharing) 
| 4 | 85.30 | [model](https://drive.google.com/file/d/1Fhg0FpNAV11phBNh6Z_J5cuVgN7sfI-D/view?usp=sharing) 
| 5 | 85.18 | [model](https://drive.google.com/file/d/1QLZcx8XL9JD08mObWEwQ2E0PeOravE-E/view?usp=sharing)
| 6 | 84.78 | [model](https://drive.google.com/file/d/1Sa5LpUnxlr1bTtY0E8CsleYxe-XFrt8k/view?usp=sharing)
| 7 | 84.32 | [model](https://drive.google.com/file/d/1p9ktbmydE5lJ5fWtES6q0uo5oJA7GmiY/view?usp=sharing) 
| 8 | 85.71 | [model](https://drive.google.com/file/d/1nN1qdapp4Ms_UefHR5jytDCQMrLKX8pV/view?usp=sharing)
| 9 | 85.17 | [model](https://drive.google.com/file/d/1haxSvWJPT-36JxRKM7aEBBRcWQOPEtG7/view?usp=sharing)

### Cross-Task Generalization

| Fold ID | AUCCESS | download
| :-----: | :-----: | :------:
| 0 | 50.86 | [model](https://drive.google.com/file/d/1fBlPESes4js0vgu6Xv3GGcILlaPUP1QV/view?usp=sharing)
| 1 | 36.58 | [model](https://drive.google.com/file/d/1KJbDCkujWvd_NHmHjyjgET__gBk9CGgC/view?usp=sharing)
| 2 | 55.44 | [model](https://drive.google.com/file/d/1ME8p3Bfvk71UD49oIa8F2f71wGp6_Ncw/view?usp=sharing)
| 3 | 38.34 | [model](https://drive.google.com/file/d/1Hat4GL8vwFW-XQEecPTwq2ApreegVOxt/view?usp=sharing) 
| 4 | 37.11 | [model](https://drive.google.com/file/d/1tUd0Gc3ASbQgcT5wAhG5F-rM8t_hjGRL/view?usp=sharing) 
| 5 | 47.23 | [model](https://drive.google.com/file/d/1681pV7zP2emnOwOSM58R7IAMS_kKkVob/view?usp=sharing)
| 6 | 38.23 | [model](https://drive.google.com/file/d/1kGy-2A3UwNdHJDeY5XSQHs3jGg5Yisu8/view?usp=sharing)
| 7 | 47.19 | [model](https://drive.google.com/file/d/1H9FQiq0dqZwrzwIaO98ytOhY_qFqEkvM/view?usp=sharing) 
| 8 | 32.23 | [model](https://drive.google.com/file/d/1qpHRNfxPSMucyf8Os02IySDb0WqbIz7K/view?usp=sharing)
| 9 | 38.76 | [model](https://drive.google.com/file/d/13chI_zNX5QyL6o9OzId2G0xQZTu-xDFm/view?usp=sharing)
