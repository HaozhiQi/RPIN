# Model Zoo

This file documents a collection of models reported in our paper: [Learning Long-term Visual Dynamics with Region Proposal Interaction Networks](https://arxiv.org/abs/2008.02265). Links to the pretrained models are provided as well.


## Prediction

We use T=5 for PHYRE, T=15 for ShapeStacks, and T=20 for training SimB/RealB:

| Dataset         | Error @ [0, T] | Error @ [T, 2 * T] | download
| :-------------: | :------------: | :----------------: | :---:
| PHYRE (Within)  |      1.308     |       11.060       | [model](https://drive.google.com/file/d/1UtH6KS9TZMVX01SYOs0faoenQtRoPC8t/view?usp=sharing)
| PHYRE (Cross)   |      3.262     |       15.185       | [model](https://drive.google.com/file/d/1KYxHlX9gWTtSQMF-c2f7CnN0YKXEbvCu/view?usp=sharing)
| ShapeStacks (3) |      1.027     |        4.728       | [model](https://drive.google.com/file/d/1WVOSi30qESQ9i6H7hRYqWyGtReToFHOz/view?usp=sharing)
| ShapeStacks (4) |      5.753     |       23.459       | [model](https://drive.google.com/file/d/1WVOSi30qESQ9i6H7hRYqWyGtReToFHOz/view?usp=sharing)
| RealB           |      0.296     |        2.336       | [model](https://drive.google.com/file/d/1yluVDlEXvI5SNz1vaqIQ2TzrNAEAEDcb/view?usp=sharing)
| SimB            |      2.55      |        25.77       | [model](https://drive.google.com/file/d/1ao3nflNaowDmPIHU9Mw_jUCYDJpUBymj/view?usp=sharing)

## Planning

### Within-Task Generalization

| Fold ID | AUCCESS | download
| :-----: | :-----: | :------:
| 0 | 85.49 | [model](https://drive.google.com/file/d/1YrCDIi0UhlS6_OqSBQqaOk2o8cCnn8Ch/view?usp=sharing)
| 1 | 86.57 | [model](https://drive.google.com/file/d/1mEJn9wevRjqykO-eUTHuMx6h5Ihkx1vR/view?usp=sharing)
| 2 | 85.58 | [model](https://drive.google.com/file/d/1AXDSYaoDUijUvbPbsgY3hgXOeiAwbhWP/view?usp=sharing)
| 3 | 84.11 | [model](https://drive.google.com/file/d/1q05qDNveN6ZVXtPSXI0P_OGbL-VjAY9t/view?usp=sharing) 
| 4 | 85.30 | [model](https://drive.google.com/file/d/1M-GWLNdVSMHbassWT65HguetMFw8jLd-/view?usp=sharing) 
| 5 | 85.18 | [model](https://drive.google.com/file/d/1BsOsByOkclv3UxpDkUSnWH6cJQhgBOp5/view?usp=sharing)
| 6 | 84.78 | [model](https://drive.google.com/file/d/1i0vPG72pPi7-GI_xA8H63s9esqm3OQjd/view?usp=sharing)
| 7 | 84.32 | [model](https://drive.google.com/file/d/1FFgZsT7Qw4u_osq0l1NeSjGHKZXJeSv4/view?usp=sharing) 
| 8 | 85.71 | [model](https://drive.google.com/file/d/194H2TYDqwPJngKx0vP2Grpu0jFtzxKTI/view?usp=sharing)
| 9 | 85.17 | [model](https://drive.google.com/file/d/1QnaFQtpe6aA1QWYxlYPX4jST4oLX_J_C/view?usp=sharing)

### Cross-Task Generalization

| Fold ID | AUCCESS | download
| :-----: | :-----: | :------:
| 0 | 50.86 | [model](https://drive.google.com/file/d/140mQEgWV3sw_AwTA1KrmLqa6bZmOBxs_/view?usp=sharing)
| 1 | 36.58 | [model](https://drive.google.com/file/d/11MRySkbdJwmYYx8CHMHbzukV25H08ou-/view?usp=sharing)
| 2 | 55.44 | [model](https://drive.google.com/file/d/1GhA0bcm3Mt-v2XypwTCoCsS9g3TB-906/view?usp=sharing)
| 3 | 38.34 | [model](https://drive.google.com/file/d/1Hq-BB4g6ZsQ0QBvf8nooCjlvQ2zuqcon/view?usp=sharing) 
| 4 | 37.11 | [model](https://drive.google.com/file/d/1KpTgAcLLlAWjKfWhL_td1BDifDJVay_F/view?usp=sharing) 
| 5 | 47.23 | [model](https://drive.google.com/file/d/1ZEhf4Q-ugZTqGLQGgLS_Ujhp79yizNq2/view?usp=sharing)
| 6 | 38.23 | [model](https://drive.google.com/file/d/1DqLo0eokvvxx70dqeEgHkNTqMSw3EXP9/view?usp=sharing)
| 7 | 47.19 | [model](https://drive.google.com/file/d/1OEjAS99weW_4uqSpbGzOR2_RsAvxzmXN/view?usp=sharing) 
| 8 | 32.23 | [model](https://drive.google.com/file/d/1MTVwYEwW2yG15mrnpRULNkyxBkktkYKT/view?usp=sharing)
| 9 | 38.76 | [model](https://drive.google.com/file/d/1XYiX4jUZObybs4RybQHPq8fDnshqcVvm/view?usp=sharing)
