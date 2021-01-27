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

| Fold ID | Error @ [0, T] | AUCCESS | download
| :-----: | :------------: | :-----: | :------:
| 0       | 3.526          |  83.50  | [pred](https://drive.google.com/file/d/1h-zEsOM0FyPog1Urh5slnpKuKXxG16bd/view?usp=sharing) \| [cls](https://drive.google.com/file/d/1kRiuzEHU2t4K2W_rp2jo5KZr6Lyhq78B/view?usp=sharing)
| 1       | 3.475          |  83.77  | [pred](https://drive.google.com/file/d/1y5Db5RZvBSv2t73jjCOxK5Uo0c17_-kA/view?usp=sharing) \| [cls](https://drive.google.com/file/d/1o2Kv9kxLsyHbp3r3vNch9DSxx5YdCBQQ/view?usp=sharing)
| 2       | 3.765          |  82.88  | [pred](https://drive.google.com/file/d/1txkVHkS1PIRXMfG9QLbw3wVhs6gwkoMQ/view?usp=sharing) \| [cls](https://drive.google.com/file/d/1BJjNypaEz1ooeOcJCaXOqrDp-Ykw0QeP/view?usp=sharing)
| 3       | 3.940          |  81.70  | [pred](https://drive.google.com/file/d/1yHvWAbrYTXo-G6K_hW131iVmU-Te2oRY/view?usp=sharing) \| [cls](https://drive.google.com/file/d/1knwqC0OzJu-TA7bsfM0KHa34YmDo5wrC/view?usp=sharing)
| 4       | 3.771          |  79.91  | [pred](https://drive.google.com/file/d/1l_7NICMVk6HwW8g52D12tDUfT0xcQSnq/view?usp=sharing) \| [cls](https://drive.google.com/file/d/1EhFzekEdsXr6UoLtsFUtugOW3bqCiccG/view?usp=sharing)
| 5       | 3.713          |  82.55  | [pred](https://drive.google.com/file/d/1sipUIrkXj4wBJBRa--i12VJn2kSfOlEi/view?usp=sharing) \| [cls](https://drive.google.com/file/d/176E-ZqL0fSCg36jfTXWCcOHhj2N0CXa0/view?usp=sharing)
| 6       | 3.790          |  82.78  | [pred](https://drive.google.com/file/d/1hyia88X8wJna28xCDMpCYBSC3Yhu-mlR/view?usp=sharing) \| [cls](https://drive.google.com/file/d/1e8Oa3_w2YbTzbFD2AogHWCCbM198lMzm/view?usp=sharing)
| 7       | 4.002          |  82.48  | [pred](https://drive.google.com/file/d/1gKsTySIMO5zfneBLhvXZFdUhxuk-ooLD/view?usp=sharing) \| [cls](https://drive.google.com/file/d/1PZg9I8DDTGAqzsWKUU-T7BISHFyaCzAf/view?usp=sharing)
| 8       | 4.477          |  82.20  | [pred](https://drive.google.com/file/d/1CWw1Ax7AtZtM_mKCENx9mh709ZobAP_I/view?usp=sharing) \| [cls](https://drive.google.com/file/d/1tMXewWW1coMEYwirds8G3ZEk8wTqqykj/view?usp=sharing)
| 9       | 4.475          |  82.74  | [pred](https://drive.google.com/file/d/1oaXHKqlpPl7slOy0KEUHPXW5TEaHOuRn/view?usp=sharing) \| [cls](https://drive.google.com/file/d/1Kmja0wYlCyH1fAIsw6mipTZS6ZCUQSWV/view?usp=sharing)
### Cross-Task Generalization

| Fold ID | Error @ [0, T] | AUCCESS | download
| :-----: | :------------: | :-----: | :------:
| 0       | 9.053          |  60.22  | [pred](https://drive.google.com/file/d/1aL-a8oTXXieUC2fnQliWEBctgknJOhmN/view?usp=sharing) \| [cls](https://drive.google.com/file/d/1eRRHk1Nm1knFLEXELvMsddt7xgXWSXXi/view?usp=sharing)
| 1       | 34.252         |  32.26  | [pred](https://drive.google.com/file/d/1_IH9ZxHiephHY8xFxH_jQz7_gXcnErZ8/view?usp=sharing) \| [cls](https://drive.google.com/file/d/1ZQonl9a_D5AmGXwe5ZmYyE_UXbfZFn2Z/view?usp=sharing)
| 2       | 17.290         |  54.37  | [pred](https://drive.google.com/file/d/1ABJxthxCaZ0v-iFiiuBwPf_hQJ5mm4dx/view?usp=sharing) \| [cls](https://drive.google.com/file/d/1X_ri051zI2yLl9KXy6_5q1UpJlxwDuXp/view?usp=sharing)
| 3       | 18.505         |  26.70  | [pred](https://drive.google.com/file/d/1nc_4OTQMZdD4n6yR6xsgbey0tRqPTcbj/view?usp=sharing) \| [cls](https://drive.google.com/file/d/1KaM3yi-qcsAfB60OyonWr34VCUIlS4-z/view?usp=sharing)
| 4       | 13.283         |  32.92  | [pred](https://drive.google.com/file/d/1MxY_JvFYz6cWZEo7rN65VPClfXca3rJM/view?usp=sharing) \| [cls](https://drive.google.com/file/d/1_d-qyrXoXpc8pqZgZ76N33pAojgiVjl3/view?usp=sharing)
| 5       | 26.303         |  51.47  | [pred](https://drive.google.com/file/d/1WZboo1jqZk0jVhKSjX9ndGFrD98KXCuW/view?usp=sharing) \| [cls](https://drive.google.com/file/d/1fL74NDh7vfX9rhGXgORvarUvNo4zuGtv/view?usp=sharing)
| 6       | 23.254         |  40.57  | [pred](https://drive.google.com/file/d/1YSs1vN39hRKRGiRaNrKR1fOTmtbn3IBW/view?usp=sharing) \| [cls](https://drive.google.com/file/d/1Di1kdF34ZjDQLs3RgyuMsMUucNHRaiO_/view?usp=sharing)
| 7       | 25.642         |  45.26  | [pred](https://drive.google.com/file/d/1my5WmE03sXzG6Ay5FFO6lD5v8_TvY2V2/view?usp=sharing) \| [cls](https://drive.google.com/file/d/1ybwPzQNBG47K4vGMGZI3JKg3R5ZI3ufp/view?usp=sharing)
| 8       | 25.176         |  28.21  | [pred](https://drive.google.com/file/d/1YKTq42i4tgdalPhCU9GLy67nwrdS4w7Z/view?usp=sharing) \| [cls](https://drive.google.com/file/d/1D4lwpvkAzMk7G6LXWJthwsDTwtmy3A9q/view?usp=sharing)
| 9       | 25.530         |  37.14  | [pred](https://drive.google.com/file/d/19srvZKxRf-P8IPdH7DpsydbWwFJDVbTY/view?usp=sharing) \| [cls](https://drive.google.com/file/d/1CrWJ89RcGSLpcKZeupbkcOcfWRxXK6Jx/view?usp=sharing)
