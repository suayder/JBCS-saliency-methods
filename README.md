# JBCS-video-saliency

## Experiments

Models references:

- ViNet - https://github.com/samyak0210/ViNet
- TMFI-Net - https://github.com/wusonghe/TMFI-Net


## Dataset

| VIDEO-ID | FRAMES |
| -------- | ------ |
| Jundiai_HSV#Block01-2024-02-28-15-06-34-538 | 1200 |
| Jundiai_HSV#Route01-2024-02-28-14-15-07-770 | 1200 |
| Jundiai_HSV#Route02-2024-02-28-14-50-03-491 | 2090 |
| Santos_HM#Block01-2024-03-07-14-18-22-116 | 9644 |
| SaoPaulo_HC#Route02-2024-03-17-16-17-27-328 | 1200 |
| Total | 15334 |


## evaluation

The individual evaluations can be done inside `saliency/` folder, which will have a notebook computing the mean of the saliency.

This code is a version of the original one: https//github.com/herrlich10/saliency