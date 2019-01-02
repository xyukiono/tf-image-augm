# tf-image-augm

This is a tensorflow implemented data augmentation code set.
You can see the results through notebooks/demo.ipynb

## Photometric augmentations
| original | additive gaussian noise | additive speckle noise |
|:---------|:--------------------|:----------------|
| ![pa_original](/samples/doc/pa_original.jpg) | ![pa_additive_gaussian_noise](/samples/doc/pa_additive_gaussian_noise.jpg) | ![pa_additive_speckle_noise](/samples/doc/pa_additive_speckle_noise.jpg) |
| random brightness | random contrast | random color transform |
| ![pa_random_brightness](/samples/doc/pa_random_brightness.jpg) | ![pa_random_contrast](/samples/doc/pa_random_contrast.jpg) | ![pa_random_color_transform](/samples/doc/pa_random_color_transform.jpg) |
| additive shade | motion blur |   |
| ![pa_additive_shade](/samples/doc/pa_additive_shade.jpg) | ![pa_motion_blur](/samples/doc/pa_motion_blur.jpg) |  |

## Deforming augmentations
| Homographic transform | Euclid transform |
|:---------|:--------------------|
| ![da_homographic](/samples/doc/da_homographic.gif) | ![da_euclid](/samples/doc/da_euclid.gif) |
| Elastic deformation | Random distortion |
| ![da_elastic](/samples/doc/da_elastic.gif) | ![da_distortion](/samples/doc/da_distortion.gif) |
















ParamGenerator (in hyper_params.py) enable you to prepare arbitrary combination of hyper parameters. 
For example, the following codes can generate 20 patterns (gpu_memory will be changed only if sleep_time is equal to 30, and n_depth and n_channel will be changed in sync.)

```python
pg.add_params('sleep_time', [5,10,20,30])
pg.add_params_if('gpu_memory', [0.3, 0.5], cond_key='sleep_time', cond_val=30)
pg.add_params('n_depth', [3,6,9,12])
pg.add_params('n_channel', [64,64,32,32], in_series=True)
```

In order to run a job script, you just type
```
./run.py --gpu=0
``` 
Then the oldest job will be executed with GPU#0 and stored `jobs/done` if it finishes successfully.

When you want to run all jobs back-to-back, you just type the following command
```
./run.py --gpu=0 --mode=monitor
```

It is also possible to run multi process with different gpus by doing something like this.
(Do it from independent terminals such as tmux)
```
./run.py --gpu=0 --mode=monitor
./run.py --gpu=1 --mode=monitor
./run.py --gpu=2 --mode=monitor
```

