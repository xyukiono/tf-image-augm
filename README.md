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

