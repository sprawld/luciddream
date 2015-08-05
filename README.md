# Lucid Dream

### Take control of your deepdream

Lucid Dream is a python script for running [Google Deepdream](https://github.com/google/deepdream).
It uses a slightly-altered version of [Bat Country](https://github.com/jrosebr1/bat-country), you can look there for requirements.


Lucid Dream allows full control over dreams, including:

- specify different iterations, scales or layers for each octave 
- zoom (with rotation) towards a target
- save full-sized visualizations of the dream process
- try multiple datasets, layers or guide images

There are 3 datasets included:

- The original [BVLC ImageNet GoogLeNet](https://github.com/BVLC/caffe/tree/master/models/bvlc_googlenet),
- The [MIT Places dataset](http://places.csail.mit.edu/downloadCNN.html),
- The [CompCars dataset](https://gist.github.com/bogger/b90eb88e31cd745525ae).

To run dream(s), use lucid.py:

```sh
python lucid.py -i --image    #input image (required)
                -o --output   #output filename root (required)

				-b --base-model     # list of datasets (default: google)
				                    # acceptable entries: google, place, car
				-l --layer          # list of layers (default: conv2/3x3)
				                    #  can also use /all to try whole set of layers
				-v --vis            # visualization - output images of the dream in progress
				
				-oc --octaves           # number of octaves in dream (default: 4)
				-it --iterations        # number of iterations in dream (default: 20)
									
				-os --octave-scale      # scale of octaves (default: 1.4)
				-ol --octave-layer      # list of layers for each octave
				-oi --octave-iterations # list of number of iterations 
				                        # for each octave 

				
				-d --dreams         # number of dreams (default: 1)
				
				-z --zoom           # scale of zoom (default: 0)
				-c --center         # x,y coordinates of zoom target
									# (default: center of image)
				-r --rotation       # rotation of zoom (default:0)
				
				-g --guide-image    # list of guide images to try
```

**Better documentation to come**
