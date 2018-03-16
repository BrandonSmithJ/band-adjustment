# Spectral Band Adjustment 

### About
This repository contains source code for the paper:

<i>["Spectral Band Adjustments for Remote Sensing Reflectance Spectra in Coastal/Inland Waters"](https://www.osapublishing.org/oe/abstract.cfm?uri=oe-25-23-28650)</i>.
N. Pahlevan, B. Smith, C. Binding, D. M. O'Donnell. Opt. Express 25, 28650-28667 (2017).

Data used in the paper to train the networks can be found [here](https://www.dropbox.com/s/h7ftg566sj8q1ll/Nima_et_al_Optics_Express_2017.zip?dl=0).

<br>

### Usage
To create the heatmap comparison, run
`python3 heatmap.py`

If the predictions were not yet generated, the script will create all necessary prediction files via the supplied network builds; a summary of the result statistics can then be found in the Results folder. 

Once all band adjustments are made, the resulting image should appear as:

<img src="Results/heatmap.png?raw=true" height=396 width=858></img>

<br> 

A command line interface is also available via cli.py:
```
$ python3 cli.py -h
usage: cli.py [-h] [-s SOURCE] [-t TARGET] [--filename FILENAME]
              [--datadir DATADIR] [--preddir PREDDIR] [--builddir BUILDDIR]
              [--gridsearch GRIDSEARCH] [--trainfmt TRAINFMT]
              [--testfmt TESTFMT]

optional arguments:
  -h, --help            show this help message and exit
  -s SOURCE, --source SOURCE
                        Source sensor [VI, MSI, OLI, OLCI, AER]
  -t TARGET, --target TARGET
                        Target sensor [VI, MSI, OLI, OLCI, AER]
  --filename FILENAME   Name of file to convert
  --datadir DATADIR     Directory data is located in
  --preddir PREDDIR     Directory predictions should go in
  --builddir BUILDDIR   Directory DNN build is located in
  --gridsearch GRIDSEARCH
                        Flag to turn on hyperparameter gridsearch
  --trainfmt TRAINFMT   Format of training file(s), with %s identifying the
                        source/target name (i.e. Rrs_LUT_%s)
  --testfmt TESTFMT     Format of file(s) to be converted, with %s identifying
                        the source/target name (i.e. Rrs_LUT_%s)

```

For example, to create an adjustment from AERONET -> VIIRS using the supplied in situ file, you can run:

`python3 cli.py --source AER --target VI --filename "Data/In Situ/Rrs_insitu_AER"`

which then creates the file "Predictions/AER_to_VI_DNN.csv". 

<br>

To train a new network, use the --datadir flag to point towards the training data location, and --trainfmt to specify the file name format.

For example:

`python3 cli.py --source AER --target VI --datadir "Data" --trainfmt "Rrs_LUT_%s"`

The "%s" represents the sensor abbreviation, which in the above case would mean there should be two files available:

./Data/Rrs_LUT_AER

./Data/Rrs_LUT_VI

The models will then be created which learn the mapping between bands of Rrs_LUT_AER and Rrs_LUT_VI.