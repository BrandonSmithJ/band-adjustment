# Spectral Band Adjustment 

### About
This repository contains source code for the paper:

<i>["Spectral Band Adjustments for Remote Sensing Reflectance Spectra in Coastal/Inland Waters"](https://www.osapublishing.org/oe/abstract.cfm?uri=oe-25-23-28650).
N. Pahlevan, B. Smith, C. Binding, D. M. O'Donnell. Opt. Express 25, 28650-28667 (2017).

The data used in the paper to train the networks can be found [here](https://www.dropbox.com/s/h7ftg566sj8q1ll/Nima_et_al_Optics_Express_2017.zip?dl=0).


### Usage
To create the heatmap comparison, run
`python3 heatmap.py`

If the predictions were not yet generated, the script will create all necessary prediction files via the supplied network builds; a summary of the result statistics can then be found in the Results folder. 

Once all band adjustments are made, the resulting image should appear as:
![heatmap.png](Results/heatmap.png?raw=true)


A command line interface is also available via cli.py:
```
$ python3 cli.py -h
usage: cli.py [-h] [-s SOURCE] [-t TARGET] [--filename FILENAME]
              [--datadir DATADIR] [--preddir PREDDIR] [--builddir BUILDDIR]
              [--gridsearch GRIDSEARCH]

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
```

For example, to create an adjustment from AERONET -> VIIRS using the supplied in situ file, you can run:

`python3 cli.py --source AER --target VI --filename "Data/In Situ/Rrs_insitu_AER"`

which then creates the file "Predictions/AER_to_VI_DNN.csv". 
