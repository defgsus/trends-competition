### Yet another [TReNDS Neuroimaging competition](https://www.kaggle.com/c/trends-assessment-prediction) sourcecode

Trying to get among the top hundred, using as few CPU power as possible (so no 4d convolutional networks and such..)

To use this repo, create a virtual python environment (for example with *virtualenv*:)
```bash
virtualenv -p python3 env
source env/bin/activate
``` 

and install requirements
```bash
pip install -r requirements.txt
```

and extract the zip file of the competition to `data-org/` directory.

One tricky part is converting the 4D fmri images to something that fits in a *normal* computer's memory. Use:
```bash
python convert_fmri.py
```
to create a CSV file from reductions of single brain volumes (3D) of each subject. The conversion will use all 
available CPU power but will still take a while. Eventually, a CSV is placed into the `data-proc/` directory.
(See the commented-out code in convert_fmri.py for *configuration*).

To test regressions on the converted data run `python test_rvr.py`. It will split
the competition's training set in half to train and test separately and output the results to a CSV in `statistics/`.

All code is *under development*, so there are the usual lines of commented-out code 
and a lack of comments at other places. 

My current best [submitted](https://www.kaggle.com/c/trends-assessment-prediction/leaderboard) 
*weighted normalized absolute error* is **0.167** using 
Relevance Vector Regression on the morphology data (loading.csv) alone.
