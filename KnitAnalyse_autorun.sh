#!/bin/bash

#### analyse scalar flu on single data set
for j in 201029_exp1_  201029_exp2_ 201029_exp3_ 201029_exp4_ 201029_exp5_ 201029_exp6_ 201029_exp7_ 201029_exp8_ 201029_exp8_ 201116_exp4_ 201119_exp1_ 201119_exp2_ 201119_exp3_ 201119_exp4_ 201119_exp5_ 201119_exp6_ 201119_exp7_ 201119_exp8_ 201119_exp9_ 201119_exp10_
do
  python3 Main_remote.py -r "path_from_root" 'knit005_' "$j" "v1" --scalar --scalarstats --scalarevent
done
#
#### analyse scalar flu on mix data set
python3 Main_remote.py -r "path_from_root" 'knit005_' "mix_" "v1" --scalar --scalarstats --scalarevent

#### create ML data sets
python3 Main_remote_NN.py -r "path_from_root" 'knit005_' "mix_" "v1" --fluNN --flurscNN --scalarstats --scalarevent
