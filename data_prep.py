from Scripts.prepare_data import (process_traj, make_catalogue_csv_files, make_train_specs, make_augment_data,
                calc_traj_channels, sep_train_and_test)
from Scripts.read_config import config 
import os
process_traj(config)
make_catalogue_csv_files()
make_train_specs()
make_augment_data()
calc_traj_channels()
sep_train_and_test()