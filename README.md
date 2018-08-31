# SOFTENG 755 A1 - Kevin Hira, khir664, 6384869

## Running the application
Make sure you have numpy, pandas, and scikit_learn installed (pip3 install pandas numpy sklearn)

Then, to run, execute the a1.py script with python3:
`python3 a1.py data_set_name [test_data_location]`

where `data_set_name` is one of: `world_cup_2018_clf`, `world_cup_2018_reg`, `traffic_flow`, `occupancy_sensor`, or `landsat`

providing a value for test_data_location (location on disk relative to current directory) will use an external test csv source for testing the models. This csv must be in the same format as the training data (see the data/ directory for the training data csvs)
