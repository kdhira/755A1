import numpy
import pandas
from sklearn import linear_model
from sklearn.model_selection import cross_val_score, train_test_split
import train_and_test

import world_cup, traffic_flow, occupancy_sensor, landsat

import sys

if len(sys.argv) < 2:
    print("Error. Usage: python3 a1.py data_set_name [external_test]")
    exit(1)

# Helper
data_set_map = {
    'world_cup_2018_clf' : {
        'training_csv' : 'data/world_cup_2018/world_cup_2018_data.csv',
        'preprocess_function' : world_cup.preprocess_clf,
        'ml_type': 'clf'
    },
    'world_cup_2018_reg' : {
        'training_csv' : 'data/world_cup_2018/world_cup_2018_data.csv',
        'preprocess_function' : world_cup.preprocess_reg,
        'ml_type': 'reg'
    },
    'traffic_flow' : {
        'training_csv' : 'data/traffic_flow/traffic_flow_data.csv',
        'preprocess_function' : traffic_flow.preprocess_reg,
        'ml_type': 'reg'
    },
    'occupancy_sensor' : {
        'training_csv' : 'data/occupancy_sensor/occupancy_sensor_data.csv',
        'preprocess_function' : occupancy_sensor.preprocess_clf,
        'ml_type': 'clf'
    },
    'landsat' : {
        'training_csv' : 'data/landsat/landsat_data.csv',
        'preprocess_function' : landsat.preprocess_clf,
        'ml_type': 'clf'
    },
}


# Env vars
data_set_name = sys.argv[1]
test_data_input_file = ''
if len(sys.argv) >= 3:
    test_data_input_file = sys.argv[2]

if data_set_name in data_set_map:
    df_train = pandas.read_csv(data_set_map[data_set_name]['training_csv'])
    train_feature, train_target = data_set_map[data_set_name]['preprocess_function'](df_train)
    df_test = None
    if test_data_input_file != None and test_data_input_file != '':
        df_test = pandas.read_csv(test_data_input_file)
        test_feature, test_target = data_set_map[data_set_name]['preprocess_function'](df_test)

        x_train, y_train, x_test, y_test = train_feature, train_target, test_feature, test_target
    else:
        x_train, x_test, y_train, y_test = train_test_split(train_feature, train_target, test_size=0.1, random_state=0)

    train_and_test.test_all(data_set_map[data_set_name]['ml_type'], x_train, y_train, x_test, y_test)


else:
    print("Error")
    exit(2)



# # Remove irrelevant columns
# df.drop(['date'], axis=1, inplace=True)



# # Split into feature and target
# df_features = df.iloc[:, numpy.arange(5)].copy()
# df_target = df.iloc[:, 5].copy()

# # # Pipeline on feature
# # num_features = df_features[['Temperature', 'Humidity', 'Light', 'CO2', 'HumidityRatio']].copy()
# # cat_features = df_features.drop(['Temperature', 'Humidity', 'Light', 'CO2', 'HumidityRatio'], axis=1, inplace=False)

# # pipeline = create_pipeline(num_features, cat_features)
# # df_features_processed = pandas.DataFrame(data=pipeline.fit_transform(df_features))
# # df_processed = pandas.concat([df_features_processed, df_target], axis=1)
# # print(df_processed)

# clf = linear_model.Perceptron(alpha=0.1)
# scores = cross_val_score(clf, df_features, df_target)

# print(scores)

# # # Split into training and test data (90:10)
# # training_data = data[:int(sample_size * 0.9)]
# # test_data = data[int(sample_size * 0.9):]

# # # Define input and target feature
# # input_features = [1,2,3,4,5]
# # target_feature = 6

# # perceptron = linear_model.Perceptron()

# # print(training_data[:, input_features])
# # print(training_data[:, target_feature])
# # perceptron.fit(training_data[:, input_features], training_data[:, target_feature])

# # test_data_actl = test_data[:, target_feature]
# # test_data_pred = perceptron.predict(test_data[:, input_features])



# # print(data)
# # print(len(training_data) + len(test_data), sample_size)
