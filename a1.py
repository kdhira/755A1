import numpy
import pandas
from sklearn import linear_model
from df_selector import *
from sklearn.model_selection import cross_val_score

# Load in file
input_file = 'data/occupancy_sensor/occupancy_sensor_data.csv'
df = pandas.read_csv(input_file, header=0)

# Remove irrelevant columns
df.drop(['date'], axis=1, inplace=True)



# Split into feature and target
df_features = df.iloc[:, numpy.arange(5)].copy()
df_target = df.iloc[:, 5].copy()

# # Pipeline on feature
# num_features = df_features[['Temperature', 'Humidity', 'Light', 'CO2', 'HumidityRatio']].copy()
# cat_features = df_features.drop(['Temperature', 'Humidity', 'Light', 'CO2', 'HumidityRatio'], axis=1, inplace=False)

# pipeline = create_pipeline(num_features, cat_features)
# df_features_processed = pandas.DataFrame(data=pipeline.fit_transform(df_features))
# df_processed = pandas.concat([df_features_processed, df_target], axis=1)
# print(df_processed)

clf = linear_model.Perceptron(alpha=0.1)
scores = cross_val_score(clf, df_features, df_target)

print(scores)

# # Split into training and test data (90:10)
# training_data = data[:int(sample_size * 0.9)]
# test_data = data[int(sample_size * 0.9):]

# # Define input and target feature
# input_features = [1,2,3,4,5]
# target_feature = 6

# perceptron = linear_model.Perceptron()

# print(training_data[:, input_features])
# print(training_data[:, target_feature])
# perceptron.fit(training_data[:, input_features], training_data[:, target_feature])

# test_data_actl = test_data[:, target_feature]
# test_data_pred = perceptron.predict(test_data[:, input_features])



# print(data)
# print(len(training_data) + len(test_data), sample_size)
