from fake_news import detect_fake_news
from predirandomforest import predict_fake_news
import pandas as pd

#Read test file
test_filepath = "./test_fake_news.csv"
test_data = pd.read_csv(test_filepath)

#Read training file
filepath1 = "./train_fake_news.csv"
training_set = pd.read_csv(filepath1)
training_set.columns = ['A', 'B', 'C']
del training_set['A']

#Test the detect_fake_news function
correct_detect_count = 0
for index, row in test_data.iterrows():
    title = row['text']
    label = row['label']
    result = detect_fake_news(title, training_set)
    
    if (result == "Fake News" and label == 1) or (result == "Real News" and label == 0):
        correct_detect_count += 1

detect_accuracy = correct_detect_count / len(test_data)

# Test the predict_fake_news function
correct_predict_count = 0
for index, row in test_data.iterrows():
    title = row['text']
    label = row['label']
    result = predict_fake_news(title)
    
    if (result == "Fake News" and label == 1) or (result == "Real News" and label == 0):
        correct_predict_count += 1

predict_accuracy = correct_predict_count / len(test_data)

# Print the precision of both functions
print("Accuracy of k-NN algorithm: ", detect_accuracy)
print("Accuracy of Random forest algorithm: ", predict_accuracy)
