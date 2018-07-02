from data import sc, get_features

try:
    from pyspark.mllib.regression import LabeledPoint
    from pyspark.mllib.classification import NaiveBayes, NaiveBayesModel

    print("Successfully imported Spark Modules train")

except ImportError as e:
    print("Can not import Spark Modules", e)

model_folder = "models/naive_base_model.conf"

train, test = get_features()

model = NaiveBayes.train(train)

try:
    print 'saving the classifier....'
    model.save(sc, "models/naive_base_model.conf")
except Exception as e:
    print 'Problem while saving the classifier!'

try:
    model = NaiveBayesModel.load(sc, model_folder)
    # Compare predicted labels to actual labels
    prediction_and_labels = test.map(
        lambda point: (model.predict(point.features), point.label))

    # Filter to only correct predictions
    correct = prediction_and_labels.filter(
        lambda (predicted, actual): predicted == actual)

    # Calculate and print accuracy rate
    accuracy = correct.count() / float(test.count())

    print ("Classifier correctly predicted category " + str(accuracy * 100) + " percent of the time")


except Exception as e:
    print e

