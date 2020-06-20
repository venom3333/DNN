using Microsoft.ML;
using Microsoft.ML.Data;
using Microsoft.ML.Vision;

using System;
using System.Collections;
using System.Collections.Generic;
using System.IO;
using System.Linq;

using Tensorflow;
using Tensorflow.Eager;

using static Microsoft.ML.DataOperationsCatalog;
using static Microsoft.ML.Vision.ImageClassificationTrainer;

namespace DNN
{
    class Program
    {
        static void Main(string[] args)
        {
            string imagesRelativePath = @"../../../images";
            string testRelativePath = @"../../../test";
            var imagesFolder = FileHelper.GetAbsolutePath(imagesRelativePath);
            var testImagesFolder = FileHelper.GetAbsolutePath(testRelativePath);

            var mlContext = new MLContext();

            // Load all the original images info.
            IEnumerable<ImageData> images = FileHelper.LoadImagesFromDirectory(
                folder: imagesFolder, useFolderNameAsLabel: true);

            // Shuffle images.
            IDataView shuffledFullImagesDataset = mlContext.Data.ShuffleRows(
                mlContext.Data.LoadFromEnumerable(images));

            // Apply transforms to the input dataset:
            // MapValueToKey : map 'string' type labels to keys
            // LoadImages : load raw images to "Image" column
            shuffledFullImagesDataset = mlContext.Transforms.Conversion
                .MapValueToKey(
                    "Label",
                    keyOrdinality: Microsoft.ML.Transforms.ValueToKeyMappingEstimator.KeyOrdinality.ByValue
                )
                .Append(mlContext.Transforms.LoadRawImageBytes(
                    "Image",
                    imagesFolder, "ImagePath")
                )
                .Fit(shuffledFullImagesDataset)
                .Transform(shuffledFullImagesDataset);

            // Split the data 80:20 into train and test sets.
            TrainTestData trainTestData = mlContext.Data.TrainTestSplit(
                shuffledFullImagesDataset, testFraction: 0.2, seed: 1);

            IDataView trainDataset = trainTestData.TrainSet;
            IDataView testDataset = trainTestData.TestSet;

            // Set the options for ImageClassification.
            var options = new ImageClassificationTrainer.Options()
            {
                FeatureColumnName = "Image",
                LabelColumnName = "Label",
                // Just by changing/selecting InceptionV3/MobilenetV2/ResnetV250
                // here instead of ResnetV2101 you can try a different 
                // architecture/ pre-trained model. 
                Arch = ImageClassificationTrainer.Architecture.ResnetV2101,
                Epoch = 100,
                BatchSize = 10,
                LearningRate = 0.01f,
                MetricsCallback = (metrics) => Console.WriteLine(metrics),
                ValidationSet = testDataset,
                // Disable EarlyStopping to run to specified number of epochs.
                EarlyStoppingCriteria = null
            };

            // Create the ImageClassification pipeline.
            var pipeline = mlContext.MulticlassClassification.Trainers.
                ImageClassification(options)
                .Append(mlContext.Transforms.Conversion.MapKeyToValue(
                    outputColumnName: "PredictedLabel",
                    inputColumnName: "PredictedLabel"));

            Console.WriteLine("*** Training the image classification model " +
                "with DNN Transfer Learning on top of the selected " +
                "pre-trained model/architecture ***");


            // Train the model
            // This involves calculating the bottleneck values, and then
            // training the final layer. Sample output is: 
            // Phase: Bottleneck Computation, Dataset used: Train, Image Index:   1
            // Phase: Bottleneck Computation, Dataset used: Train, Image Index:   2
            // ...
            // Phase: Training, Dataset used:      Train, Batch Processed Count:  18, Learning Rate:       0.01 Epoch:   0, Accuracy:  0.9166667, Cross-Entropy:  0.4787029
            // ...
            // Phase: Training, Dataset used:      Train, Batch Processed Count:  18, Learning Rate: 0.002265001 Epoch:  49, Accuracy:          1, Cross-Entropy: 0.03529362
            // Phase: Training, Dataset used: Validation, Batch Processed Count:   3, Epoch:  49, Accuracy:  0.8238096
             var trainedModel = pipeline.Fit(trainDataset);

            Console.WriteLine("Training with transfer learning finished.");

            // Save the trained model.
            mlContext.Model.Save(trainedModel, shuffledFullImagesDataset.Schema,
                "model.zip");

            // Load the trained and saved model for prediction.
            ITransformer loadedModel;
            DataViewSchema schema;
            using (var file = File.OpenRead("model.zip"))
            {
                loadedModel = mlContext.Model.Load(file, out schema);
            }

            // Evaluate the model on the test dataset.
            // Sample output:
            // Making bulk predictions and evaluating model's quality...
            // Micro - accuracy: 0.851851851851852,macro - accuracy = 0.85
            EvaluateModel(mlContext, testDataset, loadedModel);

            // Predict on a single image class using an in-memory image.
            // Sample output:
            // Scores : [0.9305387,0.005769793,0.0001719091,0.06005093,0.003468738], Predicted Label : daisy
            TryPrediction(testImagesFolder, mlContext, loadedModel);

            Console.WriteLine("Prediction on a single image finished.");

            Console.WriteLine("Press any key to finish");
            Console.ReadKey();
        }

        // Evaluate the trained model on the passed test dataset.
        private static void EvaluateModel(MLContext mlContext,
            IDataView testDataset, ITransformer trainedModel)
        {
            Console.WriteLine("Making bulk predictions and evaluating model's " +
                "quality...");

            // Evaluate the model on the test data and get the evaluation metrics.
            IDataView predictions = trainedModel.Transform(testDataset);
            var metrics = mlContext.MulticlassClassification.Evaluate(predictions);

            Console.WriteLine($"Micro-accuracy: {metrics.MicroAccuracy}," +
                              $"macro-accuracy = {metrics.MacroAccuracy}");

            Console.WriteLine("Predicting and Evaluation complete.");
        }

        // Predict on a single image.
        private static void TryPrediction(string imagesForPredictions,
            MLContext mlContext, ITransformer trainedModel, int count = 10)
        {
            // Create prediction function to try one prediction.
            var predictionEngine = mlContext.Model
                .CreatePredictionEngine<InMemoryImageData,
                ImagePrediction>(trainedModel);

            // Load test images.
            IEnumerable<InMemoryImageData> testImages =
                FileHelper.LoadInMemoryImagesFromDirectory(imagesForPredictions, false);

            // Create an in-memory image objects from the first image in the test data.
            var imagesToPredict = testImages.Take(count).Select(image => new InMemoryImageData
            {
                Image = image.Image
            });

            foreach (var imageToPredict in imagesToPredict)
            {
                // Predict on the single image.
                var prediction = predictionEngine.Predict(imageToPredict);

                Console.WriteLine($"Scores : [{string.Join(",", prediction.Score)}], " +
                    $"Predicted Label : {prediction.PredictedLabel}");
            }

        }

    }
}
