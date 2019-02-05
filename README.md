# Predicting Image Memorability with MemNet in Keras on PowerAI

This code pattern will enable you to build an application that predicts how "unique" or "memorable" images are. You'll do this through the Keras deep learning library, using the MemNet architecture. The dataset this neural network will be trained on is called "LaMem" (Large-scale Image Memorability), by MIT. In order to process the 45,000 training images and 10,000 testing images (227x227 RGB) efficiently, we'll be training the neural network on a PowerAI machine on NIMBIX, enabling us to benefit from NVLink (direct CPU-GPU memory interconnect) without needing any extra code.
Once the model has been trained on PowerAI, we'll convert it to a CoreML model and expose it via a web application written in Swift, running on a Kitura server on macOS.

When the reader has completed this pattern, they'll understand how to:

* Train a Keras model on PowerAI.
* Use a custom loss function with a Keras model.
* Convert tf.keras models that deal with images to CoreML models.
* Use the Apple Vision framework with a CoreML model in Swift to get `VNCoreMLFeatureValueObservation`s.
* Host a Web Server with Kitura
* Expose a Mustache HTTP template through Kitura

## Flow

TODO: add flow diagram

1. A Keras model is trained with the LaMem dataset.
1. The Keras model is converted to a CoreML model.
1. The user uploads their image to the kitura web app.
1. The Kitura web app uses the CoreML model for predictions.
1. The user recieves the neural network's prediction.

## Included Components

* [IBM Power Systems](https://www.ibm.com/it-infrastructure/power): A server built with open technologies and designed for mission-critical applications.
* [IBM PowerAI](https://www.ibm.com/us-en/marketplace/deep-learning-platform): A software platform that makes deep learning, machine learning, and AI more accessible and better performing.
* [Kitura](https://www.kitura.io): Kitura is a free and open-source web framework written in Swift, developed by IBM and licensed under Apache 2.0. It’s an HTTP server and web framework for writing Swift server applications.

## Featured Technologies

* [Artificial Intelligence](https://medium.com/ibm-data-science-experience): Artificial intelligence can be applied to disparate solution spaces to deliver disruptive technologies.
* [Swift on the Server](https://developer.ibm.com/swift/): Build powerful, fast and secure server side Swift apps for the Cloud.

# Prerequisites

* If you don't already have a PowerAI server, you can acquire one from [Nimbix](https://www.nimbix.net/ibm) or from the [PowerAI offering](https://cloud.ibm.com/catalog/services/powerai) on IBM Cloud.
* macOS 10.13 (High Sierra) or later

# Steps

1. [Clone the repo](#1-clone-the-repo)
1. [Download the LaMem data](#2-download-and-extract-the-lamem-data)
1. [Train the Keras model](#3-train-the-keras-model)
1. [Convert the Keras model to a CoreML model](#4-convert-the-keras-model-to-a-coreml-model)
1. [Run the Kitura web app](#5-run-the-kitura-web-app)

### 1. Clone the repo

Clone the `powerai-image-memorability` repo onto both your PowerAI server and local macOS machine. In a terminal, run:

```
git clone https://www.github.com/IBM/powerai-image-memorability
```

### 2. Download and extract the LaMem data

To download the LaMem dataset, head over to the `powerai_serverside` directory, and run the following command:

```
wget http://memorability.csail.mit.edu/lamem.tar.gz
```

Once the dataset is done downloading, run the following command to extract that data:

```
tar -xvf lamem.tar.gz
```

### 3. Train the Keras model

To train the Keras model, run the following command inside of the `powerai_serverside` directory:

```
python train.py
```

Once Python script is done running, you'll see a `memnet_model.h5` model in the `powerai_serverside` directory. Copy that over to the `webapp` directory on the macOS machine that you'd like to run the frontend on.

### 4. Convert the Keras model to a CoreML model

Inside of the `webapp` directory on your macOS machine, run the following Python script to convert your Keras model to a CoreML model:

```
python convert_model.py memnet_model.h5
```

This may take a few minutes, but when you're done, you should see a `lamem.mlmodel` file in the `webapp` directory.

### 5. Run the Kitura web app

Then, you're ready to roll! Run the following command to build & run your application:

```
swift build && swift run
```

Now, you can head over to `localhost:3333` in your favourite web browser, upload an image, and calculate its memorability.

TODO: add screenshot
