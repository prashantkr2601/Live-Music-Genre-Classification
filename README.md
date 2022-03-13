## <h2 align="center"> <hr/>Live Music Genre Classification<hr/> </h2>

<h2 align= "center">Summary</h2>

> Capsule Networks for Live Music Genre Recognition is a project aimed at creating a neural network recognizing music genre and providing a user-friendly visualization for the network's current belief of the genre of a song.
> The model has since been significantly improved and rewritten to TensorFlow.js, so that it doesn't require a backend - the network can be run inside of the user's browser.

> It's easiest to run using Docker:

```shell
docker build -t genre-recognition . && docker run -d -p 8080:80 genre-recognition
```

> The demo will be accessible at Link: https://audio-genres.netlify.app/

> By default, it will use a model pretrained by us, achieving 85% accuracy on the GTZAN dataset. You can also provide your own model, as long as it matches the input and output architecture of the provided model. If you wish to train a model by yourself, download the [GTZAN dataset](http://opihi.cs.uvic.ca/sound/genres.tar.gz) (or provide analogous) to the data/ directory, extract it, run `create_data_pickle.py` to preprocess the data and then run `train_model.py` to train the model. Afterwards you should run `model_to_tfjs.py` to convert the model to TensorFlow.js so it can be served.

```shell
cd data
wget http://opihi.cs.uvic.ca/sound/genres.tar.gz
tar zxvf genres.tar.gz
cd ..
pip install -r requirements.txt
python create_data_pickle.py
python train_model.py
python model_to_tfjs.py
```

> You can "visualize" the filters learned by the convolutional layers using `extract_filters.py`. This script for every neuron extracts and concatenates several chunks resulting in its maximum activation from the tracks of the dataset. By default, it will put the visualizations in the filters/ directory. It requires the GTZAN dataset and its pickled version in the data/ directory. Run the commands above to obtain them. You can control the number of extracted chunks using the --count0 argument. Extracting higher numbers of chunks will be slower.

> <p align="justify">Music classifications are downright marks made by people to decide the style of music. In light of the human recognition , allocating a kind to a music piece is anything but an inconsequential assignment. Disregarding that, music kind is likely the most evident descriptor which comes to mind, and it is presumably the most generally used to sort out and oversee enormous computerized music databases. The writing gives us that in the most recent decade a few specialists have given a lot of endeavors towards programmed music sort characterization. In this paper we attempt to introduce a similar examination of 3 fundamental sound order models viz. KNN, CNN and the Hinton's most recent Capsule Networks. We attempt to evaluate the presentation of the new container classifier on sound information and attempt to improve its precision over the exactness of 83% accomplished till date on the GTZAN dataset. By utilizing the Hinton's most recent dynamic routing by agreement we increment loads of the neurons demonstrating great reaction to a specific type to much further expand the limit of that highlight. We likewise adjust the direction parameters of the MFCCs with the goal that we can utilize that data to anticipate genre which was recently lost in the event of utilizing CNNs. </p>

<h2 align= "center">Introduction</h2>
 
> <p align="justify">The genre of sound records is a difficult errand in the field of music information retrieval (MIR). Sound is dissected dependent on their advanced marks for certain variables, including beat, acoustics, vitality, move capacity, and so on to characterize what sort of music it is. Genre fills in as a significant order trait in the zones of sound sign pressure, music suggestion frameworks and music association. Diverse AI strategies have end up being very effective in extricating patterns and examples from the enormous arrangement of information. In this task, we will order music based on 10 classification genre- blues, classical, rock, jazz, reggae, metal, country, pop, disco and hip-hop. In this paper, we ordered sound sort utilizing capsule networks prepared by as of late proposed dynamic routing by-agreement instrument. We will attempt to propose an engineering for Hinton capsule networks fit for sound characterization assignments and study the effect of different parameters on arrangement precision.</p>

<h2 align= "center">Problem Statement </h2>

> <p align="justify"> Every day we listen to different types of music. By types we mean different genres of music. There is a need for efficient and fast genre prediction for accurate audio classification. Genre classification also finds its application in the area of audio compression. Characterization of kind can be important to clarify some real fascinating issues, for example, making melody references, finding related tunes, discovering social orders who will like that particular tune. The motivation behind our examination is to discover how the advanced Hinton’s capsule networks perform in genre prediction. We also shall try to present a comparative analysis between capsule networks and its older counterparts like k-nearest neighbor (k-NN) and CNN (Convolutional Neural Networks) and Linear Discriminant Analysis(LDA). </p>

> <p align="justify">The significance of the audio genre classification is to efficiently classify audio genre of the problem categorizing genre, etc. With the assistance of a programmed, content-based music classification framework, we will have the option to allot appropriate names to melody files, and in this way deal with the developing tune database helpfully. Also, the areas of applications of capsule networks have not yet been fully explored. By applying CapsNet to our problem of audio classification we could accurately assess the abilities of the capsule networks and analyze its importance. Efficient and Fast Genre classification is the sole objective of the project. </p>

> <p align="justify"> A thorough study into existing tools ad technologies need in the field of Object Detection was done through exploring various projects mentioned in the research papers that we studied. Also many open source communities like Stack Overflow and Github were of help. We understood that lyrics have little importance in describing the genre of a particular music and are considered as noisy data. Further study highlighted only 2 main algorithms for audio genre classification viz CNN and KNN algorithms. One of the most prominent method we adopted was to test and select approach. Of all the customized algorithms available online we locally tested the one which deemed worthy to providing the results with our dataset. We tested our algorithm on the famous GTZAN Dataset containing labelled data for 10 most popular music genres worldwide.</p>

<h2 align= "center">Solution Approach</h2>
 
> <p align="justify"> The capsule networks have already provided datasets like traffic sign classification MNIST dataset etc. The best of the results achieved by the neural networks till date in the problem of audio signal classification is by the CNN. However Capsule Networks were introduced by Hinton because of the disadvantages of the Max pooling layer of the convolutional net which results in data loss. Since capsule networks have already provided better results in past for above mentioned problems than CNN, we expect capsule networks to perform better in this area too. Hence we opted for this project to gauge the performance of capsule networks and deepen our understanding of CapsNet.</p>

<img src="./Image/Flow Chart.png" alt="Result" width="100%" height="300px"/>

> _<h3 align= "center"> Steps for Processing Audio Signals</h3>_

> - Considering the need for an efficient music classifier in today’s world, we designed an application that acts as a genre classifier identifies the genre of the music we listen to and gives recommendations according to the music we love to listen to.
> - First of all, the user will upload a music which shall belong to a particular genre through any device
> - We use python to process the song and generate a spectrogram and find features relating to the music.
> - The features are mainly spectral in nature viz MFCCs Spectral Roll off, Spectral Flux and Spectral Centroid.
> - The genre of the music is predicted using the model that we had trained.
> - Further using the data of predicted genre, musical recommendations are given to the user to provide him with a better user experience.

> The Dataset consisted of .au files. Python has difficulty in handling .au files so first these files were converted into a more workable format .wav. It is a part of data preprocessing
> The input signal is then further preprocessed using preprocessing methods such as lyrics - filtering (removal of lyrics using python librosa). Feature Extraction is then used to find different audio classification features such as Zero Crossing Rate, MFCC, Spectral Flux etc. The extracted features are then fed into the machine learning model which is intricately designed to solve the problem of audio genre classification. The predicted genre is then compared against the labelled data to hence find the accuracy of our model.

<img src="./Image/Flow Chart2.png" alt="Result" width="100%" height="600px"/>

> _<h3 align= "center"> Steps for Processing Audio Signals</h3>_

<img src="./Image/User Diagram.png" alt="Result" width="100%" height="600px"/>

> _<h3 align= "center"> User Diagram </h3>_

<img src="./Image/Activity Diagram.png" alt="Result" width="100%" height="600px"/>

> _<h3 align= "center"> Activity Diagram </h3>_

> Implementation Details:
>
> > - The application first takes the input audio signal.
>
> > - The input audio signal is then fed into our pre-trained model.
>
> > - After the model predicts the output, the user knows the genre of the song he/she listens to
>
> > - We post the genre of our music in Google search to get data for recommendations.
>
> > - We then display the recommendations to the user We first converted our .au files to .wav as python cannot work with .au files well but can
> >   process .wav files well.

> <p align="justify">We first sampled our audio signal to standard sampling rate and then created ceps using the python_speech_features library:- Mel Frequency Capstral coefficients for efficient sampling of our audio signal During ceps transformation some of the bad indeces were encountered. We then fileted out those bad indeces by setting up audio informaton at that point to zero. We created .npy (numpy) files each storing the mfcc array. Similarly we created an array of npy files each containing fft(Fast Fourieer Transform Features) features . We found out that the mfcc features were better when it came to solving the problem of audio classification.</p>

<h2 align= "center">Result</h2>

> <p align="justify"> We thoroughly investigated the efficiency of automated and they did not disappoint us. improvement in accuracy of 3% when classifying on images of audio spectrograms. We tried several methods of classification (viz KNN, CNN, Multivariate Logistic Regression Classifier and CapsNet) each by including distinct features in our feature set and then comparing their performance. CapsNet simply outperformed other existing state of the art architectures and yielded better results.</p>

<h2 align= "center">Conclusion</h2>

> <p align="justify">Capsule Networks are a better classifier for images than the traditional CNN which overcomes the downsides of the CNN by retaining spatial information in audio spectral images. </p>

<h2 align= "center">Future Work</h2>

> <p align="justify"> We will try to employ capsule networks on different problem sets in areas of image classification in order to map its performance in different circumstances. </p>

---

 <h2 align="center"> ** Thank YOU ** </h2> 
 
 ***
