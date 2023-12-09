The project focuses on learning several NLP techniques, for example, Tokenizing, Stemming, and Embedding to classify toxic online comments on social media. LSTM was developed to classify the comments, and 2 other algorithms (Decision Tree, SGDRegressor) were also compared with LSTM to showcase the success rate of LSTM in NLP applications.

### Dataset
The dataset from problem is collected from Kaggle, named Jigsaw Unintended Bias in Toxicity Classification. 
![image](https://github.com/nabilah-afrin/Deep-learning-projects/assets/118561482/18ae8068-7c7d-4979-bd7e-01dc044ab5d5)
![image](https://github.com/nabilah-afrin/Deep-learning-projects/assets/118561482/f6f576ce-7a87-441f-9424-965c31731651)




### Result

![image](https://github.com/nabilah-afrin/Deep-learning-projects/assets/118561482/4ba204e6-af94-45a7-a36b-3e48760f33dd)

The performance of the models for toxic comment classification are:
* SGDRegressor:
* * Hyperparameters Tuned Values: learning_rate(alpha): 1e-05 and penalty: l2
* *	Train MSE Loss: 0.02281
* *	CV MSE Loss: 0.02326
*	Decision tree
* *	Hyperparameters Tuned Values: max_depth: 7 and min_samples_leaf: 100
* *	Train MSE Loss: 0.0310
* *	CV MSE Loss: 0.03128
*	LSTM
* *	Train MSE Loss: 0.0157
* *	CV MSE Loss: 0.0162
