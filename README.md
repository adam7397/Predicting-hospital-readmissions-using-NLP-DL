# Predicting-hospital-readmissions-using-NLP-DL

Going to the hospital when you have medical issues is often not an isolated issue. There is typically a good chance that you will end up going back due to further complications or other issues. For this analysis I took a look at a dataset of diabetes patients from 130 US hospitals from 1999-2008. These patients were admitted for diabetic related reasons and the set has information about the patient, how long they were admitted, which tests were performed, and which medicine was prescribed. Pandas proved useful for finding the relevant data and pre-processing it. I then used numpy and scikit-learn to put together the necessary arrays for the model. With keras I was easily able to add layers and play with optimizers and loss functions to try and get a useful result.
To start, I took a look at the data to start and provided visualizations of the subjects information. I then focused on cleaning the data, this was done to remove information which was not as useful or had missing values, which could lead to a poor result. When training I used a 67/33 split, and trained over 10 epochs with a batch size of 256. I tried many different epoch and batch sizes, but the model did not benefit from long trainings. The columns were created using pandas’ get_dummies function to provide numerical values to the strings provided. This was easier to work with and process, as all the possible features are represented as columns. I was able to complete the training on cpu due to the dataset only being 69973 x 2291, each epoch took about 1s to complete.  The model was able to achieve around 73% accuracy at predicting readmission status, and used the adam optimizer with the binary cross-entropy loss function.
Installation & Configuration
This project was completed using python and popular libraries which can be found online. Juypter notebook was also used to write the code and details about the project. It is not necessary to use it to recreate the results, but is required to view the project notebook.
Python 3.6.8: https://www.python.org/downloads/release/python-368/
Python has a package manager called ‘pip’ which was used to install all of the dependent packages and libraries.
To install a package through pip, please format a request like this:
‘pip install jupyter’
The dependent packages are as follows:
Juypter notebook
Numpy
Pandas
Tensorflow (gpu version not required)
Keras
Matplotlib
Scikit-learn

Dataset
The dataset is publicly available from the University of California Irvine Machine Learning repository. This repository contains many free, public data sets from which models can be trained and developed. The set which I used for this project is titled ‘Diabetes 130-US hospitals for years 1999-2008 Data Set’ It contains roughly 100,000 encounters with diabetes patients over a 9 year period. There is no identifiable information from this dataset, and it comes from 130 hospitals around the US. The main points from the repository describe the data in the set as:
It is an inpatient encounter (a hospital admission)
It is a diabetic encounter, that is, one during which any kind of diabetes was entered to the system as a diagnosis.
The length of stay was at least 1 day and at most 14 days.
Laboratory tests were performed during the encounter.
Medications were administered during the encounter.
 It contains 55 columns to describe each encounter. To describe each patient we have the following information:
Patient number
Race
Gender
Age (in a 10 year bucket)
Weight
The rest of the attributes (besides the readmitted status) describe what type of how they were admitted, how long they stayed, treatment they received, and how they were discharged.

Some of the data was removed from the original set in an attempt to get a better result. First, the weight column was removed as nearly all of the rows were missing it. Medical specialty and payer code also had about half of the rows missing, so these were dropped as well. In an effort to not skew the results by having multiple encounters with the same patient, only a single encounter was used for each patient. This was simple to remove, as pandas has a built in method to remove rows with a duplicate in a specified column. There is also a column for how the patient was discharged. Since we are looking to predict that a patient would need to be readmitted to the hospital, keeping patients in the dataset who were discharged to some sort of hospice care or who passed away did not make sense. The dataset contains a file to describe some of the identification numbers found in the dataset, here they were used to interpret the ‘discharge_disposition_id’ column.
	Once the columns were removed and rows cleaned, it was time to prepare the set for the model. The model prefers to work with integer values, and instead of vectorizing them I opted to create dummy columns to represent the values. For example, for a column like ‘readmitted’, there were 3 possible values: ‘NO’, ‘<30’, and ‘>30’. To represent them using dummy columns, the column is replaced with 3 columns, one for each of the 3 possible values. A patient who was not readmitted, would have a 1 in the readmitted_NO column, and 0s in the other two columns.  This made the dataset have more columns, due to some columns containing many values. The final shape of the dataset ended up being 69,973 rows with 2,291 columns.



The Model
To put pre-processed data into the model, it needed to be separated. It was first converted from a pandas Dataframe to a numpy array. Then all the rows besides the readmitted columns made up the ‘x’ set, and the readmitted status is the ‘y’. Once that was done, scikit was used to split the data, with 33% kept for testing. The model was built using keras and starts with creating a sequential model. Next an input layer is needed, I used a dense layer with a size of 16, this value can be adjusted based on outcome, with a relu activation and input shape of ‘2291,’ to represent the columns in the data set.
The Hidden layers are a mix of dropout and dense layers. I added the dropout layers as a way to try and combat overfitting in the model, and it seems to have helped a little. For the first dropout layer I drop .3 with a seed of 42. I then have a dense layer of size 32 with relu activation. There is then another dropout and dense layer, but this time the dropout is reduced to .2. 
The output layer is a dense layer with size 3 to represent the 3 columns which make up the readmitted status. I used a sigmoid activation function here instead of the relu due to it being more of curve prediction. It allowed for a bit more variation in the answer than relu. It was then compiled using the adam optimizer and binary cross entropy loss functions. I played around with many different loss and optimizer functions but these two provided the best results.

Results
The model proved to not benefit from many long epochs, as each one only took around 1 second and it would start to flatten out around 5 epochs in. I am not sure if this was just the nature of the data / problem, or just the way the model was built. It was able to achieve around 73% accuracy when it came to predicating readmittance. I tried to play with different layers, activation functions, sizes, optimizers, and loss functions but this was about the best I could do. It may have inherently been the way the data was presented, or the nature of the problem with this size of dataset. 
Fig 5:Model  Accuracy
	
Conclusion
	Many things did not work well for this project, I had to make many adjustments to get it passed 50% accuracy and there were some points where the model was not progressing at all. Using articles and lecture assignments I was able to find ways to make progress. This assignment did prove to be a fun way to actually build something on my own. I enjoyed figuring out ways to construct the data set and tweaking the model myself. It helped to have a model which was easy and fast to train as I was able to make adjustments and quickly see the outcome. I am not sure if deep learning was entirely necessary for a project of this scope, and other statistical models may have been able to produce similar or better results. The real world application of this project was my favorite part and I hope to continue to build models like this in the future!
