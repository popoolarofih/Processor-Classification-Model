The system model is divided into four main phases: data collection, training, testing, and evaluation. We collected necessary data, preprocessed the data so that they would be usable with our chosen machine learning algorithms and subsequently trained and tested the model so that we could use the model for future classification of processors. 













Figure 1.0: Flow of operations in the created system
This methodology outlines the steps involved in building a machine learning model that can distinguish between AMD and Intel processors, using publicly available specification data. Here's a detailed breakdown of each step:
1.	Data Collection:
The first step involves gathering data. We need two separate datasets, one containing specifications for AMD processors and another for Intel processors, typically in CSV format. Once we have the data, we ensure consistency by standardizing the column names between the two datasets. Then, we select features most relevant to processor classification, such as product name, release date, core/thread count, clock speed, cache size, and thermal design power (TDP).
Next, we address missing data. In the Intel dataset, where data is assumed to be mostly complete, we might choose to simply remove rows with missing values. However, for the AMD dataset, we can impute missing values by replacing them with the average value for each numeric feature. Finally, we create a new variable to represent the processor brand. We assign a value of "0" to AMD processors and "1" to Intel processors.
Table 1.0: Sample of data collected
Model	launch Date	numCores	numThreads	graphicsModel	baseClock	TDP	Cache
AMD Ryzen™ 5 6600H	2022	6	12	AMD Radeon™ 660M	3.3	45	384
AMD Ryzen™ 5 6600HS
2022	6	12	AMD Radeon™ 660M	3.3	35	384
AMD Ryzen™ 5 6600U	2022	6	12	AMD Radeon™ 660M	2.9	15	384
AMD Ryzen™ 5 PRO 6650H	2022	6	12	AMD Radeon™ 660M	3.3	45	384

Atom x6200FE	2021	2		Intel UHD Graphics for 10th Generation Intel Core Processors	1		1.5
Atom x6211E	2021	2		Intel UHD Graphics for 10th Generation Intel Core Processors	1.3		1.5
Atom x6212RE	2021	2		Intel UHD Graphics for 11th Generation Intel Core Processors	1.2		1.5
Atom x6413E	2021	4		Intel UHD Graphics for 10th Generation Intel Core Processors	1.5		1.5
Atom x6414RE	2021	4		Intel UHD Graphics for 10th Generation Intel Core Processors	1.5		1.5
Atom x6425E	2021	4		Intel UHD Graphics	2		1.5

2.	Data Exploration:
After cleaning the data, we performed Exploratory Data Analysis (EDA) to understand the data distribution and relationships between features. This helps us gain insights into the data and identify any potential issues. Common EDA techniques include:
•	Visualizing the distribution of the target variable (processor brand) using count plots to see how many processors belong to each category (AMD or Intel).
•	Creating pair plots to visually explore the relationships between different features, like core count and clock speed.
•	Calculating correlation coefficients, such as Spearman's correlation, to quantify the linear relationships between features. This helps us understand how features might influence each other.
3.	 Data Integration and Feature Engineering:
Once we have a good understanding of the data, we combine the preprocessed AMD and Intel dataframes into a single, unified dataframe. This allows us to train a single model on all the data.
However, before training the model, we need to perform some feature engineering. Categorical data, like product names, can't be directly used by machine learning models. We addressed this by using a technique called Label Encoding. This process assigns a unique numerical label to each category (e.g., "A10-5700" gets label 1, "Core i7" gets label 2).
Additionally, for numerical features like core count and clock speed, we use a technique called Standardization. This ensures that all features have a similar scale (typically zero mean and unit standard deviation). This is important because some models can be sensitive to the scale of features, and standardization helps prevent features with larger scales from dominating the model during training.
4.	 Model Training and Evaluation:
Now that our data is prepared, we can move on to training the model. Here's how we went about it:
•	Train-Test Split: We split the merged dataframe into two sets - a training set and a testing set. Typically, a 70/30 split is used, where 70% of the data is used for training the model and the remaining 30% is used for testing its performance on unseen data.
•	Model Selection and Training: We choose a suitable machine learning model for binary classification, which means it can predict one of two classes (AMD or Intel in this case). In this system, we used a Support Vector Machine (SVM) model. The model is then trained on the training set, allowing it to learn the patterns and relationships within the data. 
•	Support Vector Machine (SVM) model
The SVM algorithm finds the optimal hyperplane that maximizes the margin between the two classes in the feature space. The margin is defined as the distance between the hyperplane and the closest data points from both classes. These closest data points are called support vectors.
The algorithm works by solving an optimization problem to find the hyperplane that achieves the maximum margin. Mathematically, this optimization problem can be represented as:
minimize:    (1/2) * ||w||^2
subject to:  y_i * (w^T * x_i + b) >= 1, for all i
Where:
- `w` is the normal vector to the hyperplane
- `b` is the bias term
- `x_i` are the training data points
- `y_i` are the corresponding class labels (-1 or 1)
The optimization problem tries to find the values of `w` and `b` that satisfy the constraints (the data points of each class are on the correct side of the hyperplane) while minimizing the norm of `w` (which maximizes the margin).
In the case of linearly separable data, the SVM algorithm finds the optimal hyperplane that separates the two classes with the maximum margin. However, in cases where the data is not linearly separable, the algorithm introduces slack variables to allow for some misclassifications while still trying to maximize the margin.
Additionally, the SVM algorithm can be extended to handle non-linear decision boundaries by using kernel functions. These functions map the original data into a higher-dimensional feature space, where the data might become linearly separable.
Some common kernel functions used in SVMs include:
- Linear kernel: `K(x, y) = x^T * y`
- Polynomial kernel: `K(x, y) = (x^T * y + c)^d`
- Radial Basis Function (RBF) kernel: `K(x, y) = exp(-gamma * ||x - y||^2)`
The choice of kernel function depends on the characteristics of the data and the complexity of the decision boundary.
During the training phase, the SVM algorithm finds the support vectors and constructs the decision boundary based on these vectors. During prediction, new data points are classified based on their position relative to the decision boundary. Overall, SVMs are powerful and versatile algorithms for binary classification problems, particularly when the classes are linearly separable or when the data can be transformed into a higher-dimensional space using kernel functions.
Algorithm for SVM used in the system:
Step 1:  Load the important libraries
Step 2: Import dataset and extract the X variables and Y separately.
Step 3: Divide the dataset into train and test
Step 4: Initializing the SVM classifier model
Step 5: Fitting the SVM classifier model
svm_clf.fit(X_train, y_train)
Step 6: Coming up with predictions y_pred_test = svm_clf.predict(X_test)
Step 7: Evaluating model’s performance metrics.accuracy(y_test, y_pred_test)
