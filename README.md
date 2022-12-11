# Linear_Vector_Quantization1
LVQ implimentation on 9x9 and 10x10 2-D image with 4 cluster classes.
> Inputs to be provided:

	> Learning Rate(best results are obtained when given between 0.1 to 0.2)
	> Iterations
	> Name of the Training Pattern (trainingPattern_i.txt)(i = 1/2/3)


> Each trainingPatter_i.txt file contains info about size of the pattern and pattern itself.

> trainingPattern_1/2 have Size = 09x09, while traininPattern_3 have training pattern of Size = 10x10.

> TrainingPattern_1.txt contains the training data given in the assignment.

> **The size of trainingPatterns and Weight Matrix are not normalized.**

> There are 4 classes as: [+,*,^,o]

> Learning rate will be decreased by 5% after every epoch.

> Stopping conditions are: 

	>end of epochs or 
	>learning rate reaching 5% of the initial value provided.

> If epochs are small(less than 10), plots will be displayed for subsequent epochs.

> Output.txt file will be generated with all the required information for each epoch.
