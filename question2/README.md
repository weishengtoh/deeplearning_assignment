# Pokemon Image Classification using Custom Tensorflow Components and TensorBoard

The dataset used for this project consists of 250 pokemon images, equally divided amoung 5 pokemon classes (50 images each). 

The dataset has been predefined for the assignment, and the 5 pokemon to be identified are:
- Pikachu
- Bulbasaur
- Squirtle
- Mew Two
- Charmander

![dataset](unrelated_imgs/dataset.png)

The objective of the project is to perform a multiclass classification with the use of transfer learning, to identify which pokemon a particular image belongs to.  


## Usage

The project is meant to be executed entirely on a notebook, as it is meant only for experimentation and not production ready.  

The project make use of TensorBoard extensively during the training process, to visualise the metrics and progress while the model is being trained.  
Custom tensorflow components has also been defined to generate visualisations 

To speed up the data pipeline, the data has been converted into `TFRecords`, and prefetching has been included to speed up the entire training process while data augmentation is being performed.   

All instructions required to run the project has been included within the notebook, and it is preferable to execute the project in `Visual Studio Code`, where the use of a `Tensorboard` extension is made possible.  


## References

**Tensorflow**
- [TensorFlow docs](https://www.tensorflow.org/overview)
- [TensorFlow: Data and Deployment Specialization](https://www.coursera.org/specializations/tensorflow-data-and-deployment?utm_source=gg&utm_medium=sem&utm_content=01-CatalogDSA-ML2-US&campaignid=12490862811&adgroupid=119269357576&device=c&keyword=&matchtype=&network=g&devicemodel=&adpostion=&creativeid=503940597764&hide_mobile_promo&gclid=Cj0KCQiA64GRBhCZARIsAHOLriI0wS3o5M0fDTtRHlksNo1K9lv4f_R8fibbK5EqYcF6yuN3PUDUfjcaAoTXEALw_wcB)
