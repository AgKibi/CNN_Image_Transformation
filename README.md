# CNN_Image_Transformation
Image_Transformation_CycleGAN

Author: Agnieszka Kibitlewska

Project goal:
My goal in this project is to transform the photos downloaded from the Flickr website, taken by the user agnieszka186 from the Landscapes folder, into images stylized in the manner of Monet and Van Gogh paintings.

Work on the project is still ongoing

So far in the project:
- photos downloaded from Flickr user (agnieszka186) from Landscapes gallery (dataset: "photos"),
- downloaded photos were transferred to Google Drive in order to avoid loading photos to the working memory each time, and then each time downloading them directly from the drive and processing them,
- images of Monet and VanGogh have been prepared on the basis of which the downloaded "photos" will be stylized
- the "CycleGAN.py" script was prepared, which contains classes and functions for the generator, discriminator, cycleGAN model, model evaluation metrics.

Model cycleGAN has been builed based on Keras Code examples / Generative Deep Learning / CycleGAN (Author: A_K_Nain).

