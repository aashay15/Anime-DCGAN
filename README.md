# Anime-DCGAN
##### This is a simple implementation of DCGAN (keras) to generate anime character's faces.

This project is one of the projects from the Book -> "Generative Adversarial Networks Projects by Ahirwar & Kailash"
This readme file will help you replicate my steps and help you in succesfully training a DCGAN to generate animefaces.
Before following all the steps download all the required dependancies.
```
pip install -r requirements.txt
```

### Dataset 
To download the dataset we will use a web scraping tool gallery-dl, simply open up the terminal and type the following command.
NOTE : Run this command in the directory where you want images to load it will automatically create a folder named gallery-dl and will save
       the data. 
```
pip install gallery-dl 
gallery-dl "https://danbooru.donmai.us/posts?tags=face”
```
I downloaded around 600 images (which is less for generating highly accurate & clear images)

The dataset includes many unwanted parts other than faces, which can hinder perfomance of the network and is not suitable for this 
specific task so we will [anime-face-detector](https://github.com/qhgz2013/anime-face-detector.git) to detect animefaces and 
crop out the face portion out of each images, therefore we will get a useful clean dataset. To use the anime-face-detector please follow the 
instructions in the link mentioned above (anime-face-detector's repository).
Run the below command from anime-face-detector's cloned folder (follow complete steps from : [anime-face-detector](https://github.com/qhgz2013/anime-face-detector.git)
```
python main.py -i /path/to/image/or/folder -crop-location /path/to/store/cropped/images -start-output 1
```

After cropping out faces resize the images (64 * 64) using the image_resizer.py script(before running the script make an empty folder where you will store
cropped images)
```
python image_resizer.py 
```

We are done with all the required steps to setup the dataset and make use of it to train our DCGAN.

### Model Structure & Training Process Explanation : 
I will not cover all the basics of GANs (you can easily read theory online), but you can see the Discriminator & Generator Structures in the images below.
Generator Network :
![Discriminator](/discriminator network.png)<!-- -->
Discriminator Network :
![Generator](/generator network.png)<!-- -->


