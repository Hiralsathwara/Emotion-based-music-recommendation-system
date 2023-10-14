
# Emotion based music recommendation system

This web based app written in python will first scan your current emotion with the help of OpenCV & then crop the image of your face from entire frame once the cropped image is ready it will give this image to trained MACHINE LEARNING model in order to predict the emotion of the cropped image.This will happen for 30-40 times in 2-3 seconds, now once we have list of emotion's (contain duplicate elements) with us it will first sort the list based on frequency & remove the duplicates. After performing all the above steps we will be having a list containing user's emotion in sorted order, Now we just have to iterate over the list & recommend songs based on emotions present in the list.


## Installation & Run

Create new project in pycharm and add above files. After that open terminal and run the following command. This will install all the modules needed to run this app. 

```bash
  pip install -r requirements.txt
```

To run the app, type following command in terminal. 
```bash
  streamlit run app.py
```

## Libraries

- Streamlit
- Opencv
- Numpy
- Pandas
- Tensorflow
- Keras


## Screenshots

![app · Streamlit - Google Chrome 28-04-2022 18_08_18 (2)](https://user-images.githubusercontent.com/72250606/165754362-8e0dec51-c42a-4efe-8215-b6cc8c23923c.png)
![Video 28-04-2022 18_09_52](https://user-images.githubusercontent.com/72250606/165754424-492954ca-666e-4430-8504-5d93a5a041ab.png)
![Video 28-04-2022 18_09_57](https://user-images.githubusercontent.com/72250606/165754428-6c22b327-c9a2-401a-8f19-d1838c201777.png)
![Video 28-04-2022 18_09_47](https://user-images.githubusercontent.com/72250606/165754415-3a4559e7-2338-4591-b1dc-159436eeebc4.png)

## Demo video

 [Emotion based music recommendation system](https://youtu.be/eSBsY4WwgGw)
 

## Authors

- [Udhay Brahmi](https://github.com/Udhay-Brahmi)



## Support

For support, email udhaybrahmi786@gmail.com or udhaybrahmi@gmail.com.

