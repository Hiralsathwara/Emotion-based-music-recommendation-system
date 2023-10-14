
# Emotion-based music recommendation system

This web-based app written in Python will first scan your current emotion with the help of OpenCV & then crop the image of your face from the entire frame once the cropped image is ready it will give this image to a trained MACHINE LEARNING model in order to predict the emotion of the cropped image. This will happen for 30-40 times in 2-3 seconds,. once we have a list of emotions (containing duplicate elements) with us it will first sort the list based on frequency & remove the duplicates. After performing all the above steps we will have a list containing the user's emotions in sorted order, Now we just have to iterate over the list & recommend songs based on emotions present in the list.


## Installation & Run

Create a new project in Pycharm and add the above files. After that open the terminal and run the following command. This will install all the modules needed to run this app. 

```bash
  pip install -r requirements.txt
```

To run the app, type the following command in the terminal. 
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




## Demo video

https://github.com/Hiralsathwara/Emotion-based-music-recommendation-system/assets/127468119/d8cf6f4a-bfd6-496e-a3b6-036f1fa608f6



 

## Authors

- [HIRAL SATHWARA](https://github.com/Hiralsathwara)





