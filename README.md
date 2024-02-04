
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





## Demo video

 [Emotion-based music recommendation system](

https://github.com/Hiralsathwara/Emotion-based-music-recommendation-system/assets/127468119/5b2626f6-68fe-499c-9c3e-56b37f7c5d26

)
 

## Authors

- [HIRAL SATHWARA]([https://github.com/Udhay-Brahmi](https://github.com/Hiralsathwara)



## Support

For support, email udhaybrahmi786@gmail.com or udhaybrahmi@gmail.com.

