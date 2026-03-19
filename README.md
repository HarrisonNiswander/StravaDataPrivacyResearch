# StravaDataPrivacyResearch
Investigating the Privacy Concern in Automating Sensitive Locations Utilizing Strava Data. 

Apart of my Purdue Fort Wayne Honor's Project and Ethics of AI (CS 59300).

Abstract:

Strava is a popular website that allows users to upload and track a wide variety of activities like running, cycling, hiking, and more using GPS data. Within the site, users can view other activities uploaded by other users. Any activities tracked with GPS data that is uploaded shows a digital map of the activity and provides a deep analysis of the activity like heart rate, pace, and more. Allowing users to post activities utilizing GPS has its own inherit risk with privacy. If users upload activities that are around where they live, this could give potential criminals sensitive information about where they live, work, or frequent. Strava does have a built-in privacy feature where you can hide the start and end portions of your run to hide sensitive locations.  

In my research, I plan to explore the potential privacy risks associated with activities uploaded to Strava utilizing GPS data. Utilizing the Strava API, I will access different Strava users and pull their geolocation data from their activities. To gather this data, I will develop a python code that will allow me to gather the data from their Strava activities and organize it into a .csv file. I plan to use Strava data from my peers to have a wide variety of data and map visibility features. 

Using these data sets, I will train a machine learning model and have them identify locations that activities frequently start or end at. I also plan to use the built-in privacy feature to test the effectiveness of the machine learning model to identify sensitive locations with the start and end points hidden. Additionally, I plan to train a deep learning model on the data sets collected and try to identify locations frequented at certain times of the day from the GPS data collected while running. 
