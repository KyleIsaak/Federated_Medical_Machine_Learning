Please see explanation video using the following link:
https://youtu.be/28k4t8mhJn8

To run the project yourself or view the code, open the "Code" directory.

Federated Learning in the Clinical Environment
In recent years, the desire to keep one's data private and secure has become highly sought after in our society. One aspect of this is the privacy of medical data. While patient data can be used by Artificial Intelligence (AI) and Machine Learning (ML) models to better understand, diagnose, and treat diseases, stricter access to patient data adds an additional layer of complexity to using these techniques.

Federated Learning (FL) is a technique that allows patient data to remain private, while still using that data to build and improve ML models. This project explores and implements a proof of concept of FL, which could then be adapted to a clinical environment to build ML models to help improve patient care. Specifically we used MRI images of brains with 4 different classifications of tumors. We used Tensorflow framework to create our ML models and we used Flower framework for the FL methods. Our FL model accuracy results were within 10% of our centralized model results, although in a realistic strict data privacy environment there was an improvement. Our results show that there is definitely value in exploring FL methods, as we have room for improvement and other research applied to the current pandemic shows the possible efficacy of FL in a clinical environment.
