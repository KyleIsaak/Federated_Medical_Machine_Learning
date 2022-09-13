# Federated Learning in the Clinical Environment

In recent years, the desire to keep one's data private and secure has become highly sought 
after in our society. One aspect of this is the privacy of medical data. While patient data 
can be used by Artificial Intelligence (AI) and Machine Learning (ML) models to better understand, 
diagnose, and treat diseases, stricter access to patient data adds an additional layer of 
complexity to using these techniques. 

Federated Learning (FL) is a technique that allows patient data to remain private, 
while still using that data to build and improve ML models. This project explores and 
implements a proof of concept of FL, which could then be adapted to a clinical 
environment to build ML models to help improve patient care. Specifically we used 
MRI images of brains with 4 different classifications of tumors. We used Tensorflow 
framework to create our ML models and we used Flower framework for the FL methods. 
Our FL model accuracy results were within 10\% of our centralized model results, 
although in a realistic strict data privacy environment there was an improvement. 
Our results show that there is definitely value in exploring FL methods, as we have 
room for improvement and other research applied to the current pandemic shows the possible 
efficacy of FL in a clinical environment.


## Installation
Our project has been coded in python3. See requirements.txt for all the required python packages.
The requirements can be installed with:

```bash
pip install -r requirements.txt
```

## Running the code (Federated Learning)

Note: A Linux environment is necessary to run this code (either a VM or WSL will work)

1) In a terminal, start the central server using:
```bash
python3 server.py
```

2) Next, open up 4 more terminals, one for each of the 4 Clients. Start each client in its own terminal using the commands:
```bash
python3 client.py 0
python3 client.py 1
python3 client.py 2
python3 client.py 3
```

The Federated Learning process will only begin once all 4 clients are connected.

After a brief pause, the server should start sampling the clients, who will each be training Machine Learning models!

## Running the code (Machine Learning)

All the code for our individual Machine Learning Models is located in the Jupyter Notebook: mri_classification.ipynb


## Authors
- Coral, G
- Hait, J
- Isaak, K
- Watkins, A