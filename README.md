# Anxiety Level Prediction
#### Video Demo: [AnxietySense Demo](https://youtu.be/yrZkjMtTVE8)
#### Description:
Students share a burden of neverending work as they progress through high school and college. Many are unaware by what potential factors increase their buildup of anxiety aside from their work. This project uses a AI model, trained on physiological and health-related indicators, to predict an individual's potential anxiety level. The AI model is trained on a dataset involving (# of ) students and their anxietal factors. Leading factors are including but not limited to sleep quality, blood pressure, headaches, and self esteem. Students will be given a prediction from the AI model on their suggested anxiety levels, most significant factors leading to their anxiety, and potential solutions to dealing with their anxietial factors. Individuals will be able to identify leading factors of their anxiety, if any, and find solutions to lower their anxiety levels after interacting with this project.

<details><summary>Anxiety Factor Scaling (of trained AI model)</summary>

- **Anxiety**: Generalized Anxiety Disorder Assessment (GAD-7) (0-21, 0 is no anxiety, 21 is severe anxiety)
- **Self-Esteem**: Rosenberg Self-Esteem Scale (RSE) (0-30, 0 is high self-esteem, 30 is low self-esteem)
- **Mental Health History**: Binary (0 for no mental history, 1 for mental history available)
- **Depression**: Patient Health Questionnaire (PHQ-9) (0-27, 0 is no depression, 27 is severe depression)
___
- ***Disclaimer***: The mentioned factors (of anxiety) listed below are not officially recognized by any organization. They are declared in this document for context and interpretation. The named scales are not real and are provided for humourous effect (generated by ChatGPT) to conform with the naming conventions of the other official scales mentioned above. The scaling of each topic's intensity, however, are integrated with the AI model's training.
___
- **Headache**: Anxio-Cephalgia Index (ACI) (0-5, 0 is no headache pain, 5 is severe headache pain)
- **Blood Pressure**: Blood Pressure Severity Index (BPSI) (1-3, 1 is normal blood pressure, 3 is high blood pressure)
- **Sleep Quality**: Sleep Quality Assessment Scale (SQAS) (0-5, 0 is no difficulty with sleep, 5 is severe difficulty with sleep)
- **Breathing Problem:** Breath Harmony Index (BHI) (0-5, 0 is unrestricted breathing, 5 is severe breathing impairment)
- **Noise Level**: Tranquil Tone Index (TTI) (0-5, 0 is complete silence, 5 is extreme dissonance)
- **Living Conditions**: Habitat Comfort Meter (HCM) (0-5, 0 is ideal living standard, 5 is extreme adversity)
- **Safety**: Duct Tape Defense Index (DTDI) (0-5, 0 is a safe haven, 5 is critical safety alert)
- **Basic Needs**: Basic Needs Index (BNI) (0-5, 0 is essential fulfillment, 5 is extreme lack of essentials)
- **Academic Performance**: Academic Performance Scale (APS) (0-5, 0 is outstanding achievement, 5 is severe academic crisis)
- **Study Load**: Study Load Rating (SLR) (0-5, 0 is very light academic load, 5 is overwheming study demand)
- **Future Career Concerns**: Fortune Cookie Prophecies Barometer (FCPB) (0-5, 0 is confident career path, 5 is overwhelming career anxieties)
- **Social Support**: Entourage Entanglement Scale (EES) (0-5, 0 is abundant support network, 5 is no social support)
- **Peer Pressure**: Peer Influence Scale (PIS) (0-5, 0 is no peer influence, 5 is extreme peer influence)
- **Extracurricular Activities**: Extracurricular Overachiever-o-Meter (EOM) (0 is no involvement, 5 is active engagement)
- **Bullying**: Occasional Jokester Gauge (OJG) (0-5, 0 is no bullying incidents, 5 is severe and persistent bullying towards the individual)
</details>


<details><summary>Omitted Data (of dataset)</summary>

- **Teacher Student Relationship**: Interpreting this information is somewhat difficult and does not contribute as a considerably significant factor to one's anxiety levels.
- **Stress Level**: The purpose of the project is to assess anxiety levels based on environmental and physiological factors. Individuals' stress levels may make the AI model biased in its predictions for one's anxiety levels. The AI model may completely disregard other factors given stress levels.
</details>

### Implementation
This project utilizes a Flask app, acting as a front-end interface for the end user, to receive and display results of their anxiety levels. The backend process is powered by a Python class that trains an AI model that will predict based on given anxiety factor levels. Inputs from the end user are POSTed from a form whose values are displayed from retrieved attributes and descriptions in a database. The results are are processed through the AI model for its predictions and are sent back to the end user to receive their GAD-7 scaled anxiety level on a new page.

### Analysis of Each Prominent File/Directory
``app.py``: The main Flask file that operates all backend services when routing webpages to the client. Aside from loading the AI model to be used alongside the web app, this file is responsible for providing the webpages with sufficient data personalized to each client. It takes any important information stored from a db file in the static directory and routes it to a Flask template for further manipulation. This file also starts Flask and notifies Flask where to find the webpages in the filesystem.

``ap_backend.py``: The Python file responsible for building an AI model and handling predictions when given. The file is made from scratch, so imperfections are common in the code. This file is mainly responsible for manipulating instances of the AI model, with methods like viewing its dataset and converting GAD-7 numbers to category names for troubleshooting. A test scenario is briefly documented and available to use when running the program to get a gist of how the AnxietyPredictor class can be used.

``anxiety_factors_info.db`` Located in the static directory, this database file holds information to display on the webapp's form for AI prediction. Rather than store multiple anxietal factors in the form's template file, the database file makes it easier to access and add new factors for when any changes occur with the AI model's training.

``templates`` The directory that holds the webpages you see when running the webapp. It mainly consists of a homepage, a form to submit your anxietal factors, and a result page that shows what the AI predicts your anxiety might be on the GAD-7 scale.

``model`` This directory is intended to store the AI model as a ``.keras`` file for easy loading when using the webapp. Rather than build a new model every time the webapp is run, this directory makes it handy for start-up times to decrease when using an existing model. You are free to build the model with your own computer or server, though, I will provide a model file for those that cannot compute its training in a timely manner.

``css`` This directory holds the styling files that make the webpages look fancy. They are a work in progress and are not perfect but I am open to suggestions on improving this directory's proper use of CSS.

``images`` Self-explanatory. This directory holds images used for the webapp that you will see mostly throughout the homepage and on the form page.

``js`` This directory handles all the slider bars and animations for the form page to help you easily progress through the questionaire that is given.

### Usage
Execute ``app.py`` and navigate to your local IP (or [localhost](localhost:8080)) on port 8080.

### Dependencies (tested)
```
Flask (3.0.0)
tensorflow (2.15.0 w/ CUDA)
pandas (2.1.4)
numpy (1.26.2)
scikit-learn (1.3.2)
notebook* (7.0.6)
ipykernel* (6.28.0)
```
\* - Starred dependencies are essential for running the proof of concept Jupyter Notebook in the Backend Visualizer directory


### Suggestions
The implementation of this project is open to suggestions in the [Error](https://github.com/poibear/AnxietyPredictor/errors) section of this repository. The code is prone to sluggish execution time and is always open to optimization in all aspects (especially backend processes).

## Attribution & License
This project is licensed under the MIT License. Additionally, this project's dataset, "Student Stress Factors: A Comprehensive Analysis," is licensed under the Apache License 2.0. Refer to the [LICENSE](LICENSE.txt) file regarding appropriate usage on this project and the code snippets from Youtube alongside the [APACHE LICENSE](APACHE_LICENSE.txt) file for more details on usage of the dataset. Below are the resources used to compile this web app (it may help to learn something new from these sources).
- **ChatGPT**
    - Source: [OpenAI](https://openai.com/)
    - License: None
    - Link: [chat.openai.com](https://chat.openai.com) 
    - Modifications: None
    - Use case: Generate relevant text to the project that compels the reader
- **Student Stress Factors: A Comprehensive Analysis**
    - Source: Kaggle (Uploader: [rxnach](https://www.kaggle.com/rxnach))
    - License: Apache 2.0
    - Link: [Kaggle Dataset](https://www.kaggle.com/datasets/rxnach/student-stress-factors-a-comprehensive-analysis/)
    - Modifications: Removal of Teacher-Student Relationship and Stress Level attributes for training of AI model
- **Multi Step Form Using HTML, CSS & Javascript**
    - Source: Youtube (Uploader: [WEB CIFAR](https://www.youtube.com/@webcifar))
    - License: MIT License (under [CodePen.io's Public Licensing](https://blog.codepen.io/documentation/licensing/))
    - Link: [How to Create Multi Step Form Using HTML, CSS & JavaScript](https://www.youtube.com/watch?v=cKTgIDkRsGc)
    - Modifications: Written mainly in JQuery instead of pure JS
- **Custom Range Slider**
  - Source: Youtube (Uploader: [MinzCode](https://www.youtube.com/@minzcode))
  - License: MIT License (under [CodePen.io's Public Licensing](https://blog.codepen.io/documentation/licensing/))
  - Link: [Video: Custom Range Slider - HTML + CSS + JS](https://www.youtube.com/watch?v=gjPllrhIYsM), [Code at CodePen.io](https://codepen.io/MinzCode/pen/rNxYYOZ)
  - Modifications: Adjusted bar fill to match with sliders on result page (out of 100 to out of 21)