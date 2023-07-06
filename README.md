# Used car price prediction

### Software and tools requirements

1. [Github Account](https://github.com)
2. [VS Code IDE](https://code.visualstudio.com/download)
3. [AWS Account](https://aws.amazon.com/)
4. [Git Bash](https://git-scm.com/downloads)
5. [Anaconda](https://www.anaconda.com/products/individual)

### Steps to access the web app from your local machine:
1. Create a new repository on Github
2. Creata a new folder on your local machine and clone the repository
   ``` 
   git clone <repo_url>
   ```
3. Create a new environment in Anaconda
   ```
    conda create -n <env_name> python=3.8 -y
    ```
4. Activate the environment
   ```
   conda activate <env_name>`
   ```
5. Install the required libraries
   ```
   pip install -r requirements.txt
   ```
7. Make sure the serialized trained model (a pickle file) and all the datasets are in the 'artifact' folder

8. Run the app.py file
   ```
   python app.py
   ```
9. If the flask installation was completed successfully, the web app should be accessible now from localhost
10. Visit localhost:8080 to view the webpage
11. Provide necessary input and get a prediction of your used car
