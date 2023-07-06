# Used car price prediction

### Software and tools requirements

1. [Github Account](https://github.com)
2. [VS Code IDE](https://code.visualstudio.com/download)
3. [AWS Account](https://aws.amazon.com/)
4. [Git Bash](https://git-scm.com/downloads)
5. [Anaconda](https://www.anaconda.com/products/individual)

### Steps to access the web app from your local machine:
1. Creata a new folder on your local machine and clone the repository
   ``` 
   git clone <repo_url>
   ```
2. Create a new environment in Anaconda
   ```
    conda create -n <env_name> python=3.8 -y
    ```
3. Activate the environment
   ```
   conda activate <env_name>`
   ```
4. Install the required libraries
   ```
   pip install -r requirements.txt
   ```
5. Make sure the serialized trained model (model.pkl file) and all the datasets are in the 'artifact' folder

6. Run the app.py file
   ```
   python app.py
   ```
7. If the flask installation was completed successfully, the web app should be accessible now from localhost
8. Visit localhost:8080 to view the webpage
9. Provide necessary input and get a prediction of your used car
