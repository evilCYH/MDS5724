1. Develop the data mining pipeline for topic classification and dump/save the pipeline to be deployed, and then test the saved model locally. 
[Note: Record and submit your 5-fold CV performance (weighted F1 score) on the training data]
	Step-1: Develop and save pipeline as in "/docker_demo/app/model/topic_model_dev.py"
	Step-2: Reload and test the saved pipeline as in "/docker_demo/app/model/topic_model_test.py" 
	
2. Load the saved pipeline and deploy it as an API service, and then test the API service locally. 
	Step-1: Install python (3.12 or higher) and required packages as specified in "docker_demo/requirements.txt" 
	Step-3: Enter the path of "/docker_demo/app".
	Step-4: Run "python main.py" to start the API service.
	Step-5: Test the API service locally as in "/docker_demo/app/test/api_test.py" with API port as 5724

3. Build the Docker image for your pipeline and the working environment, and then run/test the Docker image
	Step-1：Build the Docker via "docker build -f Dockerfile_python ." (Note that the command ends with the dot ".")
	Step-2: Run and test the Docker with command "$docker run -d -p 9000:5724 --memory=600m [IMAGE ID]"
	Step-3: Test the API service locally as in "/docker_demo/app/test/api_test.py" with API port as 9000

4. Push the Docker image to a PUBLC repository of your own Alibaba Cloud Container Registry (ACR) account. 
[Note: Record and submit the URL of your Docker image in ACR]
	Step-1：Refer to the lecture notes in Week-13

5. Pull and test(/run) the Docker image from the PUBLC repository in ACR as above.
	Step-1：Test the API service locally as in "/docker_demo/app/test/api_test.py" with API port as 9000


