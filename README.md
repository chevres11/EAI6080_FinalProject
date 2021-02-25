![Northeastern University](./Northeastern%20Logo.png)


# Final Project for EAI6080 Advanced Analytical Utilization
#### Craig Bunce, Miguel A. Chevres, Zihao Zhang 
#### EAI 6080 - Advanced Analytical Utilization
#### Dr. Maria Wang
#### Northeastern University College of Professional Studies



## Imitation Learning (Option 2)

### Introduction and Dependencies Installation
This portion of the assignment has to be the most complex one of both, not only because of the increase 
in challenge in order to complete it, but the intricacies of working with several API's and frameworks at the same time
is a challenge that not a lot of assignments entail.  For this assignment in particular we had to do 3 tasks in order to 
complete it, each with its own complications and difficulties.  


The first portion of this assignment was to create a conda environment where we can install all the dependencies in order
to make the code work.  The dependencies for this portion are: 
* [TensorFlow](https://www.tensorflow.org/install)
* [OpenAI Gym](https://gym.openai.com/docs/)
* [MuJoCo](http://mujoco.org/)

In order for us to create an environment where all three could be present we utilized Anaconda.  Anaconda is a distributor
of `Python` and `R` programming language that is mostly utilized for data science projects.  It could be utilized for a plethora
of other projects, but it became popular as a data science platform.  The advantage of utilizing anaconda is that it is extremely
easy to create new python environments each with its own characteristics and dependencies (you can even specify the version of 
Python for that environment).  The reason we want to create different environments for each project, or for projects that are 
different conceptually to one another, is because sometimes the dependencies that are required for projects can interfere and collide. 
Therefore, if you have your base environment (the one that comes already installed when you download Anaconda, the basic one), and you 
start installing all the dependencies and libraries that are required for various projects they might not work and the kernel might 
shutdown.  This is why the environments are usually designated for specific projects or a group of similar projects.  For this one we 
created a new `conda` environment specifically for this project.  In order to do that we utilized the command `conda create --name
<environment name>`.  This will install all the basic requirements needed to run a python script that is written with native python language
(no libraries imported).  Once the environment was created we had to activate it by running this command `conda activate <environment name>`. 
With the environment activated we were able to install all the dependencies that were required in order to run the scripts: 
* Numpy 
* Scipy
* MatplotLib
* Theano
* Keras
* Tensorflow
* OpenAI Gym
* iPython
* mujoco-py

The versions of each dependency can be seen on the file location "/Option 2/requirements.txt".  Once we had all the libraries installed
we needed to download the physics simulator [MuJoCo](http://mujoco.org/).  `MuJoCo` is an acronym for **Mu**lti-**Jo**int dynamics with **Co**ntact. 
This platform is being developed by the Emo Todorov for Roboti LLC and serves as a physics engine simulator in order to facilitate research 
in robotics, biomechanics, graphics, and animation.  This platform has been widely utilized by Artificial Intelligence Research and Development 
companies such as OpenAI for a significant amount of their research.  A great example of `MuJoCo` being utilized for a research paper can be seen in 
the OpenAI paper titled ["Emergen Tool Use From Multi-Agent Autocurricula"](https://arxiv.org/abs/1909.07528). In this [paper](https://openai.com/blog/emergent-tool-use/)
they developed an environment for self-supervised AI agents to learn how to play Hide-And-Seek, with results that they did not even expect 
before the training started.  These agents were capable of finding the bugs in the MuJoCo environment and breaking it in order to "seek" 
the agent that was hiding.


### Warmup
Once Everything was downloaded and installed; the next portion of the assignment was to run the Warmup.  
Throughout this portion is when we started experimenting and learning how the MuJoCo environment works.  
The main task for this problem was to run the behavioral cloning (BC) on the Hopper-v1 environment and plot 
the loss function for the behavior cloning objective versus the number of learning iterations (epochs).  Before we started 
to work with the code, the first thing that had to be done is activate the environment we were going to be working on.  To do that 
we went to the Anaconda PowerShell and wrote this command: `conda activate gym`.  Once our environment was activated we went over to the IDE 
that we were going to utilize for this project.  In order to run the code we had to utilize the command line to activate the MuJoCo environment 
and identify all the parameters that were going into the `main` function.  The parameters that needed to be inputted in the Terminal are these: 
* python file name (Ex: run_clone.py)
* File location of the Pickle environment (Ex. experts/)
* Name of the environment + the command `--render` (Ex: Hopper-v1 --render)
* Number of Rollouts utilizing this command: `--num_rollouts #` (Ex. --num_rollouts 3)

For this first portion the command utilized in order to render the clone of the Hopper was `run_clone.py experts/Hopper-v2.pkl Hopper-v1 --render --num_rollouts 3`. 
The expert Hopper (Image 1), which was rendered doing the exact same process as mentioned above but instead of running the 'run_clone.py' file we ran the 'run_expert.py',
had a mean of 3778 and an STD of 2.8.  The clone hopper (Image 2) had a mean of 737 and STD of 34.5, and in Image 3 you can see how the training loss function
decreased as the number of iterations increased, representing the Learning Curve, with a final loss value of 0.0325 on the 300th epoch.  



Image #1: Expert Hopper with Mean = 3778 and STD = 2.8


![Expert Hopper](./Option%202/Expert%20Hopper%20mean=3778%20std=2.8.gif "Expert Hopper")


Image #2: Warmup Hopper with Mean = 737 and STD = 34.5


![WarmUp Hopper](./Option%202/Warmup%20Hopper%20mean=737%20std=34.5.gif "Warmup Hopper Mean = 737")


Image #3: Learning Curve for Warmup Clone Hopper


![Learning Curve](./Option%202/Picture1.png "Warmup Training Learning Curve")



### Behavioral Cloning (BC)
#### Objective 1: Run behavioral cloning (BC) and report results on two other tasks - one task where a behavioral cloning agent achieves comparable performance to the expert, and one task where it does not.  When providing results, report the mean and STD of the return over multiple rollouts in a table, and state which task was used.  Be sure to set up a fair comparison, in terms of network size, amount of data, and number of training iterations, and provide these details (and any other you feel are appropriate) in the table caption. 

When comparing the outcomes between the ant and the cheetah testing, the results where significantly different.  With the networks and parameters between both tests
remaining the same we can see a significant difference between the mean and standard deviation of the cheetah versus the ant.  Out of both renders, the one that was closest
to the expert was the cheetah.  The models were trained on 10,000 training samples for both the cheetah and the ant and both had a total of 300 epochs.
The expert cheetah(Image #4) had a mean of 3981 on 1 rollout, while the clone cheetah (Image #5) had a mean of 3262 and Standard deviation of 246 over 10 rollouts.
The results from the expert and clone cheetah are comparable and almost equilateral, it can be seen in the images that both cheetah's behave almost identical and the properties are 
extremely similar.  


Image #4: Expert Cheetah with Mean = 3981

![Expert Cheetah](./Option%202/Expert%20Cheetah%20mean=3981.gif "Expert Cheetah")



Image #5: Clone Cheetah with Mean = 3262 and STD = 246


![Clone Cheetah 2](./Option%202/Clone%20Cheetah%20mean=3262%20std=246.gif "Clone Cheetah Mean=3262")


On the other hand, we have the ant renders.  The ant, while maintaining the same parameters utilized to create the cheetah did not perform as well as the expert ant, in fact it performed significantly worse. 
The clone ant showed results of barely being able to move forward in the plane before stopping and glitching completely.  The expert ant(Image #6) had a mean
of 4839 over 1 rollout, while the clone ant(Image #7) has a mean of 877 with a standard deviation of 84 over 10 rollouts.  Our hypothesis for this is that the ant has 
completely different physics for moving than the cheetah, given the center of gravity and the position of the legs, so we believe that without changing any other parameter
but the amount of epochs we give the model in order to train and lower the loss function we can generate better results for the ants.  This hypothesis was tested on Objective 2, where we 
elaborate more on the results. 

Image #6: Expert Ant with Mean = 4839

![Expert Ant](./Option%202/Expert%20Ant%20mean=4839.gif "Expert Ant")


Image #7: Clone Ant with Mean = 877 and STD = 84

![Clone Ant](./Option%202/Clone%20Ant%20mean=877%20std=84.gif "Clone Ant Mean = 877")



#### Objective 2: Experiment with one hyperparameter that affects the performance of the behavioral cloning agent, such as the number of demonstrations, the number of training epochs, the variance of the expert policy, or something that you come up with yourself.  For one of the tasks used in the previous question, show a graph of how the BC agent's performance varies with the value of this hyperparameter, and state the hyperparameter and a brief rationale for why you chose it in the caption for the graph. 

Image #8: Clone Ant with Mean = 2137 and STD = 1357

![Clone Ant 2](./Option%202/Clone%20Ant%20mean=2137%20std=%201357.gif "Clone Ant Mean = 2137")



Image #9: Clone Cheetah with Mean = 1543 and STD = 1152


![Clone Cheetah](./Option%202/Clone%20Cheetah%20mean=1543%20std=%201152.gif "Clone Cheetah Mean = 1543")




If you wish to replicate this project, you would need to go to your Anaconda Powershell Prompt and create an environment.  Download this
[repository](https://github.com/chevres11/EAI6080_FinalProject) onto your computer in a location that can be accessed by your Anaconda shell.  Activate the environment, make sure you are in the file
location of where the repository is, open folder `Option 2` and run this command on the Powershell Prompt: `pip install -r requirements.txt`. 
This will download and install all the requirements needed in order to run this project successfully.



