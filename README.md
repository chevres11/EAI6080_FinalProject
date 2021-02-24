![Northeastern University](./Northeastern%20Logo.png)


# Final Project for EAI6080 Advanced Analytical Utilization
#### Craig Bunce, Miguel A. Chevres, Zihao Zhang 
#### EAI 6080 - Advanced Analytical Utilization
#### Dr. Maria Wang
#### Northeastern University College of Professional Studies



## Imitation Learning (Option 2)

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

Once Everything was downloaded and installed; the next portion of the assignment was to run the Warmup.  Throughout this portion is when we started experimenting and learning how the MuJoCo environment works.  The main task for this problem was to run the behavioral cloning (BC) on the Hopper-v1 environment and plot the loss function for the behavior cloning objective versus the number of learning iterations (epochs).  For our assessment

![WarmUp Hopper](./Option%202/Warmup%20Hopper%20mean=737%20std=34.5.gif "Warmup Hopper Mean = 737")


Image #1: Warmup Hopper with Mean = 737 and STD = 34.5

![Clone Cheetah](./Option%202/Clone%20Cheetah%20mean=1543%20std=%201152.gif "Clone Cheetah Mean = 1543")


Image #2: Clone Cheetah with Mean = 1543 and STD = 1152

![Clone Cheetah 2](./Option%202/Clone%20Cheetah%20mean=3262%20std=246.gif "Clone Cheetah Mean=3262")


Image #3: Clone Cheetah with Mean = 3262 and STD = 246

![Clone Ant](./Option%202/Clone%20Ant%20mean=877%20std=84.gif "Clone Ant Mean = 877")


Image #4: Clone Ant with Mean = 877 and STD = 84

![Clone Ant 2](./Option%202/Clone%20Ant%20mean=2137%20std=%201357.gif "Clone Ant Mean = 2137")


Image #5: Clone Ant with Mean = 2137 and STD = 1357

If you wish to replicate this project, you would need to go to your Anaconda Powershell Prompt and create an environment.  Download this
[repository](https://github.com/chevres11/EAI6080_FinalProject) onto your computer in a location that can be accessed by your Anaconda shell.  Activate the environment, make sure you are in the file
location of where the repository is, open folder `Option 2` and run this command on the Powershell Prompt: `pip install -r requirements.txt`. 
This will download and install all the requirements needed in order to run this project successfully.



