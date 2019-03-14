# Doom project
The idea behind this project is to use reinforcement learning on Vizdoom: https://github.com/mwydmuch/ViZDoom 

Vizdoom is an open-sourced implementation of the video game Doom, and we will be specifically concerned with 8 simplified scenarios. 

To use vizdoom, we need a plugin which make it easy for our reinforcement learning agent to interact with doom. The plugin we will use is one I adapted from an implementation which had become obsolete (see install below).

# Formal stuff
## When and where
We will meet Thursday at 13:00-14:00 (place TBD). According to the instructions Soeren has given, we only have 1 hour per week. I am very happy to listen to suggestions for how we can best make use of this time, however there will obviously be no time to debug code, etc. Therefore, expect to work independently, and make use of online resources.

## What you need to do 

### Git
 - We will communicate/share public information over this git repo, so first make sure you know how git works.
 - each group will select a name (and why not take a name out of dantes inferno? https://en.wikipedia.org/wiki/Inferno_(Dante) ). Make a folder in the `groups` directory in this git repo with your name in it, which is where you will keep project-specific information which can be shared (logbooks, etc.). entries on the leaderboard should be given under this name.
 - Within your `groups/<malebolge-or-whatever>` folder, please create a file called `group_members.txt` with your team members
 - Send me an email with your LDAP login (s123456) to get edit rights on git. Please use command line tools to edit to avoid overwriting each others work.
 - It is okay(ish) to edit the main gdoom env code, but make sure you are 100% backward compatible by default. Look at the registration code in `__init__.py` (the code I mucked aroudn in) to see how to register a new environment (i.e., new resolution, new reward functions, etc.).  
 preferred/proper way to extend environment is to write a new wrapper class and handle special stuff inside it (look at the wrappers dir); this also ensures your changes are optional (and you can do whatever in the wrapper). When making changes outside your groups dir, please use command line tool to get merge warnings.  
 it is, however, okay to add new keys to the info dict as needed

### Upload articles
Each group will implement some sort of RL algorithm for Doom. I got some ideas, but I think a good way to kick things off is with a bit of googling. Therefore, each team should make two-three entries in the file: `articles.md`

Each entry should contain your group name and correspond to an RL method/idea in an RL method which seems suitable/possible/realistic for doom. Make sure the method isn't some crazy deepmind idea that will take a team of 3 people six months to implement and 1000 GPU years, something that won't work for doom (i.e., make sure it is pixel-based), and consider indicating if there is an online implementation, etc. resources can be papers or well-written blog articles. I expect 2 entries per group by week 2, and this should go in the logbook.

### The leaderboard
We will keep a leaderboard for how well we perform on the different vizdoom tasks. Entries to the leaderboard should report kills and time alive (when appropriate), and otherwise time used to complete the task. Since RL requires a lot of compute, we will report our methods in terms of number of rendered frames in the environment. It is very important you report this number since without it, the person with the biggest GPU wins, which is not very sporting. Please note the projects are NOT evaluated by the overall performance, as we should expect systematic difference between the methods (I have some idea, but I really don't know)

### Logboook and contributions to leaderboard
You also need to upload a logbook. This will happen in your own subfolder of this gitlab repo. Soeren should have provided instructions

Your logbook should contain entries of when you make contributions to the scoreboard. I expect each team to make contributions to at least 3 tasks (this will be part of the overall evaluation). (TODO)

### What will our 'innovations' actually consist of?
I think you should have very free hands. I think the standard approach to this project is to implement a RL algorithm (for instance, policy gradient), evaluate it on as many environments as possible, and write a brief report on what you learned. But I think other options are available, which is partly why you get to do a litterature search first (see below). 

For instance, do we really need all this deep learning stuff? Perhaps you can get away with simpler methods (c.f. http://www.ifaamas.org/Proceedings/aamas2016/pdfs/p485.pdf, or perhaps even the critique in http://www.argmin.net/2018/06/25/outsider-rl/ ; can the methods in the blog, random search, be adapted to our setting, perhaps using some kind of vector encoding similar to "Reinforcement Learning", 9.5.3 (see below)?). 

In a similar but less dramatic vein, the entire question of how the observations are represented, or what constitute a suitable resolution etc. etc., is not something I know. I just selected the simplest option that wasn't completely stupid. You can try to explore other options and see if you can get anywhere. 

You can also experiment with how the reward signal is constructed. Note vizdoom (seems!) to allow output of more than just the pixel information; perhaps reward shaping, rather than the specific method, is what really helps. 


# Getting started with reinforcement learning (week 1)
There will be quite a lot to learn, and it need to happend quite fast. Therefore, you have to be selective and fairly focused in how you spend your time. A good idea is to select your overall method. I suggest: Q-type learning (DQN, SARSA, double-Q, Q-lambda, etc.) or policy gradient (Vanilla PG (bad), TRPO, PPO, etc.), and then be fairly selective in your reading. 
At any rate, to get started with the whole 'what is RL' question, look at the first few chapter from:
 http://incompleteideas.net/book/bookdraft2017nov5.pdf

(in the lingo of the book: We are in the episodic, discounted setting with a discrete, small action set and a really nasty set of states which are partially observed)

If you prefer a more condenced format, you could also look at the lectures:
 http://web.stanford.edu/class/cs234/schedule.html

Most relevant for us is the excellent Deep learning course from Stanford, which focuses on deep learning and is very practically oriented:
http://rail.eecs.berkeley.edu/deeprlcourse/
I suggest making some popcorn and focus on lecture 1-9. Use the two previous references for specific questions.

Next, quickly get vizdoom and our openai gym plugin up and running (see install). Make sure you can make a doom instance, render a few frames, and try to play against the computer (note: you got to patch openai gym to get self-play working). Note also the use of Wrappers (and openai gym idea which is pretty neat). This is your first exposure to the sometimes pretty convoluted programming patterns you will find in RL.

Next, realize you need to read code. Yes, this is booring, but my own experience is that my own from-scratch implementations could suck a golfball through a garden hose performance-wise, and looking at other peoples code is very instructive. I therefore suggest focusing on ideas where there is existing code, and you view structuring a RL program as part of the overall task.

This obviously creates a problem with evaluation. The way we will handle this is you make very clear how much of the code you wrote yourself, and which implementations you are close to if it is applicable. I.e., do you feel your implementation is a clean-room implementation? Is it clean-room, but similarly structured to a reference? is it partly re-using another implementation? is it nearly identical with modification? 

# Installation
First, you need to install vizdoom:
https://github.com/mwydmuch/ViZDoom

I kept track of the main steps I used here:

https://gitlab.gbar.dtu.dk/snippets/17

but note this was on a cluster where I had to install SDL2 without admin rights or cmake skills which made it a great deal more cumbersome than it would be on a local machine. 

Secondly, install openai gym, and make sure you can do `import gym`. 

Next, our plugin is located in the directory gdoom as `gdoom/gdoom_env.py` Note the python file contains information on how to use it, as well as the wrappers framework. The plugin is compatible with the baselines architechture, which is included and should be in your pythonpath.

Try to make sure you can render a few frames. For our purpose, each call to the step-method is one simulation tick, i.e. what you need to keep track off when you report performance.

Finally, to get self-play working, I had to patch the play function from openai. Therfore, open: `Python36/Lib/site-packages/gym/utils/play.py` and make a change so the code reads:

```python
    while running:
        if env_done:
            env_done = False
            obs = env.reset()
        else:
            # tue: patch
            action = keys_to_action.get( tuple(sorted(pressed_keys)), -1)
```
When you run the plugin, a screen should pop up and you should be able to shoot your first demon!

If there are any problems, please try to help each other; i.e. feel free to add instructions to the readme file in the gdoom dir.

Note the plugin right now miss a feature to record frames (at the very least, I have not tested the record wrapper). If you wish, I would be very happy if someone would change the code to be able to record frames. If the changes involves the main code, please do so as a copy of the gdoom env with your group name in it.

## FAQ
 - *We'd rather work on Mujoco/sonic/atari/etc.etc.* 
  And you should! Later! The idea of this project is that if we all work on the same problem, we can compare our results. This will allow everyone to learn something about the relative benefits/drawbacks of the other methods. So Doom it is.
