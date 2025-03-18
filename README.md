
# Volta-Ros Simulation

workspace setup and commands to run the volta simualtion

## Required simulation setup:
Python=>3.8 (prefreble 3.10)

Ros_distro : Noetic

Install all required ros pkgs for the simualtion


## Run Locally

Clone the project

```bash
  git clone https://github.com/Praveenkottari/volta_ws.git
```

Go to the project directory

```bash
  cd volta_ws
```

Install dependencies

```bash
  rosdep install --from-paths src --ignore-src -r -y
```

catkin_make the packages

```bash
  catkin_make
```
source ros/noetic
```bash
  source /opt/ros/noetic/setup.bash
```
source volts_ws
```bash
  source devel/setup.bash
```
## Launch simulation

launch gazebo with world  
```
 roslaunch volta_simulation gazebo.launch
```
spawn volta robot
```
 roslaunch volta_simulation simulation.launch
```
For visualization, launch rviz
```
 rosrun rviz rviz 
```
for teleoperation node
```
roslaunch volta_teleoperator teleoperator.launch keyboard:=true
```
to move around use,
* u i o
* j k l
* m , .
