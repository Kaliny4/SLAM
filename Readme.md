# Visual Slam - getting started document

This is an implementation of a basic visual slam system.
The implementation is based on opencv and the g2o optimization library.

For questions contact Henrik Skov Midtiby, hemi@mmmi.sdu.dk

To launch the visual slam system, ensure that the dependencies are installed.
These include:
- g2o-python
- opencv-python
- pillow
- pyopengl
- pygame

To launch the program, execute the following command on the command line
```
python visual_slam.py path_to_directory_with_images
```

When the program is launched, it opens two windows.
The first window show the matches between the current and the previous frame.
The second window contains a 3D visualization of the camera locations and the reconstructed 3D points.

The window with the 3D visualization captures your mouse movements, and will not let you remove the mouse from its center.
To pause this behaviour press 'p'.
The camera placement can be altered using the keys: wsadqe
The camera orientation can be altered by moving the mouse.

To go to the next image press space.
To exit the program press escape.


## Example data set

An example data set can be obtained from this link
[https://nextcloud.sdu.dk/index.php/s/yRnA3xJMEq8DGsH].


## How to install and launch the program

```
git clone git@gitlab.sdu.dk:midtiby/basic-visual-slam.git
cd basic-visual-slam
python3 -m venv env 
source env/bin/activate
pip install -r requirements.txt
python visual_slam.py input/frames_limited/
```
where `input/frames_limited/` is a path to a directory containing the image 
sequence on which the visual slam system should operate. 


## Running the program under Windows Subsystems for Linux (WSL)

To be able to run the program under Windows Subsystems for Linux (WSL), you 
need to install an X server on windows that supports xinput 2.

One option for this is the VcXsrv [https://sourceforge.net/projects/vcxsrv/].
After installing the X server, you need to let your WSL installation know the location of the X server.
This is achived by executing the following command in wsl.
	
```
export DISPLAY=$(ip route|awk '/^default/{print $3}'):0.0
```

You should now be able to launch the visual slam program as follows (assuming 
that your images are in the `input/frames_limited` directory.
```
python visual_slam.py input/frames_limited/
```
