# Employee Face Recognition System

The face recogntion system is used to train the faces of employees and detect them when they walk past the camera.
It can be used as an attendence system for the employees.

The system is built using python libraries dlib and face_recognition. Django framework is used to develop the front end for the system.


## Usage

Firstly, to run the django server, open the terminal and go the following directory

```bash
cd face_rec_django
```

and run the following command

```bash
python manage.py runserver
```

Then go to the browser and enter the url **http://127.0.0.1:8000/**

&nbsp;



## Steps to navigate website




**1) Identify faces**

This option will bring up the webcam and capture faces. So leave this open in the background to continuously detect faces.
To close the webcam, click the webcam window and press Q on the keyboard

&nbsp;


**2) Detected faces**

All the employess that are detected will be displayed here. The records can be viewed by date.



&nbsp;

**3)Add photos**

This is used to train faces of employees. Enter the id of the employee and the webcam will popup. If the employee details are not found then go to step 4 to add new employee.

For best results follow these steps

- Take 15-20 images (press space bar to click images and you can open the terminal to see number of images clicked)
- For each image slightly change the angle of your face and the distance to the camera. 

To close the webcam, click on the webcam window and press ESC key.

**NOTE:** if the webcam window doesn't popup, restart the django server by opening the terminal and pressing CTRL+C and run the command 

```bash
python manage.py runserver
```

&nbsp;

**4)Add employee**

New employee details can be added here


&nbsp;

**5)Train model**

If new employee images are captured, then the model needs to be trained again. Once you click this option, open the terminal to check the progress of the training.
