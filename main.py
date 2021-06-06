import tkinter as tk
import cv2
from PIL import ImageTk, Image
import PIL
from threading import Thread
from face_detector import *
import time
import numpy as np
from sklearn.svm import SVC
import pickle
import datetime

file_install = os.getcwd() + "\\data\\"
NUM_OF_PICTURE = 50
classifier_filename_exp = file_install + 'facemodel.pkl'
log_file = os.path.join(file_install, "log.csv")
face_detector = FaceDetector()
cap = cv2.VideoCapture(0)

try:
	with open(classifier_filename_exp, 'rb') as file:
		_, class_names = pickle.load(file)
except:
	class_names = []

try:
	X = np.load(file_install + 'dataset.npy')
	Y = np.load(file_install + 'label.npy')
except:
	X = np.zeros(shape = (0, 512))
	Y = np.zeros(shape = (0))

def show_camera():
	_, frame = cap.read()
	frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
	# try:
	# 	box = face_detector.get_boxes(frame)[0]
	# 	cv2.rectangle(frame, (box[0], box[1]), (box[2], box[3]), (0, 255, 0), 2)
	# except:
	# 	pass
	frame = cv2.flip(frame,1)
	img_from_array = PIL.Image.fromarray(frame)
	imgtk = ImageTk.PhotoImage(image = img_from_array)
	lb_camera.imgtk = imgtk
	lb_camera.configure(image=imgtk)
	lb_camera.after(10, show_camera)

def disable_button():
	if btn_register["state"] == "normal":
		btn_register["state"] = "disable"

	if btn_remove["state"] == "normal":
		btn_remove["state"] = "disable"

def enable_button():
	if btn_register["state"] == "disabled":
		btn_register["state"] = "normal"

	if btn_remove["state"] == "disabled":
		btn_remove["state"] = "normal"

def get_frame():
	# t=time.time()
	global X
	global Y
	global class_names
	name_of_person = text_name.get('1.0', 'end-1c')
	if name_of_person in class_names:
		label_error['text'] = 'Name has been added'
		return
	elif(name_of_person ==''):
		label_error['text'] = 'Please enter name'
		return
	else:
		label_error['text'] = 'Waiting'

	disable_button()
	
	face_of_person = np.zeros(shape = (0, 512))
	while(face_of_person.shape[0] <NUM_OF_PICTURE):
		face = get_frame_2()
		if face is not None:
			face_of_person = np.concatenate((face_of_person, [face]), axis=0)

	y_temp = np.ones(NUM_OF_PICTURE)*(Y[-1]+1 if len(Y)>0 else 0)
	class_names.append(name_of_person)
	Y = np.concatenate((Y, y_temp), axis=0)
	
	X = np.concatenate((X, face_of_person), axis=0)
	if(len(class_names)>1):
		training_model(X, Y, class_names)
	# print(time.time() - t)

	enable_button()

	text_name.delete(1.0,"end")
	text_name.insert(1.0, "")
	label_error['text'] = ''


def get_frame_2():
	_, frame = cap.read()
	result = face_detector.extract_face(frame)
	return result

def training_model(X, Y, class_names):
	np.save(file_install + 'dataset.npy', X)
	np.save(file_install + 'label.npy', Y)

	model = SVC(kernel='linear', probability=True)
	model.fit(X, Y)

	with open(classifier_filename_exp, 'wb') as outfile:
		pickle.dump((model, class_names), outfile)

def predict_image():
	with open(classifier_filename_exp, 'rb') as file:
		model, class_names = pickle.load(file)

	# t = time.time()
	_, frame = cap.read()
	result = face_detector.extract_face(frame)

	box = face_detector.get_boxes(frame)[0]
	frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
	face = frame[int(box[1]):int(box[3]), int(box[0]):int(box[2])]
	face = cv2.resize(face, (lb_image.winfo_width(), lb_image.winfo_height()))

	img_from_array = PIL.Image.fromarray(face)
	imgtk = ImageTk.PhotoImage(image = img_from_array)
	lb_image.imgtk = imgtk
	lb_image.configure(image=imgtk)
	right_frame.update()

	predictions = model.predict_proba([result])

	best_class_indices = np.argmax(predictions, axis=1)
	best_class_probabilities = predictions[
		np.arange(len(best_class_indices)), best_class_indices]
	best_name = class_names[best_class_indices[0]]
	# print(time.time() - t)
	# if float(best_class_probabilities) > 0.7:
	# 	label_login_name['text'] = "Name: " + best_name
	# 	label_login_prob['text'] = "Probability: %.2f" % float(best_class_probabilities)
	# 	print("Name: {}, Probability: {}".format(best_name, best_class_probabilities))
	# else:
	# 	label_login_name['text'] = "Name: UNKNOWN"
	# 	label_login_prob['text'] = "Probability: %.2f" % float(best_class_probabilities)
	# 	print("UNKNOWN", float(best_class_probabilities))
	# 	print("Name: {}, Probability: {}".format(best_name, best_class_probabilities))

	label_login_name['text'] = "Name: " + best_name
	label_login_prob['text'] = "Probability: %.2f" % float(best_class_probabilities)

	with open(log_file, 'a') as f:
		now = datetime.datetime.now().strftime("%d_%m_%Y_%H_%M_%S")
		f.write(now + ','+ best_name + '\n')
		f.close()

def remove():
	global X
	global Y
	global class_names
	name_of_person = text_name.get('1.0', 'end-1c')
	if name_of_person not in class_names:
		label_error['text'] = 'This name is not in database'
		return
	elif(name_of_person ==''):
		label_error['text'] = 'Please enter name'
		return
	else:
		label_error['text'] = 'Waiting'

	disable_button()

	index_of_person = class_names.index(name_of_person)
	label_error['text'] = class_names.index(name_of_person)
	X = np.delete(X, slice(index_of_person * NUM_OF_PICTURE, (index_of_person+1) * NUM_OF_PICTURE), 0)
	Y = np.delete(Y, slice(index_of_person * NUM_OF_PICTURE, (index_of_person+1) * NUM_OF_PICTURE), 0)
	class_names.remove(name_of_person)
	if(len(class_names)>1):
		training_model(X, Y, class_names)

	enable_button()

	text_name.delete(1.0,"end")
	text_name.insert(1.0, "")
	label_error['text'] = ''




def open_camera():
	threadShowCam=Thread(target=show_camera)
	threadShowCam.start()


def new_image():
	thread_get_frame=Thread(target=get_frame)
	thread_get_frame.start()




root = tk.Tk()
root.title('Face Recognition')
root.geometry("+100+100")

canvas = tk.Canvas(root, height=700, width=1300)
canvas.pack()

left_frame = tk.Frame(root)
left_frame.place(relwidth=0.5, relheight=0.7)

register_frame = tk.Frame(root, bg='white')
register_frame.place(relwidth=0.5, relheight=0.3, rely=0.7)

right_frame = tk.Frame(root)
right_frame.place(relwidth=0.5, relheight=1, relx=0.5)

lb_camera = tk.Label(left_frame, bg='gray')
lb_camera.place(relwidth=1, relheight=1, relx=0, rely=0)


lb_id = tk.Label(register_frame, text = "ID:", font = ("Courier, 24"), bg='white')
lb_id.place(relx=0.1, rely=0.05)

text_name = tk.Text(register_frame, font = ("Courier, 24"))
text_name.place(relx=0.2, rely=0.05, relheight=0.2, relwidth=0.55)

btn_register = tk.Button(register_frame, text = 'REGISTER', bg='yellow', font=("Courier, 24"), command = new_image)
btn_register.place(relx=0.1, rely=0.3, relwidth=0.3)

btn_remove = tk.Button(register_frame, text = 'REMOVE', bg='#dc143c', font=("Courier, 24"), command = remove)
btn_remove.place(relx=0.45, rely=0.3, relwidth=0.3)

label_error = tk.Label(register_frame, font=("Courier", 24, "bold"), fg='red', bg='white')
label_error.place(relx=0.1, rely=0.65)

lb_image = tk.Label(right_frame, bg='gray')
lb_image.place(relwidth=0.6, relheight=0.5, relx=0.2, rely=0.1)

label_login_name = tk.Label(right_frame, font=("Courier", 24, "bold"), fg='red', text = 'Name: ')
label_login_name.place(relx=0.2, rely=0.65)

label_login_prob = tk.Label(right_frame, font=("Courier", 24, "bold"), fg='red', text = 'Probability: ')
label_login_prob.place(relx=0.2, rely=0.72)

btn_login = tk.Button(right_frame, text = 'LOGIN', font=("Courier, 24"), bg='#00d3d6', command = predict_image)
btn_login.place(relx=0.2, rely=0.8, relwidth=0.6)


open_camera()
root.mainloop()
cv2.destroyAllWindows()
