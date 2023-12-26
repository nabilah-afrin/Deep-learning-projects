
glob_path_control = glob.glob(r"/content/drive/MyDrive/Parkinsons/sub-control*.nii.gz")
glob_path_patient = glob.glob(r"/content/drive/MyDrive/Parkinsons/sub-patient*.nii.gz")

control_data = np.array([process_scan(path) for path in glob_path_control])
patient_data = np.array([process_scan(path) for path in glob_path_patient])


control_labels = np.array([1 for _ in range(len(control_data))])
patient_labels = np.array([0 for _ in range(len(patient_data))])


X = np.concatenate((control_data, patient_data), axis=0)
print("Dataset Shape : ", X.shape)
Y = np.concatenate((control_labels, patient_labels), axis=0)
print("Label Shape : ", Y.shape)


# Assuming Y has two classes
Y = to_categorical(Y, num_classes=2)

# Split the data
x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=0.20, random_state=42)

print("X_train : ", x_train.shape)
print("X_test : ", x_test.shape)
print("Y_train : ", y_train.shape)
print("Y_test : ", y_test.shape)
