from tkinter import *
import tkinter as tk
from tkinter import filedialog, messagebox
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import AdaBoostClassifier, RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
import numpy as np
from imblearn.combine import SMOTEENN
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, GRU, Dense

global filename
global df, X_train, X_test, y_train, y_test
global ada_acc, rf_acc, mlp_acc, lstm_acc, gru_acc, smote_enn_acc

# ================= UPLOAD DATASET =================

def upload():
    global filename, df

    filename = filedialog.askopenfilename(initialdir="dataset")
    pathlabel.config(text=filename)

    df = pd.read_csv(filename)

    # Replace missing values
    df.replace('?', np.nan, inplace=True)

    # Fill missing values
    df.fillna(df.mode().iloc[0], inplace=True)

    text.delete('1.0', END)
    text.insert(END, 'Dataset loaded successfully\n')
    text.insert(END, "Dataset Size : " + str(len(df)) + "\n")


# ================= SPLIT DATASET =================

def splitdataset():

    global df, X_train, X_test, y_train, y_test

    label_encoder = LabelEncoder()

    for column in df.columns:
        if df[column].dtype == 'object':
            df[column] = label_encoder.fit_transform(df[column])

    X = np.array(df.drop(["Class"], axis=1))
    y = np.array(df["Class"])

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2
    )

    text.delete('1.0', END)

    text.insert(END, "Dataset split completed\n\n")

    text.insert(
        END,
        "Splitted Training Size : " + str(len(X_train)) + "\n"
    )

    text.insert(
        END,
        "Splitted Test Size : " + str(len(X_test)) + "\n\n"
    )

    text.insert(END, "Shape of X_train : " + str(X_train.shape) + "\n")
    text.insert(END, "Shape of X_test : " + str(X_test.shape) + "\n")
    text.insert(END, "Shape of y_train : " + str(y_train.shape) + "\n")
    text.insert(END, "Shape of y_test : " + str(y_test.shape) + "\n\n")


# ================= ADABOOST =================

def adaboost():

    global ada_acc

    ada = AdaBoostClassifier(
        n_estimators=100,
        random_state=0
    )

    ada.fit(X_train, y_train)

    y_pred = ada.predict(X_test)

    ada_acc = accuracy_score(y_test, y_pred)

    text.insert(
        END,
        f'Accuracy for AdaBoost is {ada_acc * 100}%\n'
    )


# ================= RANDOM FOREST =================

def random_forest():

    global rf_acc

    rf = RandomForestClassifier(
        n_estimators=100,
        random_state=0
    )

    rf.fit(X_train, y_train)

    y_pred = rf.predict(X_test)

    rf_acc = accuracy_score(y_test, y_pred)

    text.insert(
        END,
        f'Accuracy for Random Forest is {rf_acc * 100}%\n'
    )


# ================= MLP =================

def mlp():

    global mlp_acc, mlp_model

    mlp_model = MLPClassifier()

    mlp_model.fit(X_train, y_train)

    y_pred = mlp_model.predict(X_test)

    mlp_acc = accuracy_score(y_test, y_pred)

    text.insert(
        END,
        f'Accuracy for MLP is {mlp_acc * 100}%\n'
    )


# ================= LSTM =================

def lstm():

    global lstm_acc

    X_train_lstm = X_train.reshape(
        X_train.shape[0],
        X_train.shape[1],
        1
    )

    X_test_lstm = X_test.reshape(
        X_test.shape[0],
        X_test.shape[1],
        1
    )

    model = Sequential()

    model.add(
        LSTM(
            units=50,
            input_shape=(X_train.shape[1], 1)
        )
    )

    model.add(Dense(1, activation='sigmoid'))

    model.compile(
        optimizer='adam',
        loss='binary_crossentropy',
        metrics=['accuracy']
    )

    model.fit(
        X_train_lstm,
        y_train,
        epochs=10,
        batch_size=32,
        verbose=0
    )

    _, lstm_acc = model.evaluate(
        X_test_lstm,
        y_test,
        verbose=0
    )

    text.insert(
        END,
        f'Accuracy for LSTM is {lstm_acc * 100}%\n'
    )


# ================= GRU =================

def gru():

    global gru_acc

    X_train_gru = X_train.reshape(
        X_train.shape[0],
        X_train.shape[1],
        1
    )

    X_test_gru = X_test.reshape(
        X_test.shape[0],
        X_test.shape[1],
        1
    )

    model = Sequential()

    model.add(
        GRU(
            units=50,
            input_shape=(X_train.shape[1], 1)
        )
    )

    model.add(Dense(1, activation='sigmoid'))

    model.compile(
        optimizer='adam',
        loss='binary_crossentropy',
        metrics=['accuracy']
    )

    model.fit(
        X_train_gru,
        y_train,
        epochs=10,
        batch_size=32,
        verbose=0
    )

    _, gru_acc = model.evaluate(
        X_test_gru,
        y_test,
        verbose=0
    )

    text.insert(
        END,
        f'Accuracy for GRU is {gru_acc * 100}%\n'
    )


# ================= SMOTE ENN =================

def smote_enn():

    global smote_enn_acc, mlp_model

    smote_enn_model = SMOTEENN()

    X_resampled, y_resampled = smote_enn_model.fit_resample(
        X_train,
        y_train
    )

    mlp_model = MLPClassifier()

    mlp_model.fit(X_resampled, y_resampled)

    y_pred = mlp_model.predict(X_test)

    smote_enn_acc = accuracy_score(y_test, y_pred)

    text.insert(
        END,
        f'Accuracy for SMOTE-ENN is {smote_enn_acc * 100}%\n'
    )


# ================= PLOT GRAPH =================

def plot_bar_graph():

    algorithms = [
        'AdaBoost',
        'Random Forest',
        'MLP',
        'LSTM',
        'GRU',
        'SMOTE-ENN'
    ]

    accuracies = [
        ada_acc * 100,
        rf_acc * 100,
        mlp_acc * 100,
        lstm_acc * 100,
        gru_acc * 100,
        smote_enn_acc * 100
    ]

    colors = [
        'blue',
        'orange',
        'green',
        'red',
        'purple',
        'cyan'
    ]

    plt.bar(algorithms, accuracies, color=colors)

    plt.xlabel('Algorithms')
    plt.ylabel('Accuracy (%)')
    plt.title('Accuracy of Machine Learning Algorithms')

    plt.xticks(rotation=45)

    plt.tight_layout()

    plt.show()


# ================= PREDICTION =================

def predict():

    filename = filedialog.askopenfilename(
        title="Select CSV File",
        filetypes=[("CSV Files", "*.csv")]
    )

    if filename:

        input_data = pd.read_csv(filename)

        input_data.fillna(
            input_data.mode().iloc[0],
            inplace=True
        )

        label_encoder = LabelEncoder()

        for column in input_data.columns:
            if input_data[column].dtype == 'object':
                input_data[column] = label_encoder.fit_transform(
                    input_data[column]
                )

        y_pred = mlp_model.predict(input_data)

        if y_pred[0] == 1:
            messagebox.showinfo(
                "Prediction Result",
                "Fraudulent Transaction Detected"
            )

        else:
            messagebox.showinfo(
                "Prediction Result",
                "Non-Fraudulent Transaction Detected"
            )


# ================= MAIN WINDOW =================

main = tk.Tk()

main.title(
    "A DEEP LEARNING ENSEMBLE WITH DATA RESAMPLING FOR CREDIT CARDS FRAUD DETECTION"
)

main.geometry("1600x1000")

bg_color = "#32d1a7"

main.config(bg=bg_color)

# ================= TITLE =================

font = ('times', 16, 'bold')

title = tk.Label(
    main,
    text='A DEEP LEARNING ENSEMBLE WITH DATA RESAMPLING FOR CREDIT CARDS FRAUD DETECTION',
    font=("times")
)

title.config(bg='Dark Blue', fg='white')

title.config(font=font)

title.config(height=3, width=145)

title.place(x=0, y=5)

# ================= TEXT AREA =================

font1 = ('times', 12, 'bold')

text = tk.Text(main, height=20, width=180)

scroll = tk.Scrollbar(text)

text.configure(yscrollcommand=scroll.set)

text.place(x=50, y=120)

text.config(font=font1)

# ================= BUTTON STYLE =================

font1 = ('times', 13, 'bold')

button_bg_color = "lightgrey"
button_fg_color = "black"

button_hover_bg_color = "grey"
button_hover_fg_color = "white"

button_config = {
    "bg": button_bg_color,
    "fg": button_fg_color,
    "activebackground": button_hover_bg_color,
    "activeforeground": button_hover_fg_color,
    "width": 18,
    "font": font1
}

# ================= BUTTONS =================

uploadButton = tk.Button(
    main,
    text="Upload Dataset",
    command=upload,
    **button_config
)

pathlabel = tk.Label(main)

splitButton = tk.Button(
    main,
    text="Split Dataset",
    command=splitdataset,
    **button_config
)

adaboostButton = tk.Button(
    main,
    text="AdaBoost",
    command=adaboost,
    **button_config
)

rfButton = tk.Button(
    main,
    text="Random Forest",
    command=random_forest,
    **button_config
)

mlpButton = tk.Button(
    main,
    text="MLP",
    command=mlp,
    **button_config
)

lstmButton = tk.Button(
    main,
    text="LSTM",
    command=lstm,
    **button_config
)

gruButton = tk.Button(
    main,
    text="GRU",
    command=gru,
    **button_config
)

smote_ennButton = tk.Button(
    main,
    text="SMOTE-ENN",
    command=smote_enn,
    **button_config
)

plotButton = tk.Button(
    main,
    text="Plot Results",
    command=plot_bar_graph,
    **button_config
)

predict_button = tk.Button(
    main,
    text="Prediction",
    command=predict,
    **button_config
)

# ================= BUTTON POSITIONS =================

# Row 1
uploadButton.place(x=250, y=600)
splitButton.place(x=550, y=600)

# Row 2
adaboostButton.place(x=250, y=670)
rfButton.place(x=550, y=670)

# Row 3
mlpButton.place(x=250, y=740)
lstmButton.place(x=550, y=740)

# Row 4
gruButton.place(x=250, y=810)
smote_ennButton.place(x=550, y=810)

# Row 5
plotButton.place(x=250, y=880)
predict_button.place(x=550, y=880)

# ================= PATH LABEL =================

pathlabel.config(
    bg='DarkOrange1',
    fg='white',
    font=font1
)

pathlabel.place(x=900, y=600)

# ================= RUN APPLICATION =================

main.mainloop()