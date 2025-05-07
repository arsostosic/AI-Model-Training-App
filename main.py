# GUI IMPORTS
import os

import sys

from PyQt5.QtWidgets import (QApplication, QWidget, QLabel,
                             QPushButton, QVBoxLayout,
                             QHBoxLayout, QLineEdit, QGridLayout, QComboBox, QFileDialog, QDialog, QListWidget,
                             QDialogButtonBox, QMessageBox)
from PyQt5.QtGui import QIcon

from PyQt5.QtCore import Qt

# ML IMPORTS

import numpy as np
import pandas as pd
import csv

from numpy.matlib import empty
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.utils import Bunch


class MLapp(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("LR Model App")
        self.setWindowIcon(QIcon("linear-regression.png"))
        self.pr_page = MLappTrainingPage(self) # this way we avoid recursion when going back from training page to main page
        #self.setStyleSheet("background-color: hsl(282, 82%, 30%);")
        self.browse_dataset_btn = QPushButton("Browse Dataset",self)
        self.submit_dataset_btn = QPushButton("Submit",self)
        self.cancel_dataset_btn = QPushButton("Cancel",self)
        self.show_dataset_lbl = QLabel("No dataset selected",self) # temp placeholder text

        self.name_model_textbox = QLineEdit(self) # temp placeholder text
        self.name_model_textbox.setPlaceholderText(" name your model")
        self.finish_naming_btn = QPushButton("Finish",self)
        self.train_model_btn = QPushButton("Train model",self)
        self.stop_training_btn = QPushButton("Stop training",self)
        self.show_training_results_lbl = QLabel(self) # temp placeholder text
        self.name_model_lbl = QLabel("Name your model:", self)
        self.training_results_lbl = QLabel("Training results:",self)
        self.to_prediction_page_btn = QPushButton("Prediction page ➡️",self)
        self.selected_data_path = None # Used for keeping the value for submitting dataset
        self.target_column = None # New variable for selecting the target column
        self.scaler = None # Prevents multiple creation of scaler and model object throughout the code
        self.model = None
        self.cleaned_dataset = None # for preprocessed dataset(removing NaN and nulls from selected dataset)
        self.init_ui()

    def init_ui(self):

        grid = QGridLayout()
        grid.addWidget(self.show_dataset_lbl,0,0)
        grid.addWidget(self.cancel_dataset_btn, 0, 1)
        grid.addWidget(self.browse_dataset_btn, 1, 0)
        grid.addWidget(self.submit_dataset_btn, 1, 1)
        grid.addWidget(self.to_prediction_page_btn,2,1)
        self.setLayout(grid)

        vbox = QVBoxLayout()
        vbox.addWidget(self.name_model_lbl)
        vbox.addWidget(self.name_model_textbox)
        vbox.addWidget(self.finish_naming_btn)
        vbox.addWidget(self.train_model_btn)
        vbox.addWidget(self.stop_training_btn)



        grid.addLayout(vbox,2,0,alignment=Qt.AlignCenter)

        hbox = QHBoxLayout()

        hbox.addWidget(self.training_results_lbl)
        hbox.addWidget(self.show_training_results_lbl)

        grid.addLayout(hbox,3,1,alignment=Qt.AlignHCenter)

        self.to_prediction_page_btn.setObjectName("prediction_page_btn")
        self.show_training_results_lbl.setObjectName("show_training_res_lbl")
        self.training_results_lbl.setObjectName("training_res_lbl")

        self.setStyleSheet("""
        
        /*QWidget{
        background-color: hsl(275, 56%, 42%);
        }*/
        
        QLineEdit{
        background: white;
        
        }
        
        QPushButton{
        background-color: white;
        }
        QPushButton:hover{
        background-color: hsl(196, 100%, 70%);
        color: white
        }
        
        QPushButton#prediction_page_btn{
        min-width: 175px;
        min-height: 60px;
        max-width: 175px;
        max-height: 60px;
        border:2px solid black;
        border-radius: 15px;
        margin-left: 75px;
        padding: 5px;
        
        }
        
        QLabel#show_training_res_lbl{
        color: black;
        font-weight: bold;
        }
        
        QLabel#training_res_lbl{
        color: black;
        font-weight: bold;
        }
        
        """)

        # ADDING SIGNAL/SLOT
        self.to_prediction_page_btn.clicked.connect(self.goto_prediction_page)
        self.browse_dataset_btn.clicked.connect(self.browse_dataset)
        self.cancel_dataset_btn.clicked.connect(self.cancel_dataset)
        self.submit_dataset_btn.clicked.connect(self.submit_dataset)
        self.train_model_btn.clicked.connect(self.train_model)

    def goto_prediction_page(self):
        self.hide()
        self.pr_page.show()

    def browse_dataset(self):
        options = QFileDialog.Options()
        file_path, selected_filter = QFileDialog.getOpenFileNames(
            self,
            "Select File",
            "",
            "CSV Files(*.csv);;Excel Files(*.xlsx *.xls)",
            options=options
        )
        if file_path:
            self.selected_data_path = file_path[0]

        # If multiple files are selected, file_path is a list.
        # Use file_path[0] to get the first file.
        # Loop through file_path to process all files.

            file_name = os.path.basename(str(file_path).strip("[]'")) # os basename extracts the name of the file without whole path location, strip function removes unwanted leftover signs/characters in string
            self.show_dataset_lbl.setText(f"Selected file: {file_name}, \nFilter: {selected_filter}")

    def cancel_dataset(self):
        self.show_dataset_lbl.clear()
        self.show_dataset_lbl.setText("No dataset selected")
        self.selected_data_path = None # to reset data path after browsing

    def submit_dataset(self):
        if self.selected_data_path:
            print(f"Loading dataset from: {self.selected_data_path}")

            try:
                df = pd.read_csv(self.selected_data_path, dtype='object', low_memory=False, encoding='utf-8')
            except Exception as e:
                self.show_dataset_lbl.setText(f"Error loading dataset: {e}")
                return

            print("Dataset first 5 rows before conversion:\n", df.head())
            print("Dataset column types before conversion:\n", df.dtypes)

            # Checking if all values are strings
            if df.map(lambda x: isinstance(x, str)).all().all():
                print("Warning: All values are strings! Attempting conversion...")

            # Trying to convert to numerical
            df_numeric = df.apply(
                lambda col: pd.to_numeric(col.str.replace(',', ''), errors='coerce') if col.dtype == "object" else col)

            print("Dataset first 5 rows after attempted conversion:\n", df_numeric.head())
            print("Dataset column types after attempted conversion:\n", df_numeric.dtypes)

            # Removing non numerical columns
            numeric_columns = df_numeric.select_dtypes(include=[np.number]).columns
            df_numeric = df_numeric[numeric_columns]

            if df_numeric.empty:
                self.show_dataset_lbl.setText("Warning: Dataset is mostly non-numeric after cleaning.")
                print("Dataset is empty after removing non-numeric columns.")
                return

            print(f"Final dataset shape: {df_numeric.shape}")

            # User chooses target column
            self.target_column = ColumnSelectorDialog.get_selected_column(df_numeric.columns.tolist(), self)
            if self.target_column:
                self.cleaned_dataset = df_numeric  # Čuvamo očišćen dataset
                self.show_dataset_lbl.setText(f"Dataset loaded successfully. Target column: {self.target_column}")
                print("Dataset successfully prepared for training.")
            else:
                self.show_dataset_lbl.setText("Error: No target column selected.")

    # Logic for turning pd dataframe into sklearn standardized Bunch type dataset

    @staticmethod
    def dataframe_to_sklearn_dataset(df, target_column):
        """
        Convert a pandas DataFrame to a sklearn Bunch object.

        Parameters:
        df: pandas DataFrame
        target_column: str, name of the target column in the DataFrame

        Returns:
        Bunch object with sklearn dataset structure
        """
        try:
            # Checking if there is a valid target column
            if target_column not in df.columns:
                raise ValueError(f"Target column '{target_column}' not found in dataset.")

            # Conversion into numerical format (if strings detected)
            for col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce')

            # Removing rows with NaN values after the conversion
            df_cleaned = df.dropna()

            # creating sklearn Bunch object
            dataset = Bunch(
                data=df_cleaned.drop(target_column, axis=1).values,  # X vrednosti
                target=df_cleaned[target_column].values,  # y vrednosti
                feature_names=df_cleaned.drop(target_column, axis=1).columns.tolist(),
                target_names=[target_column],
                DESCR="Converted from DataFrame"
            )
            return dataset

        except Exception as e:
            print(f"Error converting dataset: {e}")
            return None

    def train_model(self):
        try:
            model_name = self.name_model_textbox.text()
            if model_name == "":
                self.name_model_textbox.clear()
                self.name_model_textbox.setPlaceholderText("Can't be empty!")
            else:
                pass

            # Check if dataset has been selected
            if self.selected_data_path is None:
                QMessageBox.warning(
                    self,
                    "Warning",
                    "Please select a dataset first!"
                )
                return

            if self.cleaned_dataset is None:
                self.submit_dataset()

                if self.cleaned_dataset is None:
                    QMessageBox.warning(
                        self,
                        "Warning",
                        "Please load and clean data"
                    )
                    return

            # Verify target column selection
            if self.target_column is None:
                QMessageBox.warning(
                    self,
                    "Warning",
                    "Please select a target column!"
                )
                return

            # Using our created dataframe

            dataset = self.dataframe_to_sklearn_dataset(self.cleaned_dataset, self.target_column)

            if dataset is None:
                QMessageBox.warning(self, "Error", "Failed to convert dataset into sklearn format.")
                return

            x = self.cleaned_dataset.drop(columns=[self.target_column])
            y = self.cleaned_dataset[self.target_column]

            x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2,
                                                                random_state=42)  # this means that 0.2 or 20%
            # of our whole dataset will be used for testing (X,Y values included)
            # and random 42 is to enable us to have same training and testing labels every time we start code and not random again

            # Validate that there are no NaNs or infinite values in the dataset
            if x.isnull().values.any() or y.isnull().values.any():
                QMessageBox.warning(self, "Error", "Dataset contains null values. Please check your data.")
                return

            if np.isinf(x.values).any() or np.isinf(y.values).any():
                QMessageBox.warning(self, "Error", "Dataset contains infinite values. Please check your data.")
                return

            # Ensure there are still valid numerical features
            if x.shape[1] == 0:
                QMessageBox.warning(self, "Error", "No valid numerical features remain after cleaning.")
                return


            # Checking for NaN or inf values
            if np.isnan(x_train.values).any() or np.isinf(x_train.values).any():
                QMessageBox.warning(self, "Error", "Dataset contains NaN or infinite values. Please clean your data.")
                return



            # If all data is removed after cleaning
            if x_train.empty or x_test.empty:
                QMessageBox.warning(self, "Error", "No valid features remain after cleaning. Check dataset.")
                return



            # Scaling the values (with mean of 0 and standard deviation of 1)
            scaler = StandardScaler()
            x_train_scaled = scaler.fit_transform(x_train)  # our x_train set will be scaled
            x_test_scaled = scaler.fit_transform(x_test)

            # Training our model using mathematical algorithm of Linear Regression

            m_alg = LinearRegression()
            m_alg.fit(x_train_scaled, y_train)

            y_predict = m_alg.predict(x_test_scaled)

            # Evaluating our model
            mse = mean_squared_error(y_test, y_predict)
            r2 = r2_score(y_test, y_predict)

            # Printing model successfulness
            self.training_results_lbl.setText(f"MSE error: {mse:.2f}\nRMSE error: {r2:.2f}")
        except Exception as e:
            self.training_results_lbl.setText(f"Error occurred: {e}")

            # dataset = MLapp.dataframe_to_sklearn_dataset(local_data,self.target_column)
            # df = pd.DataFrame(dataset.data, columns=dataset.feature_names)
            # df[self.target_column] = dataset.target   this will be our column with a value but these are values prepared in advance for houses, so we'll need smth else for this, maybe by including AI in this process

            # Checking for null values

            # print(df.isnull().sum())

            # Splitting dataset columns X-independent, Y-dependent (PRICE - column)

            # x = df.drop(self.target_column)  # means that all the other columns will be considered as independent or X values
            #y = df[self.target_column]

            # Setting up our training and testing labels/sets






class MLappTrainingPage(QWidget):
    def __init__(self, main_page): # main_page is self or instance of MLapp that we defined previously
        super().__init__()
        self.setWindowIcon(QIcon("linear-regression.png"))
        self.setWindowTitle("Model Prediction")
        self.setGeometry(800,500,600,500)
        # self.chose_model_lbl = QLabel("Chose model: ") Include later
        self.model_choice_menu_cbox = QComboBox()
        self.go_back_btn = QPushButton("⬅️ Go Back",self)
        self.main_page = main_page
        self.parameter1_location_lbl = QLabel("Location: ",self) # Temp placeholder for text, I will need function to automatically generate number of labels that matches training labels rom dataset, as well as Y value
        self.parameter2_area_lbl = QLabel("Area in m²: ", self)
        self.Y_results_lbl = QLabel("House price: 75,000 $ ", self)
        self.predict_btn = QPushButton("Predict",self)
        self.param1_input_txtbox = QLineEdit()
        self.param2_input_txtbox = QLineEdit() # Also this will need to be auto-generated automatically based on number of labels/parameters

        self.init_mt_ui()

    def init_mt_ui(self):
        self.go_back_btn.setGeometry(700,300,30,30)

        grid = QGridLayout()
        grid.addWidget(self.parameter1_location_lbl,0,0)
        grid.addWidget(self.parameter2_area_lbl,1,0)
        grid.addWidget(self.param1_input_txtbox,0,1)
        grid.addWidget(self.param2_input_txtbox,1,1)
        grid.addWidget(self.predict_btn)
        grid.addWidget(self.Y_results_lbl)
        grid.addWidget(self.go_back_btn)

        self.setLayout(grid)

        # SIGNAL/SLOT
        self.go_back_btn.clicked.connect(self.go_back)

    def go_back(self):
        self.hide()
        self.main_page.show()


class ColumnSelectorDialog(QDialog):
    def __init__(self, columns, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Odaberi ciljnu kolonu")
        self.setGeometry(300, 300, 400, 300)

        layout = QVBoxLayout()
        self.list_widget = QListWidget()
        self.list_widget.addItems(columns)
        self.list_widget.itemSelectionChanged.connect(self.on_selection_changed)

        layout.addWidget(self.list_widget)

        button_box = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        button_box.accepted.connect(self.accept)
        button_box.rejected.connect(self.reject)

        layout.addWidget(button_box)
        self.setLayout(layout)

        self.selected_column = None

    def on_selection_changed(self):
        selected_items = self.list_widget.selectedItems()
        if selected_items:
            self.selected_column = selected_items[0].text()

    @staticmethod
    def get_selected_column(columns, parent=None):
        dialog = ColumnSelectorDialog(columns, parent)
        if dialog.exec_() == QDialog.Accepted:
            return dialog.selected_column
        return None


def main():
    app = QApplication(sys.argv)
    app.setStyle("Fusion") # we added this so that when setting the bg color of the whole app, widget, other element's styles stay untouched
    window = MLapp()
    window.show()
    sys.exit(app.exec_())

if __name__ == '__main__':
    main()