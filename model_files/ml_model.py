import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.base import BaseEstimator, TransformerMixin
import cv2
import matplotlib.pyplot as plt

def preprocess_origin_cols(df):
    df["Origin"] = df["Origin"].map({1: "India", 2: "USA", 3: "Germany"})
    return df

class CustomAttrAdder(BaseEstimator, TransformerMixin):
    def _init_(self, acc_on_power=True):
        self.acc_on_power = acc_on_power
    def fit(self, X, y=None):
        return self
    def transform(self, X):
        acc_on_cyl = X[:, 3] / X[:, 1]
        if self.acc_on_power:
            acc_on_power = X[:, 3] / X[:, 5]
            return np.c_[X, acc_on_power, acc_on_cyl]
        return np.c_[X, acc_on_cyl]

def num_pipeline_transformer(data):
    num_attrs = data.select_dtypes(include=['float64', 'int64'])
    num_pipeline = Pipeline([
        ('imputer', SimpleImputer(strategy="median")),
        ('attrs_adder', CustomAttrAdder()),
        ('std_scaler', StandardScaler()),
    ])
    return num_attrs, num_pipeline

def pipeline_transformer(data):
    cat_attrs = ["Origin"]
    num_attrs, num_pipeline = num_pipeline_transformer(data)
    full_pipeline = ColumnTransformer([
        ("num", num_pipeline, list(num_attrs)),
        ("cat", OneHotEncoder(), cat_attrs),
    ])
    return full_pipeline.fit_transform(data)

def predict_mpg(config, model):
    if type(config) == dict:
        df = pd.DataFrame(config)
    else:
        df = config

    preproc_df = preprocess_origin_cols(df)
    pipeline = pipeline_transformer(preproc_df)
    prepared_df = pipeline.transform(preproc_df)
    y_pred = model.predict(prepared_df)
    return y_pred

def overlay_plot_on_image(image_path):
    # Load image
    image = Image.open(image_path)
    
    # Convert to numpy array for plotting
    image_np = np.array(image)
    
    # Create a plot
    fig, ax = plt.subplots()
    ax.imshow(image_np)
    
    # Example: Plotting a red rectangle
    rect = plt.Rectangle((50, 50), 100, 100, edgecolor='r', facecolor='none', linewidth=2)
    ax.add_patch(rect)
    
    # Remove axes for better visualization
    ax.axis('off')
    
    # Save the plot to a BytesIO object
    buf = io.BytesIO()
    plt.savefig(buf, format='png', bbox_inches='tight', pad_inches=0)
    buf.seek(0)
    
    # Convert the plot to an image
    plotted_image = Image.open(buf)
    
    return plotted_image