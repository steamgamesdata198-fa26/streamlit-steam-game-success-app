"""

Steam Game Suggestor - Streamlit app

Pick a game (or describe one), ad the app finds similar games using a name, tags, description, price

"""
import ast
import re
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st 
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel

