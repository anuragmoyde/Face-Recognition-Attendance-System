import streamlit as st
import pandas as pd
import time
import os
from datetime import datetime
from streamlit_autorefresh import st_autorefresh

# Set up auto-refresh every 2 seconds
count = st_autorefresh(interval=2000, limit=100, key="attendance_refresh")

# Get the current date for the filename
ts = time.time()
date = datetime.fromtimestamp(ts).strftime("%d-%m-%Y")
file_path = f"Attendance/Attendance_{date}.csv"

# Check if the attendance file exists
if os.path.isfile(file_path):
    # Load and display the attendance DataFrame
    df = pd.read_csv(file_path)
    st.dataframe(df.style.highlight_max(axis=0))
else:
    # Display a message if no attendance file is found for today
    st.write("Attendance file not found for today.")
