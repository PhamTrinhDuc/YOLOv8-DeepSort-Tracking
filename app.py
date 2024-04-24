from fastapi import FastAPI, UploadFile, File
from Inference import run
import streamlit as st
import os
import time 
import moviepy.editor as moviepy

# app = FastAPI()

# @app.post("/")
# async def upload_video(file_video: UploadFile = File(...)):
#     with open(file_video.filename, "wb") as buffer:
#         # shutil.copyfileobj(file_video.filename)
#         buffer.write(file_video.file.read())
#         # type_data = type(file_video)
#         run(file_video.filename)

    
#     return {"file name: ", file_video.filename}



st.title("YOLOv8 Tracking Deep Sort")
st.markdown("---")
video = st.file_uploader("Please upload your video in here !!", type=["mp4", "avi"])
if video is not None:
    # st.video(video)
    with open(video.name, "wb") as f:
        f.write(video.read())

        save_results_name = run(path_video=video.name)
    f.close()

    clip = moviepy.VideoFileClip("output_video.avi")
    clip.write_videofile("output_video.mp4")
    st.markdown("---")
    st.text("Below are the results.")
    st.video("output_video.mp4")
