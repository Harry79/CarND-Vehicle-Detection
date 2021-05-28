from moviepy.editor import VideoFileClip

import solve

solve.initUndist()
white_output = 'output_project_video.mp4'
clip1 = VideoFileClip("../project_video.mp4")
white_clip = clip1.fl_image(solve.process_image)
white_clip.write_videofile(white_output, audio=False)

print(solve.lanewidthArr)
