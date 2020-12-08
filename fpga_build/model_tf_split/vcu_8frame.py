import subprocess
import argparse
import os
import sys
import re

BATCH_SIZE = 8


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Source image generation for 8-frame offline TSM")
    parser.add_argument("vid_path", type=str)
    parser.add_argument("preclipped", action='store_true')
    args = parser.parse_args()

    if not os.path.isfile(args.vid_path):
        print("Invalid video path")
        sys.exit(1)

    ffprobe_out = subprocess.check_output(f"ffprobe -v error -select_streams v:0 -show_entries stream=width,height,r_frame_rate,nb_frames -of default=nw=1 {args.vid_path}", shell=True)
    ffprobe_out = ffprobe_out.decode(sys.stdout.encoding)
    print(ffprobe_out)

    width = int(re.search(r"width=(\d+)",ffprobe_out).group(1))
    height = int(re.search(r"height=(\d+)",ffprobe_out).group(1))
    num_frames = int(re.search(r"nb_frames=(\d+)",ffprobe_out).group(1))
    framerate = re.search(r"r_frame_rate=(\d+)/(\d+)",ffprobe_out)
    framerate = int(framerate.group(1)) / int(framerate.group(2))

    clip_len = int(num_frames // framerate)

    rate_num = int(framerate.group(1))
    rate_den = int(framerate.group(2))
    if args.preclipped:
        rate_num = BATCH_SIZE
        rate_den = clip_len


    new_h = 256
    new_w = 256
    if height > width:
        new_h = int(256*height/width)
    else:
        new_w = int(256*width/height)

    r_top = (new_h - 224)//2
    r_left = (new_w - 224)//2

    r_bot = r_top
    r_right = r_left

    leftover_side = (new_w - (r_left*2)) - 224
    r_right += leftover_side

    leftover_bot = (new_h - (r_top*2)) - 224
    r_bot += leftover_bot

    print(f"new_dims = ({new_w}, {new_h})")
    print(f"removal = ({r_left}, {r_right}, {r_top}, {r_bot})")


    subprocess.run(f"gst-launch-1.0 filesrc location={args.vid_path} ! qtdemux name=demux demux.video_0 ! h264parse ! video/x-h264, alignment=au ! omxh264dec low-latency=1 ! videoconvert ! videorate ! videoscale ! video/x-raw,width={new_w},height={new_h},framerate={rate_num}/{rate_den} ! videoconvert ! videocrop top={r_top} bottom={r_bot} left={r_left} right={r_right} ! jpegenc ! multifilesink location=test_decode/frame_%01d.jpg", shell=True)
