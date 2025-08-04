
import os
os.environ['CUDA_VISIBLE_DEVICES']='0'

import warnings
warnings.simplefilter("ignore", UserWarning)

from llava.video_streamer import VideoStreamer
from llava.model.builder import load_pretrained_model

video_paths = ["sample.mp4"]  # Replace with your video file paths

model_base = None
model_path = "model/path/to/your/model" # Replace with your model path
model_name = "llava_qwen"

device = "cuda"
device_map = "auto"

# Set the maximum number of frames to process from each video (if not online setting)
max_frames_num = 64

tokenizer, model, image_processor, max_length = load_pretrained_model(model_path, model_base, model_name, 
                                                                        torch_dtype="bfloat16", device_map=device_map, attn_implementation="sdpa")

model.eval()

question = "What is happening in the video?"

videoStreamer = VideoStreamer(video_paths, image_processor, torch_device=device, max_frames_num=max_frames_num, stream_inf=False)

outputs = videoStreamer.stream_generate_only_tok(model, tokenizer, question, use_carrier=True)
print(outputs[0])
outputs = videoStreamer.vscan_add_kv(model, tokenizer, question, use_carrier=True, add_kv=16)
print(outputs[0])