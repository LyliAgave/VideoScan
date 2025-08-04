
import os
os.environ['CUDA_VISIBLE_DEVICES']='0'

from llava.video_streamer import VideoStreamer
from llava.model.builder import load_pretrained_model

video_paths = ["sample.mp4"]  # Replace with your video file paths

model_base = None
model_path = "model/path/to/your/model" # Replace with your model path
model_name = "llava_qwen"

device = "cuda"
device_map = "auto"

max_frames_num = 64  # For online setting, this is equivalent to the memory size

tokenizer, model, image_processor, max_length = load_pretrained_model(model_path, model_base, model_name, 
                                                                        torch_dtype="bfloat16", device_map=device_map, attn_implementation="sdpa")
model.eval()

question = "What is happening in the video?"

# stream_inf=True means that the model will execute stream inference with 1 fps
videoStreamer = VideoStreamer(video_paths, image_processor, torch_device=device, max_frames_num=max_frames_num, stream_inf=False)

outputs = videoStreamer.stream_generate_inf(model, tokenizer, question, use_carrier=True)
