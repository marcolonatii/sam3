import os
from sam3 import build_sam3_image_model
from sam3.model.sam3_image_processor import Sam3Processor
import cv2
from PIL import Image
from functools import partial
from sam3.agent.client_llm import send_generate_request as send_generate_request_orig
from sam3.agent.client_sam3 import call_sam_service as call_sam_service_orig
from sam3.agent.inference import run_single_image_inference
from sam3.agent.agent_tools import get_frames
import threading
LLM_CONFIGS = {
        # vLLM-served models
        "qwen3_vl_8b_thinking": {
            "name": "qwen3_vl_8b_thinking",
            "provider": "vllm",
            "model": "Qwen/Qwen3-VL-8B-Thinking",
            "base_url": "http://127.0.0.1:8002/v1",
        },
        "gpt-5.2": {
            "name": "gpt-5.2",
            "provider": "openai",
            "model": "gpt-5.2",
            "base_url": "https://api.openai.com/v1",
            "api_key": os.getenv("OPENAI_API_KEY"),
        },
        # models served via external APIs
        # add your own
    }

def get_detection_on_frame(prompt, frame, frame_idx, processor, output_dir):
    img = Image.fromarray(frame)
    os.makedirs(output_dir, exist_ok=True)
    # prepare input args and run single image inference
    image = os.path.join(output_dir, f"frame_{frame_idx}.jpg")
    img.save(image)
    image = os.path.abspath(image)
    send_generate_request = partial(send_generate_request_orig, server_url=LLM_CONFIGS["gpt-5.2"]["base_url"], model=LLM_CONFIGS["gpt-5.2"]["model"], api_key=LLM_CONFIGS["gpt-5.2"]["api_key"])
    call_sam_service = partial(call_sam_service_orig, sam3_processor=processor)
    # todo: add prediction probability as information
    run_single_image_inference(
        image, prompt, LLM_CONFIGS["gpt-5.2"], send_generate_request, call_sam_service,
        debug=True, output_dir=output_dir
    )



if __name__ == "__main__":
    # prompt = """
    # chunk of pull up bar the man is gripping
    # """
    prompt = "identify separate lanes on the road, especially the lanes nearest to the camera"

    
    bpe_path = "../sam3/assets/bpe_simple_vocab_16e6.txt.gz"
    model = build_sam3_image_model(bpe_path=bpe_path)
    processor = Sam3Processor(model, confidence_threshold=0.5)
    video_path = "/workspace/video-agents/benchmark/tasks/lane-change-rate/pipeline/vids/lane_change.mp4"
    video_frames = get_frames(video_path)
    frames = []
    for i in range(0, len(video_frames), len(video_frames) // 4):
        frames.append((i, video_frames[i]))
    for frame_idx, frame in frames:
        threading.Thread(target=get_detection_on_frame, args=(prompt, frame, frame_idx, processor, f"output/{video_path.split('/')[-1].split('.')[0]}_output{frame_idx}")).start()