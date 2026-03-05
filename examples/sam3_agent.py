import os
from sam3 import build_sam3_image_model
from sam3.model.sam3_image_processor import Sam3Processor
import cv2
from PIL import Image
from functools import partial
from sam3.agent.client_llm import send_generate_request as send_generate_request_orig
from sam3.agent.client_sam3 import call_sam_service as call_sam_service_orig
from sam3.agent.inference import run_single_image_inference
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


if __name__ == "__main__":
    
    bpe_path = "../sam3/assets/bpe_simple_vocab_16e6.txt.gz"
    model = build_sam3_image_model(bpe_path=bpe_path)
    processor = Sam3Processor(model, confidence_threshold=0.5)
    video_path = "/workspace/video-agents/benchmark/tasks/lane-change-rate/pipeline/vids/pullup_30s.mp4"
    cap = cv2.VideoCapture(video_path)
    cap.set(cv2.CAP_PROP_POS_FRAMES, 30)
    ret, frame = cap.read()
    assert ret, "Failed to read frame 30"
    img = Image.fromarray(frame)
    img.save("test_image.jpg")
    # prepare input args and run single image inference
    image = "test_image.jpg"
    user_prompt = """
    chunk of pull up bar the man is gripping
    """
    image = os.path.abspath(image)
    send_generate_request = partial(send_generate_request_orig, server_url=LLM_CONFIGS["gpt-5.2"]["base_url"], model=LLM_CONFIGS["gpt-5.2"]["model"], api_key=LLM_CONFIGS["gpt-5.2"]["api_key"])
    call_sam_service = partial(call_sam_service_orig, sam3_processor=processor)
    # todo: add prediction probability as information
    run_single_image_inference(
        image, user_prompt, LLM_CONFIGS["gpt-5.2"], send_generate_request, call_sam_service,
        debug=True, output_dir="pull_up_bar_output"
    )

    # display output
    # if output_image_path is not None:
    #     display(Image(filename=output_image_path))