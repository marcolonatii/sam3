from sam3.agent.agent_video_tracker import *
import os
from langfuse.langchain import CallbackHandler
from uuid import uuid4
from dotenv import load_dotenv
from utils import format_agent_trace
from langchain.messages import HumanMessage, SystemMessage, ToolMessage
from langfuse import observe
from langchain_openai import ChatOpenAI

# ------------------------------------
def execute(tool_calls, messages, cfg, tools):
    if not tool_calls:
        return  # or handle no calls

    image_human_msgs = []
    for tool_call in tool_calls:  # ai_msg.tool_calls is list of dicts
        name = tool_call["name"]
        args = tool_call["args"]
        call_id = tool_call.get("id")  # required for ToolMessage

        # Find the matching tool (cleaner than if/elif chain)
        tool = next((t for t in tools if t.name == name), None)
        is_image_output = False

        if tool is None:
            # Handle unknown tool gracefully
            content = f"Error: Tool '{name}' not found."
        else:
            try:
                tool_result = tool.invoke(
                    args,
                    config={
                        **cfg,
                        "run_name": f"tool_{name}",
                    }
                )
                # Convert result to string (most models expect str content)
                content = str(tool_result) if not isinstance(tool_result, str) else tool_result
                
                print("tool_result",tool_result)
            except Exception as e:
                content = f"Tool execution failed: {str(e)}"

        # Now create proper ToolMessage
        tool_msg = ToolMessage(
              content=sam3_tracker.image_decorator.get_text_from_string(content),
              tool_call_id=call_id,      # ← critical: matches the assistant's tool call
              name=name,                 # optional but helpful
        )
        messages.append(tool_msg)
        if sam3_tracker.image_decorator.image_count(content) > 0:
          image_human_msgs.append(HumanMessage(
            content=[
              {"type": "text", "text": "the image for the tool call result"},
              {"type": "image_url", "image_url": {"url": sam3_tracker.image_decorator.get_image_from_string(content)}}
            ]
          ))
        print("tool_msg: ",tool_msg.content)

    # All ToolMessages for this turn are now appended; image HumanMessages go after
    print("image_human_msgs: ",image_human_msgs)
    messages.extend(image_human_msgs)

@observe(name="sam3_agent_run")
def run_agent(model_with_tools, human_msg, cfg, tools):
  messages = [system_msg, human_msg]
  print("message: ", messages)

  finish = False
  step = 0
  while not finish:
    step += 1
    if step > 20:
      break
    ai_msg = model_with_tools.invoke(
      messages,
      config={
        **cfg,
        "run_name": f"step_{step}",
        "metadata": {"agent": "sam3-react", "video": "nba_clip.mp4"}
      }
    )
    print("AI message:", ai_msg)
    finish = "<answer>" in ai_msg.content
    #filter the image content in the messages
    print("before: ",messages[-1].content)
    for i, content_block in enumerate(messages[-1].content):
      if "image_url" in content_block:
        messages[-1].content[i] = {"type": "text", "text": "result of get_frame"}
    print("after: ",messages[-1].content)
    messages.append(ai_msg)

    # Check the tool calls in the response
    print(ai_msg.tool_calls)

    # Step 2: Execute tools and collect results
    execute(ai_msg.tool_calls, messages, cfg, tools)
  final_response = model_with_tools.invoke(
      messages,
      config={
        **cfg,
        "run_name": f"final_response",
      }
    )
  return final_response, messages

if __name__ == "__main__":
    ENV=load_dotenv("/workspace/sam3/examples/.env")
    video_path = os.path.join(os.getenv("VID_DIR"), "chinup_final.mp4")
    sam3_tracker = Sam3TrackingTool(
        video_path=video_path,
        bpe_path=os.getenv("BPE_PATH")
    )

    trace_id = str(uuid4())
    langfuse_handler = CallbackHandler(
        trace_context={"trace_id": trace_id},
        update_trace=True,
    )  # reads LANGFUSE_* env vars
    cfg = {
        "callbacks": [langfuse_handler],
        "metadata": {
            "langfuse_session_id": "nba_clip_run_001",  # grouping label
            "langfuse_user_id": "thomas",
        },
    }
    #system prompt and query
    json_schema = {
        "total_pullup_count": "<number>"
    }
    system_msg = SystemMessage(
        "You are doing sport analysis on videos. Proceed with the tools.\n"
        "1. List objects of interest needed to answer the question.\n"
        "2. Use identify_object_by_prompt to identify the objects on frame 0. examine the images and determine if the objects are correctly identified and is what you want to track. if not, you must call identify_object_by_prompt again with another prompt. \n"
        "3. Call track_objects to track the objects through the video. After tracking, you will get i) the bounding boxes of the objects, ii) the masks of the objects, iii) the center coordinates of the objects.\n"
        "Analysis will be based on the relative position of the objects. Before you track the objects, think through how you can use the bounding boxes, masks and center coordinates to analyze the position of the objects and answer the question.\n"
        "Also, call get_tracked_objects_info to prevent the objects from being tracked multiple times.\n"
        "4. Verify the objects you want to track are successfully tracked by calling get_tracked_objects_info.\n"
        "5. If tracking fails, call reset_tracker and retry with another prompt. \n"
        "6. After ensuring the objects are tracked successfully, analyze tracked positions to solve the task.\n"
        "you can use detect_interaction to detect the interaction between the objects.\n"
        "7. Return final output wrapped in <answer>...</answer>.\n"
        f"8. Output JSON format: {json_schema}"
    )

    agent_msg = "Use tools as needed. Return concise reasoning summary and final JSON."
    COT_PROMPT = \
    "think step by step. If you think you should stop. output: <answer> ... <answer>" 
    human_msg = HumanMessage("Count the number of pullups in the video , explain your thought"+agent_msg)
    # human_msg = HumanMessage("explain the first frame and wrap it in <answer> ... <answer>")


    tools = sam3_tracker._llm_tools()
    # Initialize and bind (potentially multiple) tools to the model
    # model_with_tools = ChatGoogleGenerativeAI(model="gemini-2.5-flash", api_key="AIzaSyBt730C7DOdZRwlRgGN4bSpvBG78X5nTlw").bind_tools(tools)
    model_with_tools = ChatOpenAI(model="gpt-5.2").bind_tools(tools)
    # messages = [system_msg, human_msg]
    final_response, messages = run_agent(model_with_tools, human_msg, cfg, tools)
    format_agent_trace(messages, save_path="sam3_agent_trace.md")