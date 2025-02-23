import json
import time

from src.Module.LLM.Gemini import get_gemini_response
from src.Module.LLM.GPTVision import get_gpt_response
from src.Module.LLM.Groq import get_groq_response
from src.Storage.reader import load_task_description
from src.Utilities.process_json import detect_json
from collections import deque
from concurrent.futures import ThreadPoolExecutor, as_completed
import cv2

from src.Utilities.sentence_similarity import filter_new_items, find_most_relevant_history


def get_llm_response(image_frame=None, image_path=None, image_queue=None, text=None, instruction=None, max_tokens=4000,
                     history=None, user_comment=None, knowledge_generation_model="gemini", language="en",
                     user_profile=None, pipeline="AiGet", last_response=None):
    # Generate Environment Description
    if user_comment:
        user_request = str(text) + str({"User Question": user_comment})
    else:
        user_request = text

    if user_profile is None:
        user_profile = load_task_description("user_profile")

    context_description = None

    start_time = time.time()

    if pipeline == "AiGet":
        instruction = load_task_description("context_description").replace("[USER_PROFILE_PLACEHOLDER]", user_profile)
        if last_response is not None and user_comment is not None and user_comment != "":
            user_request = str(user_request) + str({"Previous AI Response": last_response})

        context_description = get_gemini_response(image_frame=image_frame, image_path=image_path,
                                                  image_queue=image_queue.copy(),
                                                  system_instruction=instruction,
                                                  model="gemini-2.0-flash-exp",
                                                  text=str(
                                                      user_request) + " Don't be lazy in analyzing. Try your best to describe the environment (e.g., detailed species of plants or animals).",
                                                  temperature=0.3)
        # messages = [
        #     {
        #         "role": "system",
        #         "content": [
        #             {
        #                 "type": "text",
        #                 "text": instruction}
        #         ]
        #     },
        # ]
        #
        # context_description = get_gpt_response(conversation_history=messages, image_frame=image_frame,
        #                                        image_path=image_path, image_queue=image_queue, text=user_request,
        #                                        model="gpt-4o-2024-08-06")
        print("Context Description: \n", json.dumps(detect_json(context_description), indent=2))

        most_relevant_history = find_most_relevant_history(context_description, history, top_n=10)
        history = most_relevant_history

        # Generate Knowledge
        user_request = {"context description": context_description, "Requested Response Type": text['Response Style']}
        if user_comment is not None:
            if len(user_comment) > 0:
                user_request["user questions"] = user_comment

        print(user_request)

        instruction = load_task_description("knowledge_generation_filter").replace("[USER_PROFILE_PLACEHOLDER]",
                                                                                   user_profile)

        if knowledge_generation_model.startswith("gpt") or knowledge_generation_model.startswith("llama"):
            messages = [
                {
                    "role": "system",
                    "content": instruction
                },
                {
                    "role": "system",
                    "content": "suggestion history (*avoid similar content*): " + str(history)
                },
                {
                    "role": "user",
                    "content": str(user_request)
                }
            ]
            if knowledge_generation_model.startswith("gpt"):
                knowledge_candidate = get_gpt_response(conversation_history=messages, model=knowledge_generation_model,
                                                       temperature=0.3, max_tokens=4096)
            else:
                knowledge_candidate = get_groq_response(conversation_history=messages, model=knowledge_generation_model,
                                                        temperature=0, max_tokens=6000)
        else:
            knowledge_candidate = get_gemini_response(image_path=image_path,
                                                      image_queue=deque(list(image_queue)[::8]),
                                                      system_instruction=instruction,
                                                      text=str(user_request),
                                                      history=str(history), temperature=0.3, model=knowledge_generation_model,
                                                      image_size=150,
                                                      image_process_notes="Here are FPVs for reference (please combined it with provided contextual description)")
    elif pipeline == "Baseline_w_o_S":
        instruction = load_task_description("knowledge_generation_wo_s").replace("[USER_PROFILE_PLACEHOLDER]",
                                                                                 user_profile)
        knowledge_candidate = get_gemini_response(image_frame=image_frame, image_path=image_path,
                                                  image_queue=image_queue.copy(),
                                                  system_instruction=instruction,
                                                  text=str(user_request), model="gemini-2.0-flash-exp",
                                                  temperature=0.3)
    else:
        instruction = load_task_description("knowledge_generation_wo_sp")
        knowledge_candidate = get_gemini_response(image_frame=image_frame, image_path=image_path,
                                                  image_queue=image_queue.copy(),
                                                  system_instruction=instruction,
                                                  text=str(user_request), model="gemini-1.5-pro-001",
                                                  temperature=0.3)

    # print("Knowledge Candidate: \n", knowledge_candidate)

    generated_knowledge = detect_json(knowledge_candidate)
    if generated_knowledge["Suggestion Type"] == "Live Comments":
        generated_knowledge["AI Suggestion"] = filter_new_items(generated_knowledge["AI Suggestion"], history, threshold=0.75)
        if not generated_knowledge["AI Suggestion"]:
            return detect_json(knowledge_candidate), None
    # print("Generated Knowledge: \n", json.dumps(generated_knowledge, indent=2))

    with ThreadPoolExecutor() as executor:
        # Concurrently run iconify_knowledge and locate object tasks
        future_to_name = {
            executor.submit(get_gemini_response,
                            system_instruction=load_task_description("process_output"),
                            text=str({"AI Suggestion": generated_knowledge.get("AI Suggestion"),
                                      "Requested Response Language": language}),
                            temperature=1): "iconify_knowledge",
            executor.submit(get_gemini_response,
                            system_instruction='Return the frame number (only one) that matches the mentioned subjects in "AI suggestions" and return the relative location (0,1) for the xyxy boundary. I will show this image to users for location reference. Output format: {"frame_number": "(start from 1)", "objects":["xxx"],"box":[[relx1, rely1, relx2, rely2]]}',
                            image_frame=image_frame, image_path=image_path, image_queue=image_queue.copy(),
                            text=str({"Primary": generated_knowledge.get("Primary").get("Name"),
                                      "Peripheral": generated_knowledge.get("Peripheral").get("Name"),
                                      "AI Suggestion": generated_knowledge.get("AI Suggestion")}),
                            temperature=1): "location"
        }

        results = {}
        for future in as_completed(future_to_name):
            name = future_to_name[future]
            try:
                results[name] = future.result()
            except Exception as exc:
                print(f"{name} generated an exception: {exc}")

    iconify_knowledge = results.get("iconify_knowledge")
    location = detect_json(results.get("location"))
    # print("frame no:", location.get("frame_number"))
    # location = {"frame_number": 1}

    # Process Output Format
    iconify_knowledge = detect_json(iconify_knowledge)
    iconify_knowledge["Suggestion Type"] = generated_knowledge["Suggestion Type"]
    # process_output["User Profile"] = generated_knowledge["User Profile"]
    if context_description is not None and detect_json(context_description) is not None:
        iconify_knowledge["Context Description"] = detect_json(context_description).get("Context Description")
    iconify_knowledge["English AI Suggestion"] = generated_knowledge.get("AI Suggestion")
    iconify_knowledge["Full Knowledge Generation"] = generated_knowledge
    print("process_output: \n", iconify_knowledge)
    print(f"\nTotal Request took {time.time() - start_time:.2f}s\n")
    return str(json.dumps(iconify_knowledge, indent=4, ensure_ascii=False)), location


if __name__ == '__main__':

    image_paths = ["/Users/Vincent/Documents/GitHub/PandaExp/data/sample_data/compare_snack.jpeg"]
    image_queue = deque(maxlen=3)
    for path in image_paths:
        # Load image using OpenCV
        image = cv2.imread(path)
        # convert the image from BGR to RGB
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image_queue.append(image)

    chat_history = [
        "🕷️🌸Spider lilies are not true lilies but are members of the amaryllis family, often called 'magic lilies' due to their overnight blooming.",
        "🕷️🌸Spider lilies can bloom overnight, making them seem magically appear."]
    chat_history = []
    response = get_llm_response(image_queue=image_queue, knowledge_generation_model="gemini-1.5-pro-exp-0827",
                                history=chat_history,
                                language="English",
                                text={"time": "14:15", "Response Style": "Live Comments",
                                      "intention": "compare the flavor of two Coo series by UHA Mikakuto. One is Kyoho grape flavor and other is white grape flavor"},
                                pipeline="AiGet")

    response, location = response
    response = detect_json(response)
    print(response.get("AI Suggestion").get("v"))
    print(response.get("AI Suggestion").get("a"))
    print(response.get("English AI Suggestion"))
    print(json.dumps(response, indent=2))
