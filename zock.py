import gradio as gr
import vertexai
from vertexai.generative_models import GenerativeModel, Part, FinishReason
import vertexai.preview.generative_models as generative_models

import os
import replicate
from groq import Groq

import base64
import os
import requests


# URL of the image
def img_savimg(url):

    # Send a GET request to the URL
    response = requests.get(url)

    # Check if the request was successful
    if response.status_code == 200:
        # Open a file in binary write mode
        pp = os.listdir("outs")
        with open(f"outs/vi_{len(pp)+1}.jpeg", "wb") as file:
            # Write the content of the response to the file
            file.write(response.content)
            file.close()
        print(f"Image successfully downloaded")
    else:
        print(f"Failed to download image. Status code: {response.status_code}")


# engine_id = "stable-diffusion-v1-6"
# api_host = os.getenv("API_HOST", "https://api.stability.ai")
# api_key = "sk-Pm8x3X8AaBZIYCJOI2bEouQ6eBWSfQCuliIzJzy02f3vgtwt"

# if api_key is None:
#     raise Exception("Missing Stability API key.")


def prompt_img(prompt, img_path):
    print("Entering the code")
    with open(img_path, "rb") as image_file:
        encoded_string = base64.b64encode(image_file.read()).decode("utf-8")

    output = replicate.run(
        "konieshadow/fooocus-api-realistic:612fd74b69e6c030e88f6548848593a1aaabe16a09cb79e6d714718c15f37f47",
        input={
            "prompt": prompt,
            "cn_type1": "ImagePrompt",
            "cn_type2": "ImagePrompt",
            "cn_type3": "ImagePrompt",
            "cn_type4": "ImagePrompt",
            "sharpness": 2,
            "image_seed": 6091967260935476000,
            "uov_method": "Vary (Subtle)",
            "image_number": 1,
            "guidance_scale": 3,
            "refiner_switch": 0.5,
            "negative_prompt": "unrealistic, saturated, high contrast, big nose, painting, drawing, sketch, cartoon, anime, manga, render, CG, 3d, watermark, signature, label",
            "uov_input_image": f"data:image/png;base64,{encoded_string}",
            "style_selections": "Fooocus V2,Fooocus Photograph,Fooocus Negative",
            "uov_upscale_value": 3,
            "outpaint_selections": "",
            "outpaint_distance_top": 0,
            "performance_selection": "Extreme Speed",
            "outpaint_distance_left": 0,
            "aspect_ratios_selection": "1152*896",
            "outpaint_distance_right": 0,
            "outpaint_distance_bottom": 0,
            "inpaint_additional_prompt": "",
        },
    )
    img_savimg(output[-1])
    print(output[-1])
    return output[-1]


def generate_prompt(img_data):

    print("Prompt Generating")

    client = Groq(api_key="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")

    chat_completion = client.chat.completions.create(
        messages=[
            {
                "role": "user",
                "content": f"Here is the information about the image {img_data},it tells about if it's an ad creative or not and ways to imrove the image. your task is to Use these informations and generate the perfect prompt to improve this image.",
            }
        ],
        model="llama3-8b-8192",
    )
    print("Prompt Generateddd!!!")
    try:
        print(chat_completion.choices[0].message.content)
    except:
        pass
    return chat_completion.choices[0].message.content


generation_config = {
    "max_output_tokens": 8192,
    "temperature": 1,
    "top_p": 0.95,
}

safety_settings = {
    generative_models.HarmCategory.HARM_CATEGORY_HATE_SPEECH: generative_models.HarmBlockThreshold.BLOCK_MEDIUM_AND_ABOVE,
    generative_models.HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT: generative_models.HarmBlockThreshold.BLOCK_MEDIUM_AND_ABOVE,
    generative_models.HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT: generative_models.HarmBlockThreshold.BLOCK_MEDIUM_AND_ABOVE,
    generative_models.HarmCategory.HARM_CATEGORY_HARASSMENT: generative_models.HarmBlockThreshold.BLOCK_MEDIUM_AND_ABOVE,
}


def generate(img_path):

    vertexai.init(project="eternal-cocoa-425609-v9", location="us-central1")
    model = GenerativeModel("gemini-1.5-flash-001")
    with open(img_path, "rb") as img_file:
        image_data = img_file.read()
    image1 = Part.from_data(mime_type="image/jpeg", data=image_data)
    responses = model.generate_content(
        [
            image1,
            """Analyze the uploaded image and determine if it is an ad creative do not use the symbol '*'.
Provide a detailed explanation of your assessment, considering the following aspects:
Visual elements (layout, color scheme, imagery)
Text content and messaging
Brand presence and logo placement
Call-to-action elements
Target audience appeal
If it is an ad creative, evaluate its effectiveness by discussing:
Clarity of the message
Visual impact
Emotional appeal
Brand consistency
Unique selling proposition
If it is not an ad creative, explain why it doesn't qualify as one, considering:
Lack of commercial intent
Absence of typical advertising elements
Purpose or context of the image
Regardless of whether it's an ad creative or not, suggest potential improvements or alterations that could enhance its effectiveness as an advertisement, such as:
Composition adjustments
Color palette modifications
Text refinements
Visual hierarchy enhancements
Addition or removal of specific elements
Do not use any boldings
Provide your analysis in a clear, point-by-point format without using bold text or other formatting.
Conclude with a brief summary of your overall assessment and key takeaways.""",
        ],
        generation_config=generation_config,
        safety_settings=safety_settings,
        stream=True,
    )
    full_response = ""
    for response in responses:
        full_response += response.text

    # prompt_img(generate_prompt(full_response), img_path)
    print(full_response)
    return full_response


custom_css = """
.gradio-container {
    font-family: 'Arial', sans-serif;
}
.container {
    max-width: 900px;
    margin: auto;
    padding-top: 1.5rem;
}
#component-0 {
    border: 2px dashed #3B82F6;
    border-radius: 8px;
    padding: 20px;
    margin-bottom: 20px;
}
.output-class {
    border-radius: 8px;
    border: 1px solid #3B82F6;
    padding: 20px;
    background-color: #1F2937;
}
.examples-row {
    display: flex;
    justify-content: center;
    gap: 10px;
    margin-bottom: 20px;
}
"""


def clear_output(image):
    if image is None:
        return ""
    return gr.update()


def display_local_image():
    # Replace this with the actual path to your local image
    ll = os.listdir("outs")
    print(ll)
    return f"outs\{ll[-1]}"


with gr.Blocks(
    theme=gr.themes.Soft(primary_hue="blue", secondary_hue="blue"), css=custom_css
) as iface:
    gr.Markdown("# 🎨 Ad Creative Analyzer")
    gr.Markdown(
        "Upload an image to check if it's an ad creative and get detailed analysis and improvement suggestions."
    )
    with gr.Row():
        with gr.Column(scale=2):
            input_image = gr.Image(type="filepath", label="Upload Image")
        with gr.Column(scale=1):
            gr.Markdown("### Try these examples:")
            example_images = gr.Examples(
                examples=[
                    ["test1.jpeg"],
                    ["test_zocket.jpeg"],
                    ["test3.jpeg"],
                ],
                inputs=[input_image],
                label="Example Images",
                examples_per_page=3,
            )
    analyze_button = gr.Button("Analyze Image", variant="primary")
    output = gr.Textbox(
        label="Analysis Result", lines=20, elem_classes=["output-class"]
    )

    # New button and image components

    analyze_button.click(fn=generate, inputs=input_image, outputs=output)
    input_image.change(fn=clear_output, inputs=input_image, outputs=output)

    # New click event for the local image button
    display_local_button = gr.Button("Generate", variant="secondary")
    local_image_output = gr.Image(label="Local Image")
    display_local_button.click(
        fn=display_local_image, inputs=None, outputs=local_image_output
    )

    gr.Markdown(
        "This tool uses advanced AI to analyze your image and provide insights on its effectiveness as an ad creative."
    )

iface.launch()
