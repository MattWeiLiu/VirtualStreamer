from vertexai.generative_models import GenerativeModel, Part
from google.cloud import aiplatform, storage
from google.protobuf import json_format
from google.protobuf.struct_pb2 import Value
import vertexai.preview.generative_models as generative_models


class VideoSummarizer(object):
    def __init__(self):
        self.model = GenerativeModel("gemini-1.5-pro-001")
        self.generation_config = {
            "max_output_tokens": 1024,
            "temperature": 0.2,
            "top_p": 0.95,
        }
        self.safety_settings = {
            generative_models.HarmCategory.HARM_CATEGORY_HATE_SPEECH: generative_models.HarmBlockThreshold.BLOCK_ONLY_HIGH,
            generative_models.HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT: generative_models.HarmBlockThreshold.BLOCK_ONLY_HIGH,
            generative_models.HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT: generative_models.HarmBlockThreshold.BLOCK_ONLY_HIGH,
            generative_models.HarmCategory.HARM_CATEGORY_HARASSMENT: generative_models.HarmBlockThreshold.BLOCK_ONLY_HIGH,
        }
        self.storage_client = storage.Client()
        self.bucket = self.storage_client.bucket("17live_game_video")
        self.prompt = """This is a video of a streamer playing Super Mario game.
Here are some game rules:
## Game Overview
#### Main Character: Mario
#### Objective: Rescue Princess Peach who has been kidnapped by Bowser
#### Story Setting: The game is set in a fantasy world called the "Mushroom Kingdom."

## Game Features
#### Power-ups:
 - Mushroom: Makes Mario grow larger, allowing him to take one extra hit
 - Fire Flower: Gives Mario the ability to shoot fireballs
 - Star: Makes Mario invincible for a short period

#### Enemies:
 - Goomba: (brown mushrooms like characters) Basic enemies, can be defeated by jumping on them
 - Koopa Troopa: (turtle-like characters)Faster-moving enemies, usually require more precision to defeat
 - Bowser: (leader of the turtle-like Koopa, )The ultimate boss at the end of each worldGame Features

#### Possible scenarios in the game:
 - Running into enemies will hurt you.
 - Falling into a hole will kill you.
 - Eating a mushroom will make you bigger and give you an extra life.
 - Jumping on enemies will defeat them.
 - Eating a fire flower will allow you to throw fireballs to attack enemies.
 - Eating a star will grant you temporary invincibility.

You need to briefly summarize the game's progression based on the video content and tell me how do you feel about the journey you just had from your first-person perspective? Do you think the player performed well or poorly? (e.g., poorly because they kept making you die or run into enemies, very well because they made few mistakes and quickly completed the levels)
You have to provided information from a first-person perspective."""

    def generate(self, gcs_path):
        blob = self.bucket.blob(gcs_path)
        binary_data = blob.download_as_string()

        video = Part.from_data(
            mime_type="video/mp4",
            data=binary_data
        )

        response = self.model.generate_content(
            [video, self.prompt],
            generation_config=self.generation_config,
            safety_settings=self.safety_settings,
            # stream=True
        )
        self.update_ai_vliver({"Action": "update_info", "message": response.text})
        return response.text

    def update_prompt(self, new_prompt):
        self.prompt = new_prompt
        return new_prompt

    def update_ai_vliver(self, instances):
        """
        `instances` can be either single instance of type dict or a list
        of instances.
        """
        # The AI Platform services require regional API endpoints.
        client_options = {"api_endpoint": "us-central1-aiplatform.googleapis.com"}

        # Initialize client that will be used to create and send requests.
        # This client only needs to be created once, and can be reused for multiple requests.
        client = aiplatform.gapic.PredictionServiceClient(client_options=client_options)

        # The format of each instance should conform to the deployed model's prediction input schema.
        instances = instances if type(instances) == list else [instances]
        instances = [
            json_format.ParseDict(instance_dict, Value()) for instance_dict in instances
        ]
        parameters_dict = {}
        parameters = json_format.ParseDict(parameters_dict, Value())
        endpoint = client.endpoint_path(
            project="aiops-338206", location="us-central1", endpoint="6679091135064834048"
        )
        response = client.predict(
            endpoint=endpoint, instances=instances, parameters=parameters
        )
        return response
