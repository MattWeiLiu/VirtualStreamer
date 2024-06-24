# Her & Samantha
The name of this repo is reference from [here](https://en.wikipedia.org/wiki/Her_(film)).

# Usage
[//]: # (There are two kinds of situation to use this repo )
## Slackbot test
1. Make a copy of `config_template.yaml` and name it `config.yaml`.
2. Set the following variables to prevent any impact on product environment.
   1. `pubsub_subscription.launch: False`
   2. `pubsub_publish.send_userID: test`
   3. `chat_hist_pubsub.topic: "trash-collector"`
3. Modify `memory_type, days_of_week, prompt, prompt, violation` to suit your usage scenario
4. Modify `pach.sh` to suit your usage scenario.
5. Run `sh pach.sh` to upload your model registry
6. Deploy the endpoint on Vertex AI

## Broadcast on Production
1. Make a copy of `config_template.yaml` and name it `config.yaml`.
2. Set the following variables to run AVLiver on the production environment.
   1. `pubsub_subscription.launch: True`
   2. `pubsub_subscription.topic: "media17-live-events" or "media17-live-events-test" (The latter is for testing use)`
   3. `pubsub_subscription.pull_userID: <AVLiver userID>`
   4. `pubsub_publish.send_userID: <AVLiver userID>`
   5. `tts_endpoint.speaker_name: <speaker name>`
   6. `chat_hist_pubsub.topic: AI-Vliver-chat-hist`
   7. (optional) Set `moderator.launch: True` to launch moderator.
3. Modify `memory_type, days_of_week, prompt, prompt, violation` to suit your usage scenario
4. Modify `pach.sh` to suit your usage scenario.
5. Run `sh pach.sh` to upload your model registry
6. Deploy the endpoint on Vertex AI

## Plug-in (airflow & cloud function)
![flow_chart_plug-in.png](images%2Fflow_chart_plug-in.png)
### Airflow (option)
Airflow is used to tell AI-VLiver daily information that you want AI-VLiver to know in the prompt.
1. develop your own`information_collector.py`, which would collect information from real world and summary into a short message and return by`information_collector`.
2. modify `daily_information_dag.py` to set up your aivliver's endpoint.
3. put these codes to GCP composer: [ai-vliver-daily-information](https://console.cloud.google.com/composer/environments/detail/us-central1/ai-vliver-daily-information/monitoring?referrer=search&project=aiops-338206) to trigger airflow

airflow will inject message into `variable: daily_mail in prompt.py`


### Cloud function (option)
Cloud function is used to remember the long-term memory in redis and inject it into `variable: long_term_memory in prompt.py`

cloud function will call LLM to extract useful information from each conversation to construct the structure data(dict) to save in the redis and inject into prompt ultimately.

1. write down your own LLM prompt in the prompt.py (give few shot example to LLM)
2. set up you AIVLiver's endpoint id
3. set a new key name to replace `long_term_memory`


---
Note: The name of this repo is reference from [here](https://en.wikipedia.org/wiki/Her_(film)).