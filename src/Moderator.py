import warnings
import random
import redis
import numpy as np
from datetime import datetime
from langchain.chains import LLMChain
from langchain.prompts.prompt import PromptTemplate
from langchain.chat_models import ChatVertexAI
warnings.filterwarnings('ignore')

# Redis client
redis_host = "10.5.157.99"
redis_port = 6379
redis_client = redis.Redis(host=redis_host, port=redis_port, db=0)


class Moderator(object):
    def __init__(self, config):
        self.viewer_cache_d = {}
        self.setting = config["prompt"]["moderator_prompt"]
        dic = {key: val["weight"] for key, val in self.setting.items()}

        self.strategies = list(dic.keys())
        self.probability = list(dic.values()) / np.sum(list(dic.values()))

    def trigger(self, template, memory, active) -> str:
        selection = np.random.choice(self.strategies, p=self.probability)
        mode = self.setting[selection]["mode"]
        prompt = self.setting[selection]["prompt"]

        feedback = ""
        if mode == "Default":
            prefix = self.setting[selection]["prefix"]
            feedback = self.default_mode(prefix, prompt, active)
        elif mode == "Generative":
            prefix = self.setting[selection]["prefix"]
            feedback = self.generative_mode(template, memory, prefix, prompt, active)

        return feedback

    @staticmethod
    def default_mode(prefix, prompt, active):
        if active:
            return prefix + random.choice(prompt)
        return "大家這麼安靜，" + prefix + random.choice(prompt)

    def generative_mode(self, template, memory, prefix, prompt, active):
        current_time = datetime.now()
        llm_chain = self.tmp_chain(template, memory)
        if active:
            return llm_chain.predict(input=prompt, current_time=current_time)
        return llm_chain.predict(input=prefix + prompt, current_time=current_time)

    @staticmethod
    def tmp_chain(template, memory):
        llm = ChatVertexAI(model_name="chat-bison@002",
                           project="aiops-338206",
                           max_output_tokens=1024,
                           top_p=1,
                           temperature=1)
        prompt = PromptTemplate(input_variables=["current_time", "history", "input"],
                                template=template)
        llm_chain = LLMChain(llm=llm, prompt=prompt, memory=memory)
        return llm_chain
