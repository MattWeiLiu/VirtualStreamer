import emoji
import warnings
import re
import json
import gc
import time
import io
import base64
import logging
from pydub import AudioSegment
from datetime import datetime, timedelta, date
from concurrent.futures import ThreadPoolExecutor
from langchain.memory import ConversationBufferWindowMemory, ConversationSummaryMemory
from langchain.chains import LLMChain
from langchain.prompts.prompt import PromptTemplate
from langchain.chat_models import ChatVertexAI
from langchain_community.chat_models import ChatOpenAI
from langchain.llms import VertexAI
from langchain.schema import AIMessage
from google.cloud import pubsub_v1
from opencc import OpenCC
from .prompt import *
from .Moderator import Moderator
from .vertexai_util import vertexai_predict_aiplatform
from .qa_chain_util import qa_chain
warnings.filterwarnings('ignore')

logger = logging.getLogger('AIVLIVER')


class Samantha(object):
    def __init__(self, config, daily_mail=None, long_term_memory=''):
        self.config = config
        self.days_of_week = config["days_of_week"]
        self.memory_type = config['memory_type']
        self.chat_model = config["chat_model"]
        self.chat_id = self.config['pubsub_publish']['send_userID']
        self.cc = OpenCC('s2t')
        self.daily_mail = daily_mail if daily_mail else config["prompt"]["default"]["daily_mail"]
        self.long_term_memory = (long_term_memory if long_term_memory else
                                 config["prompt"]["default"]["long_term_memory"])
        self.viewer_cache_d = {}
        self.speaker = self.config['tts_endpoint']['speaker_name']
        self.language = self.config['tts_endpoint']['language']
        self.sample_rate = self.config['tts_endpoint']['sample_rate']
        self.voice_format = self.config['tts_endpoint']['voice_format']
        self.speed_adj = 0.
        self.volume_adj = 0.
        self.external_content = self.config['external_content']
        self.input_var = []
        self.initial_chain()
        
        # Moderator
        self.moderator = Moderator(config)

        # initial pubsub client
        self.publisher = pubsub_v1.PublisherClient()
        
        # ThreadPoolExecutor
        self.executor = ThreadPoolExecutor()
        
    def initial_chain(self, prior_memory=None):
        # LLM initialization
        logger.info('use LLM model: {}'.format(self.chat_model))
        if 'gemini' in self.chat_model:
            llm = VertexAI(model_name=self.chat_model,
                           project="aiops-338206",
                           max_output_tokens=150, temperature=0.9
                           )
        elif self.chat_model == "chat-bison@002":
            llm = ChatVertexAI(model_name="chat-bison@002",
                               project="aiops-338206",
                               max_output_tokens=150,
                               temperature=1.0
                               )
        elif 'gpt' in self.chat_model:
            llm = ChatOpenAI(model=self.chat_model, temperature=1.0, max_tokens=150)
            
        else:
            raise Exception("not support llm: {} yet".format(self.chat_model))
        
        self._prompt = SamanthaPrompt(config=self.config,
                                      daily_mail=self.daily_mail,
                                      long_term_memory=self.long_term_memory)
        
        self.prompt = PromptTemplate.from_template(self._prompt.character_prompt)
        
        if self.config['chain_type'] == 'conversation':
            if self.memory_type == "summary":
                self.memory = ConversationSummaryMemory(llm=VertexAI(model_name="text-bison@002",
                                                                     project="aiops-338206",
                                                                     max_output_tokens=512,
                                                                     top_p=1,
                                                                     temperature=0.7),
                                                        input_key='history')

                if prior_memory:
                    self.memory.buffer = prior_memory
            elif self.memory_type == "bufferwindow":
                self.k = 20
                self.memory = ConversationBufferWindowMemory(k=self.k, input_key='history')
                if prior_memory:
                    self.memory.chat_memory = prior_memory
            else:
                self.memory = None

            if self.memory:
                self.chain = LLMChain(prompt=self.prompt, memory=self.memory, llm=llm, verbose=False)
            else:
                self.chain = LLMChain(prompt=self.prompt, llm=llm, verbose=False)
                
        elif self.config['chain_type'] == 'qa':
            self.chain = qa_chain(self.config)
            self.memory = self.chain.get_qa_chain_memory(prior_memory)
            retriver = self.chain.get_retriver_from_external(self.external_content)
            self.chain.get_qa_chain(llm, retriver, self.memory, self.prompt)

    def conversation(self, query_str) -> [str, str]:
        viewer_open_id = ""
        if query_str.startswith("觀眾名字(") or query_str.startswith("觀眾("):
            match = re.search(r'\((.*?)\)', query_str)
            viewer_open_id = match.group(1)
            start = match.span()[1]
            comment = query_str[start:]
            comment = comment[3:] if comment.startswith("說了：") else comment.strip()
            query_str = viewer_open_id + ": " + comment

        current_time = datetime.now()
        current_time = current_time + timedelta(hours=8)
        day_of_week = date.today().weekday()
        today_weekday = self.days_of_week[day_of_week]
        time_str = current_time.strftime("%Y-%m-%d %H:%M")
        current_time = f"今天是{today_weekday}，現在時間是 {time_str}"
        
        logger.info('  -- input LLM: {}'.format(query_str))
        try:
            if "{current_time}" in self._prompt.character_prompt:
                subtitle = self.chain.predict(input=query_str, current_time=current_time)
            else:
                subtitle = self.chain.predict(input=query_str)
            subtitle = self.dialogue_truncation(subtitle)
        except Exception as e:
            print(e)
            subtitle = self.recursive_conversation(query_str, current_time, viewer_open_id, 1)
        
        if self.is_violation(subtitle):
            self.executor.submit(self.send_to_hist_pubsub, {"Datetime": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                                                            "ChatID": self.chat_id,
                                                            "Viewer": viewer_open_id,
                                                            "Input": query_str,
                                                            "Response": subtitle,
                                                            "Action": "Conversation",
                                                            "VIOLATION": True})
            subtitle = self.recursive_conversation(query_str, current_time, viewer_open_id, 1)

        # send to pubsub
        self.executor.submit(self.send_to_hist_pubsub, {"Datetime": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                                                        "ChatID": self.chat_id,
                                                        "Viewer": viewer_open_id,
                                                        "Input": query_str,
                                                        "Response": subtitle,
                                                        "Action": "Conversation",
                                                        "VIOLATION": False})

        # simple Chinese to traditional Chinese
        if self.config["s2t"]:
            subtitle = self.cc.convert(subtitle)
        speech = self.subtitle_process(subtitle)
        
        logger.info('response: {}'.format(subtitle))
        resp_d = self.tts(subtitle, speech)
        self.send_to_action_pubsub(resp_d)
        
        logger.info('Done one round conversation!!')
        
        # clear chat_memory
        if self.memory:
            self.clear_memory()
        gc.collect()
        return subtitle, speech
    
    def tts(self, subtitle, speech):
        logger.info('  -- call tts api......')
        tts_response = vertexai_predict_aiplatform(project=self.config['tts_endpoint']['project_id'],
                                                   location=self.config['tts_endpoint']['location'],
                                                   endpoint_id=self.config['tts_endpoint']['endpoint_id'],
                                                   instances=[{'speaker': self.speaker,
                                                               'language': self.language,
                                                               'speed': self.speed_adj,
                                                               'increase_in_db': self.volume_adj,
                                                               'text': speech,
                                                               'sample_rate': self.sample_rate,
                                                               'voice_format': self.voice_format
                                                               }]
                                                   )
        logger.info('  -- get wav encode str from api!')
        length = get_voice_length(tts_response.predictions[0])
        
        # sent wav and respond to pubsub
        resp_d = {"type": 112,
                  "aiVLiverAction": {"command": "speak",
                                     "subtitle": subtitle,
                                     "wav": tts_response.predictions[0],
                                     "length": length,
                                     "interrupt": False,
                                     }
                  }
        return resp_d
    
    def send_to_hist_pubsub(self, data_dict):
        topic_path = self.publisher.topic_path(self.config['chat_hist_pubsub']['project'],
                                               self.config['chat_hist_pubsub']['topic'])
        msg_str = json.dumps(data_dict, ensure_ascii=False)
        self.publisher.publish(topic_path, data=msg_str.encode("utf-8"))
        
    def send_to_action_pubsub(self, data_dict):
        topic_path = self.publisher.topic_path(self.config['action_pubsub']['project'],
                                               self.config['action_pubsub']['topic'])
        msg_str = json.dumps(data_dict, ensure_ascii=False)
        self.publisher.publish(topic_path, data=msg_str.encode("utf-8"), userID=self.chat_id)

    def recursive_conversation(self, query_str, current_time, viewer_open_id, j) -> [str, str]:
        try:
            if "current_time" in self.input_var:
                subtitle = self.chain.predict(input=query_str, current_time=current_time)
            else:
                subtitle = self.chain.predict(input=query_str)
            subtitle = self.dialogue_truncation(subtitle)
            fail_flag = False
        except Exception:
            subtitle = 'sorry, I suffer some error >"<'
            fail_flag = True
        
        if self.is_violation(subtitle) or fail_flag:
            self.executor.submit(self.send_to_hist_pubsub, {"Datetime": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                                                            "ChatID": self.chat_id,
                                                            "Viewer": viewer_open_id,
                                                            "Input": query_str,
                                                            "Response": subtitle,
                                                            "Action": "Conversation",
                                                            "VIOLATION": True})
            if j:
                subtitle = self.recursive_conversation(query_str, current_time, viewer_open_id, j - 1)
            else:
                subtitle = self.config['violation_respond']
                if self.memory:
                    self.memory.clear()  # if reach j times, clear memory to reset.
        return subtitle

    def is_violation(self, subtitle) -> bool:
        for key in self._prompt.violation:
            if key in subtitle:
                return True
        return False

    def trigger(self, active=False):
        subtitle = self.moderator.trigger(self.prompt.template, self.memory, active)
        subtitle = self.dialogue_truncation(subtitle)

        # simple Chinese to traditional Chinese
        if self.config["s2t"]:
            subtitle = self.cc.convert(subtitle)
        speech = self.subtitle_process(subtitle)
        # Save to memory
        self.chain.memory.save_context({"input": "",
                                        "history": self.chain.memory.buffer},
                                       {"output": subtitle})

        # Store to Bigquery
        self.executor.submit(self.send_to_hist_pubsub, {"Datetime": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                                                        "ChatID": self.chat_id,
                                                        "Viewer": 'moderator',
                                                        "Input": "",
                                                        "Response": subtitle,
                                                        "Action": "Trigger",
                                                        "VIOLATION": False})
        
        logger.info('response: {}'.format(subtitle))
        resp_d = self.tts(subtitle, speech)
        
        self.send_to_action_pubsub(resp_d)
        logger.info('Done one round conversation!!')
        
        # clear chat_memory
        if self.memory:
            self.executor.submit(self.clear_memory())
        gc.collect()
        return subtitle, speech

    def greeting(self, open_id):
        # is_greeting = self.moderator.greeting(open_id, self.long_term_memory)

        last_visited = self.viewer_cache_d.get(open_id, None)
        is_greet = False if last_visited and (time.time() - last_visited) < 600 else True

        if is_greet:  # case that user come in first time in last 10min
            if isinstance(self.long_term_memory, str):  # long_term_memory is a dict string
                json_acceptable_string = self.long_term_memory.replace("'", "\"")
                try:
                    long_term_memory_d = json.loads(json_acceptable_string)
                except Exception:
                    long_term_memory_d = {}
            else:  # case long_term_memory = {}
                long_term_memory_d = self.long_term_memory

            if (open_id not in long_term_memory_d.keys()) and (int(time.time()) % 4 > 0):
                # if open_id is not fans in long term memory, then sample to greet
                logger.info('  --> skip greeting for viewer {}'.format(open_id))
                is_greet = False

        self.viewer_cache_d[open_id] = time.time()
        
        if is_greet:
            subtitle, speech = self.conversation(f"觀眾({open_id}) 進入了直播間")
            return subtitle, speech
        return None, None

    def subtitle_process(self, subtitle):
        # delete emoji & symbol
        speech = emoji.get_emoji_regexp().sub(u'', subtitle)
        speech = re.sub(r'[_.,"\'\-?:!;@/]', " ", speech)
        return speech

    def access_current_summary(self):
        if self.chain.memory:
            if self.config['chain_type'] == 'conversation':
                return self.chain.memory.load_memory_variables({})['history']
            elif self.config['chain_type'] == 'qa':
                return self.chain.memory.load_memory_variables({})['chat_history']
        return ""

    def current_uuid(self):
        return self.chat_id

    def clear_memory(self):
        if isinstance(self.memory, ConversationBufferWindowMemory):
            if len(self.memory.buffer) > self.k:
                self.memory.buffer.pop(0)
                self.memory.buffer.pop(0)
        elif isinstance(self.memory, ConversationSummaryMemory):
            self.memory.chat_memory.clear()
            
    def update_info(self, message, kind="character_prompt"):
        if kind == "character_prompt":
            self.config["prompt"]["character_prompt"] = message
        elif kind == "daily_mail":
            self.daily_mail = message

        if self.memory_type == "summary":
            prior_memory = self.chain.memory.buffer
        elif self.memory_type == "bufferwindow":
            prior_memory = self.chain.memory.chat_memory
        else:
            prior_memory = None
        self.initial_chain(prior_memory)
        
        return self.prompt.template
    
    def update_long_term_memory(self, long_term_memory):
        self.long_term_memory = long_term_memory
        
        if self.memory_type == "summary":
            prior_memory = self.chain.memory.buffer
        elif self.memory_type == "bufferwindow":
            prior_memory = self.chain.memory.chat_memory
        else:
            prior_memory = None
        self.initial_chain(prior_memory)
    
    def update_external_content(self, decoded_content):
        self.external_content = decoded_content
        
        if self.memory_type == "summary":
            prior_memory = self.chain.memory.buffer
        elif self.memory_type == "bufferwindow":
            prior_memory = self.chain.memory.chat_memory
        else:
            prior_memory = None
        self.initial_chain(prior_memory)
        
    def update_llm(self, model_name):
        self.chat_model = model_name
        self.initial_chain()
        
    def dialogue_truncation(self, subtitle):
        """
        Truncate the dialogue string imagined by LLM
        Returns:
            str: Truncated subtitle
        """
        m = re.search(r'\[', subtitle)
        if m:
            p = m.span()[0]
            subtitle = subtitle[:p]
            if self.memory_type == "bufferwindow":
                self.memory.buffer.pop()
                self.memory.buffer.append(AIMessage(content=subtitle))
        return subtitle

    
def get_voice_length(base64_str):
    data = base64.b64decode(base64_str)
    voice_file = io.BytesIO(data)
    audio_segment = AudioSegment.from_file(voice_file)
    length_in_milliseconds = len(audio_segment)
    return length_in_milliseconds
