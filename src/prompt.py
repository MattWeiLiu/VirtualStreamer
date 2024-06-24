class SamanthaPrompt(object):
    def __init__(self, config, daily_mail="", long_term_memory=""):
        if daily_mail:
            self.daily_mail = daily_mail
        else:
            self.daily_mail = config["prompt"]["default"]["daily_mail"]
        
        if long_term_memory:
            self.long_term_memory = long_term_memory
        else:
            self.long_term_memory = config["prompt"]["default"]["long_term_memory"]
        if isinstance(self.long_term_memory, dict):
            self.long_term_memory = '{' + str(self.long_term_memory) + '}'
        
        self.character_prompt = config["prompt"]["character_prompt"] % (self.daily_mail, self.long_term_memory)

        # self.memory_prompt = config["prompt"]["memory_prompt"]
        
        self.violation = config["violation"]
