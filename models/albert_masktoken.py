from typing import List
from random import choice, sample

from transformers import AlbertTokenizer, AlbertForMaskedLM
from textblob import TextBlob
from textblob.exceptions import NotTranslated
from nltk.corpus import stopwords

class AlbertMaskToken:
    """
    A question generating algorithm
    based on ALBERT language model
    trained on the masked token task
    (no training required
    Like Zero-Shot Learning)
    """

    def __init__(self) -> None:
        model_name = 'albert-base-v2'
        self.tokeniser = AlbertTokenizer.from_pretrained(model_name)
        self.model = AlbertForMaskedLM.from_pretrained(model_name)
        self.mask_token = "[MASK]"
        self.seed_tokens = ["can","what","when","why"]
        self.stopword_tokens = set(stopwords.words('english'))
        self.ignored_albert_tokens = ["evalle","joyah"]

    def quick_generate(self, text:str) -> str:
        """
        generate a question
        without using the albert model
        """
        question = self.extract_seed_question(text)
        return self.rephrase_for_fluency(question) 

    def generate(
        self, text:str, max_iterations:int=5, 
        post_processing:bool=True, verbose:bool=False
    ) -> str:
        """
        generate a relevant question
        for given a sentence
        """
        question = self.extract_seed_question(text)
        if verbose:
            print(f"seed:{question}")
        for iteration in range(max_iterations):
            question = self.grow_question(question)
            if verbose:
                print(f"{iteration}:{question}")
        if post_processing:
            question = self.remove_tokens_for_fluency(question,self.ignored_albert_tokens)
            question = self.rephrase_for_fluency(question) 
        return question

    def extract_seed_question(self,text:str) -> str:
        """
        to add some relevance to the generated question
        a keyword is chosen from the input text and 
        acts as a seed from which to grow the question
        """
        seed_question = choice(self.seed_tokens)
        keywords = self.extract_keywords(text)
        if any(keywords):
            number_of_samples = min(len(keywords),2)
            seed_question += f" {' '.join(sample(keywords,number_of_samples))}"
        seed_question += " ?"
        return seed_question
    
    def extract_keywords(self, text:str) -> List[str]:
        """
        e.g. "i like cats and dogs"
        -> ["cats","dogs"]
        """
        return list(filter(lambda token:token not in self.stopword_tokens, text.split()))

    def grow_question(self, question:str) -> str:
        """
        given a question
        insert a mask into the sentence
        and predict a word that could be placed there
        """
        tokens = question.split()
        index = choice(range(len(tokens)))
        new_tokens = tokens[:index] + [self.mask_token] + tokens[index:]
        return self.predict_masked_words(text=' '.join(new_tokens))

    def predict_masked_words(self, text:str) -> str:
        inputs = self.tokeniser(text, return_tensors="pt")
        tokens = self.get_albert_tokens(ids=inputs.input_ids[0])

        outputs = self.model(**inputs)
        predicted_tokens = self.get_albert_tokens(ids=outputs.logits[0].argmax(dim=1))
    
        replaced_tokens = self.replace_mask_tokens_with_predicted_tokens(tokens,predicted_tokens)
        return self.format_albert_tokens_as_string(replaced_tokens)
    
    def get_albert_tokens(self,ids:List[int]) -> List[str]:
        return self.tokeniser.convert_ids_to_tokens(ids)
    
    def replace_mask_tokens_with_predicted_tokens(self,tokens:List[str],predictions:List[str]) -> List[str]:
        for index,token in enumerate(tokens):
            if token==self.mask_token:
                tokens[index] = predictions[index]
        return tokens

    @staticmethod
    def format_albert_tokens_as_string(tokens:List[str]) -> str:
        """
        e.g. ['[CLS]', '▁the', '▁capital', '▁of', '▁france','[SEP]'] 
        -> 'the capital of france'
        """
        return ''.join(map(lambda token:token.replace("▁"," "), tokens[1:-1]))
    
    @staticmethod
    def rephrase_for_fluency(text:str) -> str:
        """
        pass it through a translator
        to make the grown sentence sound a bit more natural
        """
        analysis = TextBlob(text)
        try:
            return ' '.join(analysis.translate(to='ar').translate(to='en').words) + "?"
        except NotTranslated:
            return text
    
    @staticmethod
    def remove_tokens_for_fluency(text:str,tokens:List[str]) -> str:
        """
        e.g. tokens = [evalle]
        text = can evalle evalle evalle evalle like your cats ?
        -> can like your cats ?
        """
        for token in tokens:
            text = text.replace(token,'')
        return text
