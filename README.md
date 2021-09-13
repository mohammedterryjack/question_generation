# Natural Language Question Generation
## Mohammed Terry-Jack
---

### Task Description:

return a `question (y)` given `N utterances (x)`

using them `freely` (using only the last utterance `x[N]`) 

or `contextually` (using all utterances `x[0:N]`)

---
### Example:

> Utterances (x):
>> A: Do you like movies?

>> B: Yes, I love Titanic!

> Question (y):
>>A: What do you think about it?

---
### Solution(s):

We present the following four solutions to this problem (each of which are explained in detail below):

- Baseline (Heuristics)
- Albert Mask Token (Pre-trained)
- Semantic-Image Captioning Character-based LSTM (Unsupervised Learning)
- Semantic-Image Captioning Character-based LSTM (Supervised Learning)

---
### Dependencies:

```python
transformers==4.5.1
textblob==0.15.3
numpy==1.19.5
pandas==1.0.1
keras==2.6.0
spacy==2.2.3
nltk==3.3
```

---
### Tests:

```
python interact_albert.py

python interact_lstm.py
```
---

### Background Research:

**Linguistic Considerations**:

Questions are far more than simply ways to request information from someone, they are often used for a variety of linguistic mechanisms, including providing answers or even to prevent an answer from being given (e.g. rhetorical questions).  They may be used to steer the conversation into a new direction, which would imply a change of subject from the utterances preceding it.  

However, for this task, we shall limit ourselves to generating coherent questions (the question is related to the utterances via a common meaning, topic, sentiment, or all of the above) which avoid breaking away from the prior utterances completely.

Coherence is the common thread that glues the question to the utterances before it. However, a very coherent dialogue is often characterised as dull and lacking in excitement or engagement.  This is because such features innevitably come from novelty and unpredictability in the response, which necessitates changing the meaning, topic, sentiment, etc.  In contrast, if we consider a dialogue which is very engaging (e.g. a comedy) we find very sudden changes and large breaks in the continuity of the dialogues.

Therefore, a good question (and a good dialogue) has competing objectives to consider. Minimising the differences will lead to greater coherence, while maximising the difference will ensure greater unpredictability, novelty and excitement.  There two opposing objectives end up delicately balancing one another in most dialogues.

**Question Generation**:

As for the computational task of Question Generation, numerous [papers with code](https://paperswithcode.com/task/question-generation) have been written on the task as well as some very unique ways fo frame this problem (see this [github repo](https://github.com/teacherpeterpan/Question-Generation-Paper-List) for some ideas)

---

# 1. Baseline (Heuristic)

**Description**:

Our baseline method (to compare all future approaches against) is a very simple heuristic which randomly chooses a seed token from a predefined list of words typically found at the beginning of generic questions

```python
from random import choice

seed_tokens = ["can","what","when","why"]
seed_question = choice(seed_tokens)
```

The keywords are then extracted from the utterances using a simple `stopword` filter (We use `NLTK` for its inbuilt stopword list)

```python
from nltk.corpus import stopwords

stopword_tokens = set(stopwords.words('english'))

def extract_keywords(text:str) -> List[str]:
    return list(filter(lambda token:token not in stopword_tokens, text.split()))

keywords = extract_keywords(text)
```

A couple of keywords are randomly sampled and appended to the seed question to ensure it has some superficial level of coherence 

```python
from random import sample

number_of_samples = min(len(keywords),2)
seed_question += f" {' '.join(sample(keywords,number_of_samples))}"
```

![](img/baseline/without_rephrasing.png)

In the example above, the keywords "live" and "near" were extracted and appended to the question word "why".  This can sound unnatural and so the final step is to translate the question into another language (e.g. arabic) and back into english to encourage more natural phrasing (We use `Textblob`'s inbuilt translate function).

```python
from textblob import TextBlob
from textblob.exceptions import NotTranslated

def rephrase_for_fluency(text:str) -> str:
    analysis = TextBlob(text)
    try:
        return ' '.join(analysis.translate(to='ar').translate(to='en').words) + "?"
    except NotTranslated:
        return text
```

This method is known as `spinning` and can modify words (e.g. like -> love)

![](img/baseline/example2.png)

![](img/baseline/cats_love.jpeg)

...or add additional words (e.g. you) for further fluency 

![](img/baseline/example1.png)

**Pros**:

- The baseline method does not require any training to begin using 
- it is a lightweight solution

**Cons**:

- Although spinning produces a natural sounding question, it does not ensure that the final question makes sense nor that it sounds intelligent (e.g. "Can you love cats?" is a strange question to ask).  
- This method will not introduce any semantic novelty into the question (the meaning of the question will be nearly identical to the original utterances) thus creating a coherent yet dull question overall


**Interact**:

If you wish to interact this model on its own, you can create a new python file containing the following script:

```python
from models.albert_masktoken import AlbertMaskToken
question = AlbertMaskToken().quick_generate(utterances)
```

**Dependencies**:

```python
textblob==0.15.3
nltk==3.3
```
---

# 2. Albert Mask Token (Pre-trained)

**Description**:
The next method maintains the main advantage of the baseline (i.e. it does not need training) yet improves upon it by being able to generate words and, thus, add novelty into the semantics of the question.  To generate questions, we use a pre-trained instance of `Albert` (Albert stands for "A lite BERT") as it promises to have similar performance to `BERT` while being lighter.  

The reason we use Albert is to do with the tasks it was trained on.  Most pretrained Generative language models (e.g. `GPT3`, `GPT-Neo`, `T5`, Bert, etc) are able to generate text, but they are not specifically trained to generate questions.  Now Albert is no exception, however, unlike T5 and the GPT family, BERT (and Albert by extension) are trained on a specific task which does make it possible for us to generate questions without having to retrain of fine-tune it at all!  This is the Masked Token Prediction task, 

![](img/albert/mask_token_prediction.png)

We can use Albert's ability to predict a masked token to trick it into inserting new words into a question, thus growing the sentence from within!  

e.g.
```python 
iteration 1:
	question: "what cats?"
	inserted mask: "what [MASK] cats?"
	predicted word: "what [cute] cats"

iteration 2:
	question: "what cute cats?"
	inserted mask: "what [MASK] cute cats?"
	predicted word: "what [are] cute cats?"

iteration 3:
	question: "what are cute cats?"
	inserted mask: "what are cute cats [MASK]?"
	...etc
```

This method of natural language generation via 'inserting' new words differs from most generation tasks (which involve predicting the next word or character at the end of a given sequence), but the reason we choose such a method is due to the convenience of constraining the generated text into a desired format (i.e. the form of a question) by specifying the initial and final tokens (e.g. "what ... ?"). 


First we construct the function which inserts the mask token somewhere inside the question

```python
from random import choice

mask_token = "[MASK]"

def grow_question(question:str) -> str:
    tokens = question.split()
    index = choice(range(len(tokens)))
    new_tokens = tokens[:index] + [mask_token] + tokens[index:]
    return predict_masked_words(text=' '.join(new_tokens))

```

In order to predict the masked token, we need to load in the pretrained Albert model from Huggingface's `transformers` library and pass it the question containing the masked tokens

```python
from transformers import AlbertTokenizer, AlbertForMaskedLM

model_name = 'albert-base-v2'

tokeniser = AlbertTokenizer.from_pretrained(model_name)
model = AlbertForMaskedLM.from_pretrained(model_name)


def predict_masked_words(text:str) -> str:
    inputs = tokeniser(text, return_tensors="pt")
    outputs = model(**inputs)

	tokens = tokeniser.convert_ids_to_tokens(inputs.input_ids[0])
	predicted_tokens = tokeniser.convert_ids_to_tokens(outputs.logits[0].argmax(dim=1))
	
	replaced_tokens = replace_mask_tokens_with_predicted_tokens(tokens,predicted_tokens)
    return format_albert_tokens_as_string(replaced_tokens)
```

Replacing the masked token is simply a matter of finding the predicted token in the position of the masked token

```python
def replace_mask_tokens_with_predicted_tokens(tokens:List[str],predictions:List[str]) -> List[str]:
	for index,token in enumerate(tokens):
		if token==mask_token:
			tokens[index] = predictions[index]
	return tokens
```

And the tokens (with the masks replaced with predictions) are then converted into a string - cutting off the special tokens added by albert's tokeniser to the start ("[CLS]") and end ("[SEP]")

```python
def format_albert_tokens_as_string(tokens:List[str]) -> str:
    return ''.join(map(lambda token:token.replace("▁"," "), tokens[1:-1]))

```

We can repeat this process as many times as you like to grow the question iteratively

```python
def generate(text:str, max_iterations:int=5) -> str:
	question = extract_seed_question(text)
	for iteration in range(max_iterations):
		question = grow_question(question)
	return question

```

And thats it!

```python
generate(text="i like cats")
```

![](img/albert/cats_example1.png)

![](img/albert/cats_example2.png)

![](img/albert/stuffed_cat.jpeg)

**Pros**:

- The model works off-the-shelf without needing to be retrained or fine-tuned 

- The generated questions add far more novelty and engagement than the baseline, while keeping it relevant (e.g. "Are you wondering what cats really look like?" and "Could you love two of your favorite stuffed cats?" as questions to the utterance "i like cats")


**Cons**:

- The question forms are constrained to the typical Wh-questions (e.g. "who..?", "what...?", "where..?", etc) and cannot considerably vary from these formats for additional surprise and creativity often found in natural questions (e.g. "i always wanted furry cats but I hear they require a lot of grooming dont they?")

- The generated questions are not always perfect and sometimes require post-processing (i.e. removing certain spurious tokens, spinning for fluency, etc)

- The seed question still depends on the baseline's heuristics to extract a couple of keywords from the utterance  


**Interact**:

If you wish to interact this model on its own, you can create a new python file containing the following script:

```python
from models.albert_masktoken import AlbertMaskToken
question = AlbertMaskToken().generate(utterances)
```

**Dependencies**:

```python
transformers==4.5.1
```

---
# 3. Semantic-Image Captioning Character-based LSTM (Unsupervised Learning)

**Description**:

In order to add even more originality to our generated questions, we are going to have to start training a model. However, this can often lead to another problem for ML tasks, finding labelled data for your specific task (i.e. utterances -> question). If there are no datasets publically available, it can be a very expensive to label a new dataset from scratch!  Fortunately, for this particular task, it so happens that there is a way we can use unlabelled data to train a model to generate questions given utterances!

![](img/lstm/encoder_decoder.png)

The problem can be thought of as encoding the utterances into a semantic space and then decoding these semantics into a question (utterance -> semantics -> question). 

![](img/lstm/sentence_encoder.png) 

Since there are already a plethora of pre-trained methods readily available for embedding utterances into a semantic space (e.g. `word2vec`, `universal sentence encoder`, etc), we can leverage this fact to simplify our model's learning task. We simply need to train a decoder (a model that converts the semantics -> question) since the encoder (The model which converts the utterance -> semantics) is already solved.  We are essentially learning to decode the semantic vector of a pre-trained sentence encoder (and in so doing, cutting our model's learning objective in half). In other words, we are creating an `encoder-decoder` model by fitting a decoder (e.g. a `character-based LSTM`) to a pre-trained encoder (e.g. `spacy`). 

![](img/lstm/image_captioning2.png)

This approach is common for Image-Captioning tasks (whereby a decoder is trained to convert an image vector into a caption and the image vector is often obtained via a pre-trained image encoder, such as `Resnet`, etc).  Our formulation of the task, therefore, would be a form of `Semantic-Captioning`.

**But how does this allow our model to use unlabelled data?**  Here comes the secret sauce: 

Since the semantics of the utterances are formless (i.e. they are no longer dependent on how the words are phrased, etc) we can obtain the semantics from utterances which are formulated as questions too (e.g. question -> semantics -> question) - as the semantics of an utterance, phrased as a sentence or not, will look near identical in the semantic space!  Knowing this, we can simply use a large list of unlabelled questions for both the expected outputs (that the decoder must learn to generate) and their semantics can also act as the training inputs. The effect of this will be that the decoder learns to generate questions to any utterances with similar semantics to the question itself (ensuring the question is relevant and coherent to the utterances given). This is how we shall generate a question given the semantics of an utterance similar to that question. Thus we are actually doing `unsupervised learning` using only a set of unlabelled questions.


**Model**:

We build our character-based LSTM decoder in keras

```python
from keras import Model
from keras.layers import Input, Dropout, Dense, LSTM, Embedding, add

def build_character_based_LSTM(
    semantic_vector_length:int,
    character_vector_length:int,
    character_sequence_length:int,
    hidden_layer_length:int,
    dropout_rate:float,
    optimisation:str,
    activation:str,
    weights:Optional[str],
    loss:str,
) -> Model:
        meaning_layer1 = Input(shape=(semantic_vector_length,))
        meaning_dropout1 = Dropout(dropout_rate)(meaning_layer1)
        meaning_layer2 = Dense(hidden_layer_length, activation=activation)(meaning_dropout1)
        characters_layer1 = Input(shape=(character_sequence_length,))
        characters_layer2 = Embedding(character_vector_length, character_vector_length, mask_zero=True)(characters_layer1)
        characters_dropout2 = Dropout(dropout_rate)(characters_layer2)
        characters_layer3 = LSTM(hidden_layer_length)(characters_dropout2)
        layer3 = add([meaning_layer2, characters_layer3])
        layer4 = Dense(hidden_layer_length, activation=activation)(layer3)
        layer5 = Dense(character_vector_length, activation='softmax')(layer4)
        model = Model(
            inputs=[meaning_layer1, characters_layer1], 
            outputs=layer5
        )
        if weights is not None: model.load_weights(weights)
        model.compile(loss=loss, optimizer=optimisation)
        return model 

```
![](img/lstm/architecture.png)

To keep the overall model light, we use spacy as our pre-trained encoder

```python
from numpy import array
from spacy import load 

semantic_encoder = load('en_core_web_sm')

def _get_semantic_vector(text:str) -> array:
	if not any(text):text=start_token
    return semantic_encoder(text).vector
```

The output characters which the decoder model will be able to generate are the letters between a-z, a space and two special tokens to indicate the start and end of a question:

```python
a = ord('a')
z = ord('z')
start_token='|'
stop_token='?'
character_set = [' '] + list(map(chr,range(a,z+1))) + [start_token,stop_token]
number_of_characters = len(character_set)
```

The maximum recursion depth of our LSTM will determine the maximum length of the generated question (e.g. 100 characters). The other hyperparameters of our model are specified below:

```python
model = build_character_based_LSTM(
    semantic_vector_length=len(_get_semantic_vector("")),
    character_vector_length=number_of_characters,
    character_sequence_length=recursion_depth,
    hidden_layer_length=256,
    dropout_rate=.5,
    activation="relu",
    loss = "categorical_crossentropy",
    optimisation= "adam",
    weights=path_to_model_weights,
)
```

To generate a question, we iteratively predict the next character until we reach the maximum recursion_depth or we encounter the special token that indicates the end of a question

```python
def generate(sentence:str) -> str:
    generated_question = start_token
    for _ in range(recursion_depth):
        generated_question += _predict_next_character(
            meaning=sentence,
            contextual_characters=generated_question,
        )
        if stop_token in generated_question:
            break
    return generated_question
```

Whereby to predict the next character in the sequence, we input the semantic vector of the utterance and the generated sequence of characters so far as inputs into our trained LSTM.  

```python
    def _predict_next_character(meaning:str,contextual_characters:str) -> str:
        X_meaning = array([_get_semantic_vector(meaning)])
        X_characters = array([_pad(_convert_characters_to_index(contextual_characters))])
        output_vector = model.predict((X_meaning,X_characters),verbose=False)
        predicted_index = _greedy_decode(output_vector)
        return index_character_mapping.get(predicted_index)
    
```

Note that while the input utterance semantics are represented by a fixed-length vector (the output of the pre-trained encoder) the input sequence generated so far must also be converted into a fixed-length vector (achieved by padding the sequence with 0s if it is shorter than the full sequence length - which is the maximum recursion_depth)

```python
from keras.preprocessing.sequence import pad_sequences

def _pad(indexes:List[int]) -> array:
    return pad_sequences([indexes], maxlen=recursion_depth)[0]

```

Also note that the model expects the character indexes (as opposed to the characters as strings) and also outputs character indexes as predictions.  Therefore, we need to dictionaries to map characters into indexes (`character_index_mapping`) and indexes back into characters again (`index_character_mapping`).

```python
index_character_mapping = dict(enumerate(character_set))

character_index_mapping = {
    character:index for index,character in index_character_mapping.items()
}

def _convert_characters_to_index(characters:str) -> List[int]:
    return list(map(character_index_mapping.get,characters))

```

**Decoding Strategies**:

One of the most important factors about NLG (if not The Most Important) is the decoding strategy.  It is a hot topic of research!

>"Neural probabilistic text generators are far from
perfect; prior work has shown that they often generate text that is generic, unnatural and sometimes even non-existent". 

The model's outputs actually predict something akin to a probability distribution across the k possible output characters.  A `greedy` decoding strategy would simply take whichever index has the largest value (`argmax`) 

```python
def _greedy_decode(predicted_vector:array) -> int:
    return predicted_vector[0].argmax()
```

![](img/decoding/argmax_greedy.png)


However, this will not yield the best results in the long-run.  Even thought we greedily select the next most likely character (the `local optimum`) it does not mean this character leads to a sequence of characters which have the highest probability overall (the `global optimum`). 

![](img/decoding/global_optimum.png)

For example, the token "park" may have a higher probability (.36) than the token "grocery" (.15), but only when you pick "grocery" can you have the sequence "grocery store" which has an overall higher probability score (.135) than any sequence that could be created with the token "park" (e.g. "park today" has a lower probability of .12)

A simple alternative, then, is to sample from the entire output according to their probabilities (so that the characters with higher probabilities are more likely to be selected) but still allowing the chance for less likely characters to be selected too.  While this random sampling strategy works better than the greedy strategy (which actually only produces a sequence of " " characters in both examples below), it can lead to some fairly noisy results.

![](img/decoding/greedy_vs_probabilistic.png)

![](img/decoding/greedy_vs_probabilistic2.png)

Therefore we can introduce a constant called `temperature` to multiply with the probabilities as a way to increase the probability of picking the more likely characters and decrease the probability of picking the least likely characters (which are the source of the noise).  

![](img/decoding/temperature_sampling.png)

```python
from numpy.random import choice

def _temperature_decode(predicted_vector:array,temperature:float) -> int:
    probabilities = predicted_vector[0]
    probabilities *= temperature
    probabilities /= probabilities.sum()
    return choice(range(probabilities.size),1,p=probabilities)[0]


```
This produces better generated sequences of characters than greedy with far less noise than pure random sampling

![](img/decoding/temperature_v_greedy.png)

![](img/decoding/pokemon_mj.jpeg)

Perhaps one-step better than this is `nucleus sampling` (p-decoding) wherein only the indexes which make up the top_p % of the probability distribution are sampled from. 

```python
def _nucleus_decode(predicted_vector:array, top_p:float, temperature:Optional[float]) -> int:
    probabilities = predicted_vector[0]
    probabilities /= probabilities.sum()
    ranked_indexes = sorted(range(probabilities.size), key=lambda index:probabilities[index],reverse=True)
    probabilities.sort()
    ranked_probabilities = probabilities[::-1]
    for index in range(ranked_probabilities.size):
        if ranked_probabilities[:index].sum() >= top_p:
            nuclues_probabilities = ranked_probabilities[:index]
            nuclues_indexes = ranked_indexes[:index]
            break
    if temperature: nuclues_probabilities *= temperature
    nuclues_probabilities /= nuclues_probabilities.sum()
    return choice(nuclues_indexes,1,p=nuclues_probabilities)[0]
```

![](img/decoding/p_decode_vs_temperature.png)

Nucleus sampling produces slightly higher quality text as compared with temperature sampling.  However, we can also include temperature with nucleus sampling to improve it even further: 

![](img/decoding/p_decode_with_temperature.png)


However, for near optimal results, it has become the industry standard to use `beam search` (maintaining the top-k generated sequences).  However, maintaining multiple sequences (even if constrained to k beams) is much more computationally expensive than keeping track of a single sequence (as the above strategies do).  Is there any way to arrive at a near-optimal sequence without having to maintain multiple sequences (as with beam search)?  

![](img/decoding/beam_search.jpeg)

Beam search, however, is somewhat of an enigma.  It somehow manages to generate near-optimal sequences with only a small number of beams.  What is more mysterious, however, is the the phenomena known as the `beam search curse` 

> "a specific phenomenon where using a larger beam
size leads to worse performance"

In fact, "the success of beam search does not stem from its ability to approximate exact decoding in practice, but rather due to a hidden inductive bias embedded in the algorithm. This inductive bias appears to be paramount for generating desirable text". This [paper](https://aclanthology.org/2020.emnlp-main.170.pdf) analyses exactly how beam search is biasing generated sequences, in the hope that understanding this will allow for a more direct way to replicate the success of a beam search. The authors conclude that: 

> "We provide a plausible answer—inspired by psycholinguistic theory—as to why beam search (with small beams) leads to high-quality text" 

>"beam search enforces uniform information density in text"

>"beam search is trying to optimize for UID"

>"beam search has an inductive bias which can be linked to the promotion of uniform information density (UID), a theory from cognitive science regarding even distribution of information in linguistic signals"
>> "The UID hypothesis states that—subject to the constraints of the grammar—humans prefer sentences that distribute information (in the sense of information theory) equally across the linguistic signal, e.g., a sentence. In other words, human-produced text, regardless of language, tends to have evenly distributed surprisal, formally defined in information theory as negative log-probability"

![](img/decoding/beam_search_analysis.png)

They then suggest "a battery of possible sentence-level UID measures" as candidate "decoding objectives that explicitly enforce this property" to replicate the success of beam search (note the above example depicting one such objective on a single sequence which generates the exact same sequence as generated by a k=5 beam search)

>"This insight naturally leads to the development of several new regularizers that likewise enforce the UID property"


**Training**:

Training the model is as simple as fitting it to the formatted question data (We also save the model weights after each epoch so that we can load in the trained model later)

```python
def train(
    questions:List[str], 
    batch_size:int, epochs:int, 
    save_to_file_path:str="char_lstm_weights.hdf5",
) -> None:
    steps_per_epoch = len(questions)//batch_size
    for epoch in range(epochs):
        model.fit(
            _get_training_data(questions, batch_size), 
            verbose=True, epochs=1, steps_per_epoch=steps_per_epoch, 
        )
        model.save_weights(save_to_file_path)
```

To format the training data, we create a generator to iterate through the questions and get its semantics, then we convert the characters into indexes and iterate through them one by one, making each an expected character output (encoded as a one-hot vector) with the preceding characters the training inputs (padded to ensure consistent sequence lengths).  

```python
from tensorflow.keras.utils import to_categorical
def _one_hot_encode_output(index:int) -> array:
    return to_categorical([index], num_classes=number_of_characters)[0]
```

When a batch has been reached, the semantic vectors (`X_meaning`) and the input character sequences (`X_characters`) are yielded along with the expected character outputs (`Y_characters`)

```python
def _get_training_data(questions:List[str],batch_size:int) -> Tuple[Tuple[array,array],array]:
    iteration_count,X_meaning, X_characters, Y_characters = 0,[],[],[]
    while True:
        for question in questions:
            iteration_count+=1

            semantic_vector = _get_semantic_vector(question)
            processed_question = _preprocess_characters(question)
            character_indexes = _convert_characters_to_index(processed_question)
            for index in range(1, len(character_indexes)):
                contextual_character_indexes = character_indexes[:index]
                next_character_index = character_indexes[index]                    
                X_meaning.append(semantic_vector)
                X_characters.append(_pad(indexes=contextual_character_indexes))
                Y_characters.append(_one_hot_encode_output(next_character_index))

            if iteration_count==batch_size:
                yield ([array(X_meaning), array(X_characters)],array(Y_characters))
                iteration_count,X_meaning, X_characters, Y_characters = 0,[],[],[]
```

**Dataset**:

Now we can train our decoder model, we just need a dataset of unlabelled questions.  We can pick one of the many datasets which contain questions, such as the `Quora` question pairs dataset. 

The dataset is publically available and can be downloaded from [here](https://raw.githubusercontent.com/MLDroid/quora_duplicate_challenge/master/data/quora_duplicate_questions.tsv)

Since we dont require labelled questions, we will just extract all the unique questions in the dataset and ignore any labels they provide.

```python
from pandas import read_csv

def load_quora_data(path_to_file:str) -> List[str]:
    quora_dataset = read_csv(path_to_file, sep='\t', header=0)
    questions = quora_dataset.question1.append(quora_dataset.question2)
    questions = questions.apply(str)
    return sorted(set(questions))

```

Now we just sit back and wait for our model to train

```python
train_data = load_quora_data('data/quora_duplicate_questions.tsv')
train(train_data,batch_size=6,epochs=500,save_to_file_path="lstm_quora.hdf5") 
```

Slowly but surely, the model does learn to generate words from characters and even some very quora-sounding questions!

![](img/quora/quora-like-questions.png)

We can tell that these generated questions are being conditioned on the input utterances as all these generated questions are nearly identical for the given utterance "i hate you", despite the varying temperatures (which would otherwise cause very different questions to be generated)

![](img/quora/conditioned_on_utterance.png)

And even though we didnt wait for the model to finish training fully, the questions being generated are already more free and creative than the prior methods 

![](img/quora/creative_questions.png)


**Pros**:

- Only requires unlabelled data to train (unsupervised learning)

- Since the utterances are vectorised into a fixed-length semantic vector, the input length is not a problem (it can be any length - be it an entire conversation or a single word)!

- since we are only training a decoder, it is far lighter than learning an entire encoder-decoder model 

- Since the decoder is a character-level LSTM, it has a very small output size (as the vocabulary consists of just the alphabet)

- The model is able to generate more creative, natural sounding questions (which need not start with typical "Wh"-question words)

**Cons**:

- The encoder is very simple and a far superior semantic representation can be achieved if this is replaced

- The decoder could be trained in less iterations and produce more exressive questions still if we were to fine-tune a pre-trained generative language model

- The decoding strategy is currently temperature sampling and could be improved using nucleus-sampling, beam-search or one of the suggested uid measures


**Interact**:

If you wish to interact this model on its own, you can create a new python file containing the following script:

```python
from models.semantic_captioning_character_lstm import SemanticCaptioningCharacterLSTM

question_generator = SemanticCaptioningCharacterLSTM(recursion_depth=100,path_to_model_weights="models/lstm_dailydialog.hdf5")
question = question_generator.generate(utterances,top_p=.95,temperature=.9)
```

**Dependencies**:

```python
numpy==1.19.5
keras==2.6.0
spacy==2.2.3
```
---

# 4. Semantic-Image Captioning Character-based LSTM (Supervised Learning)

Although the model described above can be trained on unlabelled questions, it can also be trained on labelled data in a more typical supervised fashion, if such data is available! When training the model, we would just use the semantics of the utterances (instead of the question) and everything else remains the same as before.

This method can, in theory, produce a better trained model than the unsupervised version simply because you can train it to respond with questions that have very different semantics than the input utterances (to steer the conversation into a new direction or add a bit more novelty). 

It so happens that for this task, there is an abundance of datasets to map utterances to questions!  Its just that we need to be a bit creative in how we view these datasets.  For instance, every single question-answer dataset (e.g. `SQUAD2.0`, `Google Natural Questions`, etc) is mapping a question to an answer.  So if we reverse the order, we get a dataset that maps answers (or utterances) onto questions! Et Voila!

Lets show how this is done with SQUAD2.0 which we can download from [here](https://rajpurkar.github.io/SQuAD-explorer/dataset/train-v2.0.json)

The SQUAD dataset comes as a json with the following keys: 

```python
from enum import Enum

class SquadDataKeys(Enum):
    DATA = "data"
    TITLE = "title"
    PARAGRAPH = "paragraphs"
    CONTEXT = "context"
    QUESTIONANSWER = "qas"
    QUESTION = "question"
```

We can load in the json and iterate through it to extract the titles (as our short utterances) and the contexts (as our long utterances) and the questions that go with them:

```python
from json import loads

def load_squad_data(path_to_file:str) -> List[Tuple[str,str]]:
    with open(path_to_file) as json_file:
        squad2_data = loads(json_file.read())
        for topic_data in squad2_data[SquadDataKeys.DATA.value]:
            short_context = topic_data[SquadDataKeys.TITLE.value]
            for paragraph in topic_data[SquadDataKeys.PARAGRAPH.value]:
                long_context = paragraph[SquadDataKeys.CONTEXT.value]
                for index,question_data in enumerate(paragraph[SquadDataKeys.QUESTIONANSWER.value]):
                    question = question_data[SquadDataKeys.QUESTION.value]
                    if index == 0: yield (short_context, question)
                    yield (long_context, question)
```

![](img/squad/example_questions.png)

Now we train the model as before (modifying the train function to take in both questions, which we use for outputs as before, and question_contexts, which we use to extract the semantic vectors now, previously taken from the questions as well):

```python
contexts,questions = zip(*load_squad_data("data/train-v2.0.json"))

train(question_contexts=contexts,questions=questions,batch_size=6,epochs=100,save_to_file_path="lstm_squad2.hdf5") 

```

![](img/squad/training.png)

We can compare some randomly generated text from the semi-trained squad model below (also shown alongside the semi-trained quora model):

![](img/squad/semi_trained_example.png)

It also happens we have an even better dataset which was actually designed to provides utterances and then questions (well actually it was designed to simulate realistic conversations, but it so happens to have questions within the conversations).  This dataset is `DailyDialog` which we can download from [here](https://aclanthology.org/attachments/I17-1099.Datasets.zip)

This dataset also provides dialogue_act labels which we can use to identify the questions automatically and extract them along with the the utterances that precede it to form our training data:

```python
def load_daily_dialog(path_to_file:str) -> List[Tuple[str,str]]:
    with open(f"{path_to_file}/{DailyDialogKeys.ACT_FILENAME.value}") as dialogue_act_file:
        dialogue_acts = dialogue_act_file.readlines()

    with open(f"{path_to_file}/{DailyDialogKeys.DIALOGUE_FILENAME.value}") as dialogue_file:
        dialogues = dialogue_file.readlines()

    for line_dialogue,line_acts in zip(dialogues,dialogue_acts):
        question_in_dialogue = DailyDialogKeys.QUESTION_ACT.value in line_acts[1:]
        if question_in_dialogue:
            utterances = line_dialogue.split(DailyDialogKeys.END_OF_UTTERANCE.value)[:-1]
            question_act_indexes = [
                position for position, act_index in enumerate(line_acts.strip().split()) \
                if position > 0 and act_index == DailyDialogKeys.QUESTION_ACT.value
            ]
            for question_index in question_act_indexes:
                question = utterances[question_index]
                context = utterances[:question_index]
                yield (' '.join(context),question)

class DailyDialogKeys(Enum):
    QUESTION_ACT = "2"
    END_OF_UTTERANCE = "__eou__"
    ACT_FILENAME = "dialogues_act.txt"
    DIALOGUE_FILENAME = "dialogues_text.txt"
```

![](img/dailydialog/example_question1.png)

![](img/dailydialog/example_question2.png)

![](img/dailydialog/example_question3.png)


We can now train our final model on this dataset 

```python
contexts,questions = zip(*load_daily_dialog("data/daily_dialog"))
model.train(question_contexts=contexts,questions=questions,batch_size=6,epochs=100,save_to_file_path="lstm_dailydialog.hdf5") 
```

(and this time we will wait for it to finish training!)

![](img/dailydialog/epoch2.png)

we wait...

![](img/dailydialog/epoch18.png)

and wait ...is that "how much would you steal?" i see? and ..."that all fart"!!! (lol)

![](img/dailydialog/epoch39.png)

...some sensible questions forming now

![](img/dailydialog/epoch43.png)

...albeit slightly short, safe questions

![](img/dailydialog/epoch46.jpeg)

...if we test out the model after 66 epochs, the responses tend toward being short, safe and generic - although far more interesting than the baseline and albert questions (which is impressive given that these questions are formed purely at the character level!)

![](img/dailydialog/semi_trained_example.png)

![](img/dailydialog/semi_trained_example2.png)

I've included the trained model weights `lstm_dailydialog.hdf5` so that the model can be loaded and played around with.  Enjoy!

---

The above is also provided in this Colab Notebook if you wish to follow along

---

# Deployment:

## How would you put this module into production and make it accessible to a set of clients? 
- This requires it to be hosted on a server, or via other serverless means (e.g. Amazon AWS, Dynamo DB, etc) 

- the clients would then send jsonified utterances to, and receive questions back from, the hosted model - via a specified endpoint (i.e. using API Gateway, )

## What are the challenges in doing so? 
- The model's inference time should not be too long or it may need optimising first

- the model would only be used for inference when on the server (to retrain it, would require the newly trained model to be uploaded - which can cause disruptions to the clients unless there are parallel versions running which allow for silent changes)

- If the model requires many dependencies and specific versioning to run properly, it may be safer to package it in a Flask container where the environment is controlled and isolated

- if there are many users querying the model at the same time, there should be mechanisms in place (e.g. elastic beanstalk ) to automatically scale up the model instances 

- If the model takes a relatively long time to instantiate and warm up, it may be better to keep the server running (however this can be expensive and potentially wasteful as noone may be using it and yet it is still running) 