# sequence2sequence-translator
English-to-Hindi Neural Machine Translation

This project demonstrates a Sequence-to-Sequence (Seq2Seq) translation pipeline using the Helsinki-NLP/opus-mt-en-hi model from Hugging Face Transformers. The model translates English text into Hindi using TensorFlow.

Overview

The notebook sets up a translation model fine-tuned on the IIT Bombay English-Hindi Parallel Corpus. It uses the Hugging Face datasets, transformers, and sacrebleu libraries for dataset loading, tokenization, training, and evaluation.

Requirements
!nvidia-smi
!pip install datasets transformers[sentencepiece] sacrebleu -q

Libraries Used

transformers

datasets

tensorflow

sacrebleu

Model and Dataset

Model checkpoint: Helsinki-NLP/opus-mt-en-hi

Dataset: cfilt/iitb-english-hindi

Workflow

Load dataset

raw_dataset = load_dataset("cfilt/iitb-english-hindi")


Load tokenizer and model

tokenizer = AutoTokenizer.from_pretrained("Helsinki-NLP/opus-mt-en-hi")
model = TFAutoModelForSeq2SeqLM.from_pretrained("Helsinki-NLP/opus-mt-en-hi")


Preprocess data
Input and target sequences are tokenized and truncated to a maximum length of 128.

Prepare TensorFlow datasets
The processed data is batched using DataCollatorForSeq2Seq.

Compile and train

optimizer = AdamWeightDecay(learning_rate=2e-5, weight_decay_rate=0.01)
model.compile(optimizer=optimizer)
model.fit(train_dataset, validation_data=validation_dataset, epochs=1)


Save and reload model

model.save_pretrained("tf_model/")


Generate translation

input_text = "bibek villager"
tokenized = tokenizer([input_text], return_tensors='np')
out = model.generate(**tokenized, max_length=128)
print(tokenizer.decode(out[0], skip_special_tokens=True))

Example Output
Input: "bibek villager"
Output: "बिबेक ग्रामीण"

Notes

Modify max_input_length and max_target_length for longer sentences.

Use num_train_epochs > 1 for better fine-tuning results.

Evaluation can be extended using BLEU or SacreBLEU metrics.
